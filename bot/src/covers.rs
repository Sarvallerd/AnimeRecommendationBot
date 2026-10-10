//! Optional Jikan cover lookup. Only a positive MAL ID leaves this process.
use crate::catalog::MalId;
use reqwest::{Client, Url};
use std::{
    collections::{HashMap, VecDeque},
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

pub type CoverFuture<'a> = Pin<Box<dyn Future<Output = Option<Url>> + Send + 'a>>;

pub trait CoverProvider: Send + Sync {
    fn lookup(&self, mal_id: MalId) -> CoverFuture<'_>;
}

pub struct NoCovers;
impl CoverProvider for NoCovers {
    fn lookup(&self, _mal_id: MalId) -> CoverFuture<'_> {
        Box::pin(async { None })
    }
}

#[derive(Clone)]
struct Cached {
    value: Option<Url>,
    until: Instant,
}

struct Cache {
    entries: HashMap<MalId, Cached>,
    order: VecDeque<MalId>,
    cooldown_until: Option<Instant>,
}

pub struct JikanCovers {
    client: Client,
    base: Url,
    state: Mutex<Cache>,
    next_start: Mutex<Instant>,
    capacity: usize,
    interval: Duration,
}

// Absurd Retry-After values cannot be represented by every platform's Instant.
const MAX_COOLDOWN: Duration = Duration::from_secs(365 * 24 * 60 * 60);

impl JikanCovers {
    pub fn new() -> Result<Self, reqwest::Error> {
        Self::build(
            Url::parse("https://api.jikan.moe/v4/anime/").expect("fixed URL"),
            1024,
            Duration::from_millis(1100),
        )
    }

    fn build(base: Url, capacity: usize, interval: Duration) -> Result<Self, reqwest::Error> {
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_millis(1400))
            .user_agent("AnimeRecommendationBot/1.0")
            .build()?;
        Ok(Self {
            client,
            base,
            state: Mutex::new(Cache {
                entries: HashMap::new(),
                order: VecDeque::new(),
                cooldown_until: None,
            }),
            next_start: Mutex::new(Instant::now()),
            capacity: capacity.max(1),
            interval,
        })
    }

    async fn cached(&self, id: MalId) -> Option<Option<Url>> {
        let state = self.state.lock().await;
        let cached = state
            .entries
            .get(&id)
            .filter(|entry| entry.until > Instant::now())
            .map(|entry| entry.value.clone());
        if cached.is_some() {
            return cached;
        }
        if state
            .cooldown_until
            .is_some_and(|until| until > Instant::now())
        {
            return Some(None);
        }
        None
    }

    async fn store(&self, id: MalId, value: Option<Url>, ttl: Duration) {
        let mut state = self.state.lock().await;
        state.order.retain(|old| *old != id);
        state.order.push_back(id);
        state.entries.insert(
            id,
            Cached {
                value,
                until: Instant::now() + ttl,
            },
        );
        while state.entries.len() > self.capacity {
            if let Some(old) = state.order.pop_front() {
                state.entries.remove(&old);
            } else {
                break;
            }
        }
    }

    async fn cooldown(&self, duration: Duration) {
        let mut state = self.state.lock().await;
        let now = Instant::now();
        let until = now.checked_add(duration.min(MAX_COOLDOWN)).unwrap_or(now);
        if state.cooldown_until.is_none_or(|old| old < until) {
            state.cooldown_until = Some(until);
        }
    }

    async fn lookup_inner(&self, id: MalId) -> Option<Url> {
        if let Some(hit) = self.cached(id).await {
            return hit;
        }
        let mut next = self.next_start.lock().await;
        let now = Instant::now();
        if *next > now {
            tokio::time::sleep(*next - now).await;
        }
        if let Some(hit) = self.cached(id).await {
            return hit;
        }
        *next = Instant::now() + self.interval;
        // Keep the gate until the response is cached or a cooldown is set. A
        // second lookup for the same ID must not race an unfinished request.
        let endpoint = self.base.join(&id.to_string()).ok()?;
        let response = match self.client.get(endpoint).send().await {
            Ok(response) => response,
            Err(_) => {
                self.cooldown(Duration::from_secs(30)).await;
                return None;
            }
        };
        let status = response.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let duration = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| {
                    value
                        .parse::<u64>()
                        .ok()
                        .map(Duration::from_secs)
                        .or_else(|| {
                            httpdate::parse_http_date(value).ok().and_then(|date| {
                                date.duration_since(std::time::SystemTime::now()).ok()
                            })
                        })
                })
                .map(|duration| duration.max(Duration::from_secs(60)))
                .unwrap_or(Duration::from_secs(60));
            self.cooldown(duration).await;
            return None;
        }
        if status == reqwest::StatusCode::NOT_FOUND {
            self.store(id, None, Duration::from_secs(600)).await;
            return None;
        }
        if !status.is_success() {
            if status.is_server_error() {
                self.cooldown(Duration::from_secs(30)).await;
            } else {
                self.store(id, None, Duration::from_secs(600)).await;
            }
            return None;
        }
        let mut response = response;
        let mut body = Vec::new();
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) if body.len() + chunk.len() <= 256 * 1024 => {
                    body.extend_from_slice(&chunk)
                }
                Ok(None) => break,
                _ => {
                    self.cooldown(Duration::from_secs(30)).await;
                    return None;
                }
            }
        }
        let parsed: serde_json::Value = match serde_json::from_slice(&body) {
            Ok(parsed) => parsed,
            Err(_) => {
                self.cooldown(Duration::from_secs(30)).await;
                return None;
            }
        };
        if parsed
            .pointer("/data/mal_id")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(id))
        {
            self.cooldown(Duration::from_secs(30)).await;
            return None;
        }
        let value = parsed
            .pointer("/data/images/jpg/image_url")
            .and_then(serde_json::Value::as_str)
            .and_then(valid_image_url);
        self.store(
            id,
            value.clone(),
            if value.is_some() {
                Duration::from_secs(86400)
            } else {
                Duration::from_secs(600)
            },
        )
        .await;
        value
    }
}

impl CoverProvider for JikanCovers {
    fn lookup(&self, mal_id: MalId) -> CoverFuture<'_> {
        Box::pin(async move {
            if mal_id <= 0 {
                return None;
            }
            match tokio::time::timeout(Duration::from_millis(1500), self.lookup_inner(mal_id)).await
            {
                Ok(value) => value,
                Err(_) => {
                    self.cooldown(Duration::from_secs(30)).await;
                    None
                }
            }
        })
    }
}

fn valid_image_url(value: &str) -> Option<Url> {
    if value.len() > 2048 {
        return None;
    }
    let url = Url::parse(value).ok()?;
    if url.scheme() != "https"
        || url.host_str() != Some("cdn.myanimelist.net")
        || !url.username().is_empty()
        || url.password().is_some()
        || !matches!(url.port(), None | Some(443))
    {
        return None;
    }
    Some(url)
}

pub fn disabled() -> Arc<dyn CoverProvider> {
    Arc::new(NoCovers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::Mutex as StdMutex,
        thread,
    };

    fn serve_once(
        status: &str,
        body: Vec<u8>,
        headers: &str,
        delay: Duration,
    ) -> (Url, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = Url::parse(&format!(
            "http://{}/v4/anime/",
            listener.local_addr().unwrap()
        ))
        .unwrap();
        let status = status.to_owned();
        let headers = headers.to_owned();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 2048];
            let _ = stream.read(&mut request);
            thread::sleep(delay);
            let head = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        });
        (base, handle)
    }
    #[test]
    fn urls_are_restricted_to_the_image_cdn() {
        assert!(valid_image_url("https://cdn.myanimelist.net/images/anime/1.jpg").is_some());
        for url in [
            "http://cdn.myanimelist.net/x",
            "https://other.example/x",
            "https://user@cdn.myanimelist.net/x",
            "https://cdn.myanimelist.net:444/x",
        ] {
            assert!(valid_image_url(url).is_none());
        }
    }

    #[tokio::test]
    async fn cache_ttl_and_capacity_are_bounded() {
        let covers = JikanCovers::build(
            Url::parse("https://api.jikan.moe/v4/anime/").unwrap(),
            2,
            Duration::ZERO,
        )
        .unwrap();
        let image = valid_image_url("https://cdn.myanimelist.net/a.jpg").unwrap();
        covers
            .store(1, Some(image.clone()), Duration::from_secs(2))
            .await;
        covers.store(2, None, Duration::from_secs(2)).await;
        assert_eq!(covers.cached(1).await, Some(Some(image)));
        assert_eq!(covers.cached(2).await, Some(None));
        covers.store(3, None, Duration::from_secs(2)).await;
        assert_eq!(covers.cached(1).await, None);
        covers.store(4, None, Duration::from_millis(1)).await;
        tokio::time::sleep(Duration::from_millis(3)).await;
        assert_eq!(covers.cached(4).await, None);
    }

    #[tokio::test]
    async fn concurrent_lookup_coalesces_and_paces_different_ids() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = Url::parse(&format!(
            "http://{}/v4/anime/",
            listener.local_addr().unwrap()
        ))
        .unwrap();
        let starts = Arc::new(StdMutex::new(Vec::<Instant>::new()));
        let recorded = starts.clone();
        let thread = thread::spawn(move || {
            for request in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut data = [0u8; 2048];
                let n = stream.read(&mut data).unwrap();
                let line = String::from_utf8_lossy(&data[..n]);
                let id = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .rsplit('/')
                    .next()
                    .unwrap()
                    .parse::<i32>()
                    .unwrap();
                recorded.lock().unwrap().push(Instant::now());
                if request == 0 {
                    // The first response arrives after the pacing interval.
                    thread::sleep(Duration::from_millis(1150));
                }
                let body = format!("{{\"data\":{{\"mal_id\":{id},\"images\":{{\"jpg\":{{\"image_url\":\"https://cdn.myanimelist.net/{id}.jpg\"}}}}}}}}");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let covers = JikanCovers::build(base, 2, Duration::from_millis(1100)).unwrap();
        let (a, b) = tokio::join!(covers.lookup(10), covers.lookup(10));
        assert_eq!(a, b);
        assert!(a.is_some());
        assert!(covers.lookup(11).await.is_some());
        thread.join().unwrap();
        let times = starts.lock().unwrap();
        assert_eq!(times.len(), 2);
        assert!(times[1].duration_since(times[0]) >= Duration::from_millis(1100));
    }

    #[tokio::test]
    async fn rate_limit_and_bad_responses_cool_down_globally() {
        let cases = [
            (
                "429 Too Many Requests",
                b"".to_vec(),
                "Retry-After: 120\r\n",
                110,
            ),
            (
                "429 Too Many Requests",
                b"".to_vec(),
                "Retry-After: 18446744073709551615\r\n",
                300 * 24 * 60 * 60,
            ),
            ("200 OK", b"not-json".to_vec(), "", 20),
            ("200 OK", br#"{"data":{"mal_id":11}}"#.to_vec(), "", 20),
            ("200 OK", vec![b'x'; 256 * 1024 + 1], "", 20),
        ];
        for (status, body, headers, min_seconds) in cases {
            let (base, server) = serve_once(status, body, headers, Duration::ZERO);
            let covers = JikanCovers::build(base, 2, Duration::ZERO).unwrap();
            assert_eq!(covers.lookup(10).await, None);
            server.join().unwrap();
            let until = covers.state.lock().await.cooldown_until.unwrap();
            assert!(until.duration_since(Instant::now()) >= Duration::from_secs(min_seconds));
            assert_eq!(covers.lookup(11).await, None);
        }
    }

    #[tokio::test]
    async fn missing_or_invalid_cdn_url_is_cached_as_absent() {
        for image in [
            "http://cdn.myanimelist.net/a.jpg",
            "https://evil.example/a.jpg",
        ] {
            let body = format!("{{\"data\":{{\"mal_id\":10,\"images\":{{\"jpg\":{{\"image_url\":\"{image}\"}}}}}}}}").into_bytes();
            let (base, server) = serve_once("200 OK", body, "", Duration::ZERO);
            let covers = JikanCovers::build(base, 2, Duration::ZERO).unwrap();
            assert_eq!(covers.lookup(10).await, None);
            server.join().unwrap();
            assert_eq!(covers.cached(10).await, Some(None));
        }
    }

    #[tokio::test]
    async fn slow_lookup_is_bounded_and_cools_down() {
        let (base, server) = serve_once("200 OK", b"{}".to_vec(), "", Duration::from_millis(1600));
        let covers = JikanCovers::build(base, 2, Duration::ZERO).unwrap();
        let start = Instant::now();
        assert_eq!(covers.lookup(10).await, None);
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(covers.state.lock().await.cooldown_until.unwrap() > Instant::now());
        server.join().unwrap();
    }
}
