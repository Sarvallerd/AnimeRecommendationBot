use super::*;
type JikanCovers = UpstreamCovers;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex as StdMutex,
    },
    thread,
};

#[derive(Clone)]
struct Captured {
    method: String,
    path: String,
    body: String,
}

struct FixtureReply {
    status: &'static str,
    headers: String,
    body: Vec<u8>,
    delay: Duration,
}

impl FixtureReply {
    fn json(body: String) -> Self {
        Self {
            status: "200 OK",
            headers: String::new(),
            body: body.into_bytes(),
            delay: Duration::ZERO,
        }
    }
    fn status(status: &'static str) -> Self {
        Self {
            status,
            headers: String::new(),
            body: Vec::new(),
            delay: Duration::ZERO,
        }
    }
}

struct FixtureServer {
    url: Url,
    calls: Arc<StdMutex<Vec<Captured>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl FixtureServer {
    fn new(handler: impl Fn(&Captured) -> FixtureReply + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let recorded = calls.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let thread = thread::spawn(move || {
            while !stopping.load(Ordering::SeqCst) {
                let (mut stream, _) = match listener.accept() {
                    Ok(value) => value,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    let n = stream.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&chunk[..n]);
                    if let Some(head_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..head_end]);
                        let length = head
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|value| value.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= head_end + 4 + length {
                            break;
                        }
                    }
                }
                let request = String::from_utf8_lossy(&bytes);
                let first = request.split_whitespace().collect::<Vec<_>>();
                let captured = Captured {
                    method: first.first().unwrap_or(&"").to_string(),
                    path: first.get(1).unwrap_or(&"").to_string(),
                    body: request
                        .split_once("\r\n\r\n")
                        .map_or("", |(_, body)| body)
                        .to_owned(),
                };
                recorded.lock().unwrap().push(captured.clone());
                let reply = handler(&captured);
                thread::sleep(reply.delay);
                let head = format!(
                    "HTTP/1.1 {}\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
                    reply.status,
                    reply.body.len(),
                    reply.headers
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&reply.body);
            }
        });
        Self {
            url,
            calls,
            stop,
            thread: Some(thread),
        }
    }
    fn calls(&self) -> Vec<Captured> {
        self.calls.lock().unwrap().clone()
    }
}

impl Drop for FixtureServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

fn jikan_fixture(id: MalId) -> FixtureReply {
    FixtureReply::json(format!("{{\"data\":{{\"mal_id\":{id},\"images\":{{\"jpg\":{{\"image_url\":\"https://cdn.myanimelist.net/{id}.jpg\"}}}}}}}}"))
}

fn anilist_fixture(id: MalId) -> FixtureReply {
    FixtureReply::json(format!("{{\"data\":{{\"Media\":{{\"idMal\":{id},\"coverImage\":{{\"large\":\"https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/{id}.jpg\"}}}}}}}}"))
}

fn local_remote(
    jikan: &FixtureServer,
    anilist: &FixtureServer,
    jikan_interval: Duration,
    anilist_interval: Duration,
) -> RemoteCovers {
    RemoteCovers::build(
        jikan.url.join("v4/anime/").unwrap(),
        anilist.url.join("graphql").unwrap(),
        jikan_interval,
        anilist_interval,
    )
    .unwrap()
}

#[tokio::test]
async fn composite_prefers_jikan_and_sends_only_numeric_anilist_variables() {
    let jikan = FixtureServer::new(|_| jikan_fixture(10));
    let anilist = FixtureServer::new(|_| anilist_fixture(10));
    let remote = local_remote(&jikan, &anilist, Duration::ZERO, Duration::ZERO);
    assert_eq!(
        remote.lookup(10).await.unwrap().host_str(),
        Some("cdn.myanimelist.net")
    );
    assert_eq!(
        remote.lookup(10).await.unwrap().host_str(),
        Some("cdn.myanimelist.net")
    );
    let jikan_sent = jikan.calls();
    assert_eq!(jikan_sent.len(), 1);
    assert_eq!(jikan_sent[0].method, "GET");
    assert_eq!(jikan_sent[0].path, "/v4/anime/10");
    assert!(jikan_sent[0].body.is_empty());
    assert!(anilist.calls().is_empty());

    let jikan_missing = FixtureServer::new(|_| FixtureReply::status("404 Not Found"));
    let anilist_live = FixtureServer::new(|_| anilist_fixture(10));
    let remote = local_remote(
        &jikan_missing,
        &anilist_live,
        Duration::ZERO,
        Duration::ZERO,
    );
    assert_eq!(
        remote.lookup(10).await.unwrap().host_str(),
        Some("s4.anilist.co")
    );
    let sent = anilist_live.calls();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].method, "POST");
    assert_eq!(sent[0].path, "/graphql");
    let posted: serde_json::Value = serde_json::from_str(&sent[0].body).unwrap();
    assert_eq!(posted["variables"], serde_json::json!({"id": 10}));
    assert_eq!(
        posted["query"],
        "query ($id: Int!) { Media(idMal: $id, type: ANIME) { idMal coverImage { large } } }"
    );
}

#[tokio::test]
async fn jikan_failures_fall_back_and_cached_anilist_positive_skips_cooldown() {
    for response in [
        FixtureReply::status("500 Internal Server Error"),
        FixtureReply::status("429 Too Many Requests"),
        FixtureReply::json("not-json".into()),
        FixtureReply::status("404 Not Found"),
        FixtureReply {
            delay: Duration::from_millis(1600),
            ..jikan_fixture(10)
        },
    ] {
        let response = Arc::new(response);
        let jikan = FixtureServer::new(move |_| FixtureReply {
            status: response.status,
            headers: response.headers.clone(),
            body: response.body.clone(),
            delay: response.delay,
        });
        let anilist = FixtureServer::new(|_| anilist_fixture(10));
        let remote = local_remote(&jikan, &anilist, Duration::ZERO, Duration::ZERO);
        assert_eq!(
            remote.lookup(10).await.unwrap().host_str(),
            Some("s4.anilist.co")
        );
        assert_eq!(
            remote.lookup(10).await.unwrap().host_str(),
            Some("s4.anilist.co")
        );
        assert_eq!(jikan.calls().len(), 1);
        assert_eq!(anilist.calls().len(), 1);
    }
}

#[tokio::test]
async fn anilist_rejects_errors_mismatch_malformed_and_bad_images() {
    let cases = [
        (
            r#"{"data":{"Media":{"idMal":10,"coverImage":{"large":"https://s4.anilist.co/a.jpg"}}},"errors":[{"message":"no"}]}"#,
            true,
        ),
        (
            r#"{"data":{"Media":{"idMal":11,"coverImage":{"large":"https://s4.anilist.co/a.jpg"}}}}"#,
            true,
        ),
        (r#"{"data":{}}"#, true),
        (r#"{"data":{"Media":null}}"#, false),
        (
            r#"{"data":{"Media":{"idMal":10,"coverImage":{"large":null}}}}"#,
            false,
        ),
        (
            r#"{"data":{"Media":{"idMal":10,"coverImage":{"large":"https://evil.example/a.jpg"}}}}"#,
            true,
        ),
    ];
    for (body, transient) in cases {
        let jikan = FixtureServer::new(|_| FixtureReply::status("404 Not Found"));
        let body = body.to_owned();
        let anilist = FixtureServer::new(move |_| FixtureReply::json(body.clone()));
        let remote = local_remote(&jikan, &anilist, Duration::ZERO, Duration::ZERO);
        assert_eq!(remote.lookup(10).await, None);
        assert_eq!(
            remote.anilist.state.lock().await.cooldown_until.is_some(),
            transient
        );
        assert_eq!(
            remote.anilist.state.lock().await.entries.contains_key(&10),
            !transient
        );
    }
}

#[tokio::test]
async fn graphql_error_statuses_override_partial_data() {
    let partial =
        r#""data":{"Media":{"idMal":10,"coverImage":{"large":"https://s4.anilist.co/10.jpg"}}}"#;
    for (errors, expected_cooldown, expected_negative) in [
        (r#"[{"status":429},{"status":404}]"#, 100, false),
        (r#"[{"status":404}]"#, 0, true),
        (r#"[{"status":404},{"status":500}]"#, 20, false),
    ] {
        let body = format!("{{{partial},\"errors\":{errors}}}");
        let anilist = FixtureServer::new(move |_| FixtureReply {
            headers: "Retry-After: 120\r\n".to_owned(),
            ..FixtureReply::json(body.clone())
        });
        let upstream = UpstreamCovers::build_with(
            Provider::AniList,
            anilist.url.join("graphql").unwrap(),
            2,
            Duration::ZERO,
            Duration::from_millis(2300),
            Duration::from_millis(2000),
        )
        .unwrap();
        assert_eq!(upstream.lookup(10).await, None);
        let state = upstream.state.lock().await;
        assert_eq!(state.entries.contains_key(&10), expected_negative);
        if expected_cooldown == 0 {
            assert!(state.cooldown_until.is_none());
        } else {
            assert!(
                state.cooldown_until.unwrap().duration_since(Instant::now())
                    >= Duration::from_secs(expected_cooldown)
            );
        }
        drop(state);
        assert_eq!(upstream.lookup(10).await, None);
        assert_eq!(anilist.calls().len(), 1);
    }
}

#[tokio::test]
async fn upstream_cooldowns_are_independent_and_both_fail_cleanly() {
    let jikan = FixtureServer::new(|_| FixtureReply::status("500 Internal Server Error"));
    let anilist = FixtureServer::new(|request| {
        let posted: serde_json::Value = serde_json::from_str(&request.body).unwrap();
        anilist_fixture(posted["variables"]["id"].as_i64().unwrap() as MalId)
    });
    let remote = local_remote(&jikan, &anilist, Duration::ZERO, Duration::ZERO);
    for id in [10, 11] {
        assert_eq!(
            remote.lookup(id).await.unwrap().host_str(),
            Some("s4.anilist.co")
        );
    }
    assert_eq!(jikan.calls().len(), 1);
    assert_eq!(anilist.calls().len(), 2);
    assert!(remote.jikan.state.lock().await.cooldown_until.is_some());
    assert!(remote.anilist.state.lock().await.cooldown_until.is_none());

    let jikan = FixtureServer::new(|_| FixtureReply::status("500 Internal Server Error"));
    let anilist = FixtureServer::new(|_| FixtureReply::status("500 Internal Server Error"));
    let remote = local_remote(&jikan, &anilist, Duration::ZERO, Duration::ZERO);
    assert_eq!(remote.lookup(10).await, None);
    assert_eq!(jikan.calls().len(), 1);
    assert_eq!(anilist.calls().len(), 1);
    assert!(remote.jikan.state.lock().await.cooldown_until.is_some());
    assert!(remote.anilist.state.lock().await.cooldown_until.is_some());
}

#[tokio::test]
async fn anilist_rate_headers_and_missing_responses() {
    let reset = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 180;
    let anilist = FixtureServer::new(move |_| FixtureReply {
        headers: format!("Retry-After: 120\r\nX-RateLimit-Reset: {reset}\r\n"),
        ..FixtureReply::status("429 Too Many Requests")
    });
    let upstream = UpstreamCovers::build_with(
        Provider::AniList,
        anilist.url.join("graphql").unwrap(),
        2,
        Duration::ZERO,
        Duration::from_millis(2300),
        Duration::from_millis(2000),
    )
    .unwrap();
    assert_eq!(upstream.lookup(10).await, None);
    assert!(
        upstream
            .state
            .lock()
            .await
            .cooldown_until
            .unwrap()
            .duration_since(Instant::now())
            >= Duration::from_secs(170)
    );

    let anilist = FixtureServer::new(move |_| FixtureReply {
        headers: format!("X-RateLimit-Remaining: 0\r\nX-RateLimit-Reset: {reset}\r\n"),
        ..anilist_fixture(10)
    });
    let upstream = UpstreamCovers::build_with(
        Provider::AniList,
        anilist.url.join("graphql").unwrap(),
        2,
        Duration::ZERO,
        Duration::from_millis(2300),
        Duration::from_millis(2000),
    )
    .unwrap();
    assert!(upstream.lookup(10).await.is_some());
    assert!(upstream.state.lock().await.cooldown_until.is_some());
    assert!(upstream.lookup(10).await.is_some());
    assert_eq!(anilist.calls().len(), 1);

    let anilist = FixtureServer::new(|_| FixtureReply::status("404 Not Found"));
    let upstream = UpstreamCovers::build_with(
        Provider::AniList,
        anilist.url.join("graphql").unwrap(),
        2,
        Duration::ZERO,
        Duration::from_millis(2300),
        Duration::from_millis(2000),
    )
    .unwrap();
    assert_eq!(upstream.lookup(10).await, None);
    assert_eq!(upstream.cached(10).await, Some(None));
    assert!(upstream.state.lock().await.cooldown_until.is_none());
    assert_eq!(upstream.lookup(10).await, None);
    assert_eq!(anilist.calls().len(), 1);
}

#[tokio::test]
async fn anilist_redirect_and_oversize_are_transient() {
    for reply in [
        FixtureReply::status("302 Found"),
        FixtureReply {
            body: vec![b'x'; 256 * 1024 + 1],
            ..anilist_fixture(10)
        },
    ] {
        let reply = Arc::new(reply);
        let anilist = FixtureServer::new(move |_| FixtureReply {
            status: reply.status,
            headers: reply.headers.clone(),
            body: reply.body.clone(),
            delay: reply.delay,
        });
        let upstream = UpstreamCovers::build_with(
            Provider::AniList,
            anilist.url.join("graphql").unwrap(),
            2,
            Duration::ZERO,
            Duration::from_millis(2300),
            Duration::from_millis(2000),
        )
        .unwrap();
        assert_eq!(upstream.lookup(10).await, None);
        assert!(upstream.state.lock().await.cooldown_until.is_some());
    }
}

#[tokio::test]
async fn anilist_paces_distinct_ids_and_coalesces_concurrent_id() {
    let anilist = FixtureServer::new(|request| {
        let posted: serde_json::Value = serde_json::from_str(&request.body).unwrap();
        let id = posted["variables"]["id"].as_i64().unwrap() as MalId;
        let mut reply = anilist_fixture(id);
        if id == 10 {
            reply.delay = Duration::from_millis(1000);
        }
        reply
    });
    let upstream = UpstreamCovers::build_with(
        Provider::AniList,
        anilist.url.join("graphql").unwrap(),
        2,
        Duration::from_millis(2100),
        Duration::from_millis(2300),
        Duration::from_millis(2000),
    )
    .unwrap();
    let (a, b) = tokio::join!(upstream.lookup(10), upstream.lookup(10));
    assert_eq!(a, b);
    assert!(a.is_some());
    let first_next_start = *upstream.next_start.lock().await;
    assert!(upstream.lookup(11).await.is_some());
    let second_next_start = *upstream.next_start.lock().await;
    let calls = anilist.calls();
    assert_eq!(calls.len(), 2);
    // These are client-side send-start timestamps plus the fixed interval.
    // A server receive timestamp includes unrelated socket/scheduler delays.
    assert!(second_next_start.duration_since(first_next_start) >= Duration::from_millis(2100));
}

#[tokio::test]
async fn anilist_slot_timeout_sends_nothing_and_sets_no_cooldown() {
    let anilist = FixtureServer::new(|_| anilist_fixture(10));
    let upstream = UpstreamCovers::build_with(
        Provider::AniList,
        anilist.url.join("graphql").unwrap(),
        2,
        Duration::from_millis(2100),
        Duration::from_millis(100),
        Duration::from_millis(2000),
    )
    .unwrap();
    *upstream.next_start.lock().await = Instant::now() + Duration::from_millis(250);
    assert_eq!(upstream.lookup(10).await, None);
    assert!(anilist.calls().is_empty());
    assert!(upstream.state.lock().await.cooldown_until.is_none());
}

#[tokio::test]
async fn nearly_used_wait_slot_still_gets_full_http_budget() {
    let anilist = FixtureServer::new(|_| FixtureReply {
        delay: Duration::from_millis(1000),
        ..anilist_fixture(10)
    });
    let upstream = UpstreamCovers::build_with(
        Provider::AniList,
        anilist.url.join("graphql").unwrap(),
        2,
        Duration::ZERO,
        Duration::from_millis(2300),
        Duration::from_millis(2000),
    )
    .unwrap();
    *upstream.next_start.lock().await = Instant::now() + Duration::from_millis(1700);
    assert!(upstream.lookup(10).await.is_some());
    assert_eq!(anilist.calls().len(), 1);
    assert!(upstream.state.lock().await.cooldown_until.is_none());
}

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
#[tokio::test]
async fn cache_ttl_and_capacity_are_bounded() {
    let covers = JikanCovers::build(
        Url::parse("https://api.jikan.moe/v4/anime/").unwrap(),
        2,
        Duration::ZERO,
    )
    .unwrap();
    let image = Url::parse("https://cdn.myanimelist.net/a.jpg").unwrap();
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
    let requests = Arc::new(AtomicUsize::new(0));
    let recorded = requests.clone();
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
            recorded.fetch_add(1, Ordering::SeqCst);
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
    let first_next_start = *covers.next_start.lock().await;
    assert!(covers.lookup(11).await.is_some());
    let second_next_start = *covers.next_start.lock().await;
    thread.join().unwrap();
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    assert!(second_next_start.duration_since(first_next_start) >= Duration::from_millis(1100));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn total_deadline_keeps_gate_until_cooldown_is_visible() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = Url::parse(&format!(
        "http://{}/v4/anime/",
        listener.local_addr().unwrap()
    ))
    .unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let seen = requests.clone();
    let server = thread::spawn(move || {
        let until = Instant::now() + Duration::from_millis(2100);
        while Instant::now() < until {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut request = [0u8; 2048];
                    let _ = stream.read(&mut request);
                    let number = seen.fetch_add(1, Ordering::SeqCst) + 1;
                    if number == 1 {
                        thread::sleep(Duration::from_millis(1600));
                    }
                    let body = br#"{"data":{"mal_id":10,"images":{"jpg":{"image_url":"https://cdn.myanimelist.net/10.jpg"}}}}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.write_all(body);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(2))
                }
                Err(error) => panic!("local fixture failed: {error}"),
            }
        }
    });
    let covers = Arc::new(JikanCovers::build(base, 2, Duration::from_millis(1100)).unwrap());
    *covers.next_start.lock().await = Instant::now() + Duration::from_millis(300);
    let first_covers = covers.clone();
    let first = tokio::spawn(async move { first_covers.lookup(10).await });
    tokio::time::sleep(Duration::from_millis(1650)).await;
    // Delay cooldown publication across the independent HTTP deadline. The first
    // lookup must still own the pacing gate while waiting for this lock.
    let state = covers.state.lock().await;
    tokio::time::sleep(Duration::from_millis(220)).await;
    assert!(covers.next_start.try_lock().is_err());
    let second_covers = covers.clone();
    let second = tokio::spawn(async move { second_covers.lookup(10).await });
    drop(state);
    assert_eq!(first.await.unwrap(), None);
    assert_eq!(second.await.unwrap(), None);
    server.join().unwrap();
    assert_eq!(requests.load(Ordering::SeqCst), 1);
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
        let body = format!(
            "{{\"data\":{{\"mal_id\":10,\"images\":{{\"jpg\":{{\"image_url\":\"{image}\"}}}}}}}}"
        )
        .into_bytes();
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
