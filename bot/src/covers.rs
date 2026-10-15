//! Optional Jikan and AniList cover lookup. Only a positive MAL ID leaves this process.
use crate::catalog::MalId;

mod provider;
use provider::{graphql_error, GraphqlError, Provider, MAX_COOLDOWN};
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

pub struct RemoteCovers {
    jikan: UpstreamCovers,
    anilist: UpstreamCovers,
}

struct UpstreamCovers {
    provider: Provider,
    client: Client,
    base: Url,
    state: Mutex<Cache>,
    next_start: Mutex<Instant>,
    capacity: usize,
    interval: Duration,
    slot_timeout: Duration,
}

impl RemoteCovers {
    pub fn new() -> Result<Self, reqwest::Error> {
        Self::build(
            Url::parse("https://api.jikan.moe/v4/anime/").expect("fixed URL"),
            Url::parse("https://graphql.anilist.co").expect("fixed URL"),
            Duration::from_millis(1100),
            Duration::from_millis(2100),
        )
    }

    fn build(
        jikan: Url,
        anilist: Url,
        jikan_interval: Duration,
        anilist_interval: Duration,
    ) -> Result<Self, reqwest::Error> {
        Ok(Self {
            jikan: UpstreamCovers::build_with(
                Provider::Jikan,
                jikan,
                1024,
                jikan_interval,
                Duration::from_millis(1200),
                Duration::from_millis(1400),
            )?,
            anilist: UpstreamCovers::build_with(
                Provider::AniList,
                anilist,
                1024,
                anilist_interval,
                Duration::from_millis(2300),
                Duration::from_millis(2000),
            )?,
        })
    }

    /// Local HTTP fixture entry point for integration tests; no endpoint setting is read at runtime.
    #[doc(hidden)]
    pub fn with_loopback_endpoints(jikan: Url, anilist: Url) -> Option<Self> {
        if ![&jikan, &anilist].iter().all(|url| {
            url.scheme() == "http"
                && matches!(url.host_str(), Some("127.0.0.1" | "localhost"))
                && url.username().is_empty()
                && url.password().is_none()
        }) {
            return None;
        }
        Self::build(
            jikan,
            anilist,
            Duration::from_millis(1100),
            Duration::from_millis(2100),
        )
        .ok()
    }
}

impl CoverProvider for RemoteCovers {
    fn lookup(&self, mal_id: MalId) -> CoverFuture<'_> {
        Box::pin(async move {
            if mal_id <= 0 {
                return None;
            }
            if let Some(url) = self.jikan.positive(mal_id).await {
                return Some(url);
            }
            if let Some(url) = self.anilist.positive(mal_id).await {
                return Some(url);
            }
            if let Some(url) = self.jikan.lookup(mal_id).await {
                return Some(url);
            }
            self.anilist.lookup(mal_id).await
        })
    }
}

impl UpstreamCovers {
    #[cfg(test)]
    fn build(base: Url, capacity: usize, interval: Duration) -> Result<Self, reqwest::Error> {
        Self::build_with(
            Provider::Jikan,
            base,
            capacity,
            interval,
            Duration::from_millis(1200),
            Duration::from_millis(1400),
        )
    }

    fn build_with(
        provider: Provider,
        base: Url,
        capacity: usize,
        interval: Duration,
        slot_timeout: Duration,
        http_timeout: Duration,
    ) -> Result<Self, reqwest::Error> {
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(http_timeout)
            .user_agent("AnimeRecommendationBot/1.0")
            .build()?;
        Ok(Self {
            provider,
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
            slot_timeout,
        })
    }

    async fn positive(&self, id: MalId) -> Option<Url> {
        self.state
            .lock()
            .await
            .entries
            .get(&id)
            .filter(|entry| entry.until > Instant::now())
            .and_then(|entry| entry.value.clone())
    }

    fn warn(
        &self,
        stage: &'static str,
        reason: &'static str,
        status: Option<u16>,
        elapsed: Duration,
    ) {
        if let Some(status) = status {
            log::warn!(
                "cover provider={} stage={stage} reason={reason} status={status} duration_ms={}",
                self.provider.name(),
                elapsed.as_millis()
            );
        } else {
            log::warn!(
                "cover provider={} stage={stage} reason={reason} duration_ms={}",
                self.provider.name(),
                elapsed.as_millis()
            );
        }
    }

    fn debug(&self, stage: &'static str, reason: &'static str) {
        log::debug!(
            "cover provider={} stage={stage} reason={reason}",
            self.provider.name()
        );
    }

    async fn cached(&self, id: MalId) -> Option<Option<Url>> {
        let state = self.state.lock().await;
        let cached = state
            .entries
            .get(&id)
            .filter(|entry| entry.until > Instant::now())
            .map(|entry| entry.value.clone());
        if cached.is_some() {
            self.debug("cache", "hit");
            return cached;
        }
        if state
            .cooldown_until
            .is_some_and(|until| until > Instant::now())
        {
            self.debug("cache", "cooldown");
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

    async fn wait_slot(&self, id: MalId, next: Instant) -> Option<Option<Url>> {
        if let Some(hit) = self.cached(id).await {
            return Some(hit);
        }
        let now = Instant::now();
        if next > now {
            tokio::time::sleep(next - now).await;
        }
        self.cached(id).await
    }

    async fn lookup_http(&self, id: MalId, next: &mut Instant) -> Option<Url> {
        let request = self.provider.request(&self.client, &self.base, id)?;
        let started = Instant::now();
        *next = started + self.interval;
        let response = match request.send().await {
            Ok(response) => response,
            Err(_) => {
                self.warn("http", "transport", None, started.elapsed());
                self.cooldown(Duration::from_secs(30)).await;
                return None;
            }
        };
        let status = response.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let duration = self.provider.rate_limit_duration(response.headers());
            self.warn(
                "http",
                "rate_limited",
                Some(status.as_u16()),
                started.elapsed(),
            );
            self.cooldown(duration).await;
            return None;
        }
        if status == reqwest::StatusCode::NOT_FOUND {
            self.debug("http", "not_found");
            self.store(id, None, Duration::from_secs(600)).await;
            return None;
        }
        if !status.is_success() {
            self.warn(
                "http",
                if status.is_redirection() {
                    "redirect"
                } else {
                    "status"
                },
                Some(status.as_u16()),
                started.elapsed(),
            );
            self.cooldown(Duration::from_secs(30)).await;
            return None;
        }
        if self.provider == Provider::AniList
            && response
                .headers()
                .get("X-RateLimit-Remaining")
                .is_some_and(|header| header.as_bytes() == b"0")
        {
            self.cooldown(self.provider.rate_limit_duration(response.headers()))
                .await;
        }
        let mut response = response;
        let mut body = Vec::new();
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) if body.len().saturating_add(chunk.len()) <= 256 * 1024 => {
                    body.extend_from_slice(&chunk)
                }
                Ok(None) => break,
                _ => {
                    self.warn(
                        "body",
                        "oversize_or_transport",
                        Some(status.as_u16()),
                        started.elapsed(),
                    );
                    self.cooldown(Duration::from_secs(30)).await;
                    return None;
                }
            }
        }
        let parsed: serde_json::Value = match serde_json::from_slice(&body) {
            Ok(parsed) => parsed,
            Err(_) => {
                self.warn(
                    "body",
                    "malformed",
                    Some(status.as_u16()),
                    started.elapsed(),
                );
                self.cooldown(Duration::from_secs(30)).await;
                return None;
            }
        };
        if self.provider == Provider::AniList {
            match graphql_error(&parsed) {
                Some(GraphqlError::RateLimited) => {
                    self.warn("graphql", "rate_limited", Some(429), started.elapsed());
                    self.cooldown(self.provider.rate_limit_duration(response.headers()))
                        .await;
                    return None;
                }
                Some(GraphqlError::Missing) => {
                    self.debug("graphql", "not_found");
                    self.store(id, None, Duration::from_secs(600)).await;
                    return None;
                }
                Some(GraphqlError::Transient) => {
                    self.warn("graphql", "errors", None, started.elapsed());
                    self.cooldown(Duration::from_secs(30)).await;
                    return None;
                }
                None => {}
            }
        }
        let value = match self.provider.parse_cover(id, &parsed) {
            Ok(value) => value,
            Err(reason) => {
                self.warn("body", reason, Some(status.as_u16()), started.elapsed());
                self.cooldown(Duration::from_secs(30)).await;
                return None;
            }
        };
        self.debug(
            "body",
            if value.is_some() {
                "success"
            } else {
                "missing"
            },
        );
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

impl CoverProvider for UpstreamCovers {
    fn lookup(&self, mal_id: MalId) -> CoverFuture<'_> {
        Box::pin(async move {
            if mal_id <= 0 {
                return None;
            }
            let deadline = tokio::time::Instant::now() + self.slot_timeout;
            if let Ok(Some(hit)) = tokio::time::timeout_at(deadline, self.cached(mal_id)).await {
                return hit;
            }
            let mut gate = match tokio::time::timeout_at(deadline, self.next_start.lock()).await {
                Ok(gate) => gate,
                // This lookup sent nothing; the holder remains responsible for
                // caching its response or establishing a cooldown.
                Err(_) => return None,
            };
            match tokio::time::timeout_at(deadline, self.wait_slot(mal_id, *gate)).await {
                Ok(Some(hit)) => return hit,
                Ok(None) => {}
                Err(_) => {
                    self.debug("gate", "slot_timeout");
                    return None;
                }
            }
            // Reqwest's full request timeout starts here, after the waiting slot.
            // Keep the gate until the response is cached or cooldown is visible.
            self.lookup_http(mal_id, &mut gate).await
        })
    }
}

pub fn disabled() -> Arc<dyn CoverProvider> {
    Arc::new(NoCovers)
}

#[cfg(test)]
mod tests;
