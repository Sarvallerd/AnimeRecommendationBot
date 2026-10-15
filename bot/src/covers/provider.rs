//! Provider-specific request and response contracts for optional cover lookup.
use crate::catalog::MalId;
use reqwest::{Client, Url};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Provider {
    Jikan,
    AniList,
}

impl Provider {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Jikan => "jikan",
            Self::AniList => "anilist",
        }
    }

    fn image_host(self) -> &'static str {
        match self {
            Self::Jikan => "cdn.myanimelist.net",
            Self::AniList => "s4.anilist.co",
        }
    }
}

// Absurd Retry-After values cannot be represented by every platform's Instant.
pub(super) const MAX_COOLDOWN: Duration = Duration::from_secs(365 * 24 * 60 * 60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GraphqlError {
    RateLimited,
    Missing,
    Transient,
}

pub(super) fn graphql_error(parsed: &serde_json::Value) -> Option<GraphqlError> {
    let errors = parsed.get("errors")?;
    let Some(errors) = errors.as_array() else {
        return Some(GraphqlError::Transient);
    };
    if errors.is_empty() {
        return None;
    }
    let statuses: Vec<_> = errors
        .iter()
        .map(|error| {
            error.get("status").and_then(|status| {
                status
                    .as_u64()
                    .or_else(|| status.as_str().and_then(|value| value.parse().ok()))
            })
        })
        .collect();
    if statuses.contains(&Some(429)) {
        Some(GraphqlError::RateLimited)
    } else if statuses.iter().all(|status| *status == Some(404)) {
        Some(GraphqlError::Missing)
    } else {
        Some(GraphqlError::Transient)
    }
}

fn valid_image_url(value: &str, host: &str) -> Option<Url> {
    if value.len() > 2048 {
        return None;
    }
    let url = Url::parse(value).ok()?;
    if url.scheme() != "https"
        || url.host_str() != Some(host)
        || !url.username().is_empty()
        || url.password().is_some()
        || !matches!(url.port(), None | Some(443))
    {
        return None;
    }
    Some(url)
}

impl Provider {
    pub(super) fn request(
        self,
        client: &Client,
        base: &Url,
        id: MalId,
    ) -> Option<reqwest::RequestBuilder> {
        Some(match self {
            Provider::Jikan => client.get(base.join(&id.to_string()).ok()?),
            Provider::AniList => client.post(base.clone()).json(&serde_json::json!({
                "query": "query ($id: Int!) { Media(idMal: $id, type: ANIME) { idMal coverImage { large } } }",
                "variables": {"id": id}
            })),
        })
    }

    pub(super) fn parse_cover(
        self,
        id: MalId,
        parsed: &serde_json::Value,
    ) -> Result<Option<Url>, &'static str> {
        let image = match self {
            Provider::Jikan => {
                if parsed
                    .pointer("/data/mal_id")
                    .and_then(serde_json::Value::as_i64)
                    != Some(i64::from(id))
                {
                    return Err("id_mismatch");
                }
                parsed.pointer("/data/images/jpg/image_url")
            }
            Provider::AniList => {
                let data = parsed
                    .get("data")
                    .and_then(serde_json::Value::as_object)
                    .ok_or("malformed_data")?;
                let media = data.get("Media").ok_or("malformed_media")?;
                if media.is_null() {
                    return Ok(None);
                }
                if media.get("idMal").and_then(serde_json::Value::as_i64) != Some(i64::from(id)) {
                    return Err("id_mismatch");
                }
                media.pointer("/coverImage/large")
            }
        };
        let Some(image) = image else {
            return Ok(None);
        };
        if image.is_null() {
            return Ok(None);
        }
        let text = image.as_str().ok_or("malformed_image")?;
        if text.is_empty() {
            return Ok(None);
        }
        valid_image_url(text, self.image_host())
            .map(Some)
            .ok_or("invalid_image_url")
    }

    pub(super) fn rate_limit_duration(self, headers: &reqwest::header::HeaderMap) -> Duration {
        let mut delay = Duration::from_secs(60);
        if let Some(retry) = headers
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
        {
            let parsed = retry
                .parse::<u64>()
                .ok()
                .map(Duration::from_secs)
                .or_else(|| {
                    httpdate::parse_http_date(retry)
                        .ok()
                        .and_then(|date| date.duration_since(std::time::SystemTime::now()).ok())
                });
            if let Some(parsed) = parsed {
                delay = delay.max(parsed);
            }
        }
        if self == Provider::AniList {
            if let Some(reset) = headers
                .get("X-RateLimit-Reset")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .and_then(|seconds| {
                    std::time::SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(seconds))
                })
                .and_then(|time| time.duration_since(std::time::SystemTime::now()).ok())
            {
                delay = delay.max(reset);
            }
        }
        delay.min(MAX_COOLDOWN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn urls_are_restricted_to_the_image_cdn() {
        assert!(valid_image_url(
            "https://cdn.myanimelist.net/images/anime/1.jpg",
            Provider::Jikan.image_host()
        )
        .is_some());
        for url in [
            "http://cdn.myanimelist.net/x",
            "https://other.example/x",
            "https://user@cdn.myanimelist.net/x",
            "https://cdn.myanimelist.net:444/x",
        ] {
            assert!(valid_image_url(url, Provider::Jikan.image_host()).is_none());
        }
        assert!(valid_image_url(
            "https://s4.anilist.co/cover.jpg",
            Provider::AniList.image_host()
        )
        .is_some());
        for url in [
            "http://s4.anilist.co/cover.jpg".to_owned(),
            "https://s4.anilist.co.evil.example/cover.jpg".to_owned(),
            "https://user:pass@s4.anilist.co/cover.jpg".to_owned(),
            "https://s4.anilist.co:444/cover.jpg".to_owned(),
            format!("https://s4.anilist.co/{}", "x".repeat(2048)),
        ] {
            assert!(valid_image_url(&url, Provider::AniList.image_host()).is_none());
        }
    }
}
