use crate::{
    catalog::MalId,
    db::{PositionId, RequestId},
    dialogue::state::AnimeIntent,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Recommend,
    Rate,
    Feedback,
    Cancel,
    Select {
        intent: AnimeIntent,
        request_id: RequestId,
        mal_id: MalId,
    },
    AnimeScore {
        request_id: RequestId,
        score: i16,
    },
    RecommendationScore {
        position_id: PositionId,
        score: i16,
    },
}

impl Action {
    pub fn valid(&self) -> bool {
        match self {
            Self::Select {
                request_id, mal_id, ..
            } => *request_id > 0 && *mal_id > 0,
            Self::AnimeScore { request_id, score } => *request_id > 0 && (1..=10).contains(score),
            Self::RecommendationScore { position_id, score } => {
                *position_id > 0 && (0..=5).contains(score)
            }
            _ => true,
        }
    }
}

pub trait TokenSource: Send + Sync {
    fn token(&self) -> Result<String, getrandom::Error>;
}

pub struct RandomTokenSource;
impl TokenSource for RandomTokenSource {
    fn token(&self) -> Result<String, getrandom::Error> {
        token()
    }
}

pub fn token() -> Result<String, getrandom::Error> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes)?;
    let mut token = String::with_capacity(35);
    token.push_str("a1:");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        token.push(char::from(HEX[usize::from(byte >> 4)]));
        token.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    Ok(token)
}

pub fn well_formed(data: &str) -> bool {
    data.len() == 35
        && data.starts_with("a1:")
        && data.as_bytes()[3..].iter().all(u8::is_ascii_hexdigit)
        && data.as_bytes()[3..]
            .iter()
            .all(|ch| !ch.is_ascii_uppercase())
}
