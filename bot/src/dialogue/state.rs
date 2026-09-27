use crate::{catalog::MalId, db::RequestId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimeIntent {
    Recommend,
    Rate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryContext {
    pub request_id: RequestId,
    pub raw_query: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedSelection {
    pub query: QueryContext,
    pub seed_mal_id: MalId,
    pub bundle_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum State {
    #[default]
    Idle,
    AwaitingQuery {
        intent: AnimeIntent,
    },
    ChoosingAnime {
        intent: AnimeIntent,
        query: QueryContext,
        candidates: Vec<MalId>,
    },
    Selected {
        intent: AnimeIntent,
        selection: ResolvedSelection,
    },
    AwaitingFeedback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Actor {
    pub chat_id: i64,
    pub user_id: i64,
}
