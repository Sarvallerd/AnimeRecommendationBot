use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::sync::Mutex as AsyncMutex;

use super::{
    callback::{self, Action, RandomTokenSource, TokenSource},
    state::{Actor, ResolvedSelection, State},
};
use crate::db::PositionId;
use crate::db::{DeliveryInput, UserProfile};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecommendationFormat {
    Text,
    Photo,
}

#[derive(Clone, Debug)]
pub(crate) struct PendingFeedback {
    pub actor: Actor,
    pub profile: UserProfile,
    pub action_key: String,
    pub body: String,
    pub saved: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PendingRecommendationDelivery {
    pub actor: Actor,
    pub selection: ResolvedSelection,
    pub input: DeliveryInput,
    pub format: RecommendationFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackStatus {
    Pending,
    Active,
    Processing,
    Consumed,
}

#[derive(Clone, Debug)]
pub struct CallbackRecord {
    pub action: Action,
    pub generation: u64,
    pub message_id: Option<i32>,
    pub action_key: String,
    pub status: CallbackStatus,
}

pub struct Session {
    token_source: Arc<dyn TokenSource>,
    pub generation: u64,
    pub state: State,
    pub callbacks: HashMap<String, CallbackRecord>,
    pub(crate) pending_recommendation_delivery: Option<PendingRecommendationDelivery>,
    pub(crate) recommendation_formats: HashMap<PositionId, RecommendationFormat>,
    pub(crate) pending_feedback: Option<PendingFeedback>,
}

impl Default for Session {
    fn default() -> Self {
        Self::with_token_source(Arc::new(RandomTokenSource))
    }
}

impl Session {
    pub fn with_token_source(token_source: Arc<dyn TokenSource>) -> Self {
        Self {
            token_source,
            generation: 0,
            state: State::Idle,
            callbacks: HashMap::new(),
            pending_recommendation_delivery: None,
            recommendation_formats: HashMap::new(),
            pending_feedback: None,
        }
    }
    pub fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.state = State::Idle;
        self.callbacks.clear();
        self.recommendation_formats.clear();
    }
    pub fn begin(&mut self, state: State) {
        self.reset();
        self.state = state;
    }
    pub fn issue(&mut self, action: Action) -> Result<String, getrandom::Error> {
        let token = loop {
            let candidate = self.token_source.token()?;
            if !self.callbacks.contains_key(&candidate) {
                break candidate;
            }
        };
        self.callbacks.insert(
            token.clone(),
            CallbackRecord {
                action,
                generation: self.generation,
                message_id: None,
                action_key: format!("callback:{token}"),
                status: CallbackStatus::Pending,
            },
        );
        Ok(token)
    }
    pub fn activate(&mut self, tokens: &[String], message_id: i32) {
        if message_id <= 0 {
            return;
        }
        for token in tokens {
            if let Some(record) = self.callbacks.get_mut(token) {
                if record.status == CallbackStatus::Pending {
                    record.message_id = Some(message_id);
                    record.status = CallbackStatus::Active;
                }
            }
        }
    }
    pub fn discard(&mut self, tokens: &[String]) {
        for token in tokens {
            self.callbacks.remove(token);
        }
    }
    pub fn claim(&mut self, token: &str, message_id: i32) -> Option<(Action, String)> {
        if !callback::well_formed(token) {
            return None;
        }
        let record = self.callbacks.get_mut(token)?;
        if record.generation != self.generation
            || record.message_id != Some(message_id)
            || record.status != CallbackStatus::Active
            || !record.action.valid()
            || !action_matches_state(&record.action, &self.state, self.pending_feedback.as_ref())
        {
            return None;
        }
        record.status = CallbackStatus::Processing;
        Some((record.action.clone(), record.action_key.clone()))
    }
    pub fn finish(&mut self, token: &str, success: bool) {
        if let Some(record) = self.callbacks.get_mut(token) {
            if record.status == CallbackStatus::Processing {
                record.status = if matches!(record.action, Action::RecommendationDescription { .. })
                {
                    CallbackStatus::Active
                } else if success {
                    CallbackStatus::Consumed
                } else {
                    CallbackStatus::Active
                };
            }
        }
    }
}

fn action_matches_state(
    action: &Action,
    state: &State,
    pending_feedback: Option<&PendingFeedback>,
) -> bool {
    match action {
        Action::Recommend | Action::Rate | Action::Feedback | Action::Cancel => true,
        Action::RetryQuery { action_key } => matches!(state,
            State::PendingQuery { input, .. } if &input.action_key == action_key),
        Action::RetryFeedback { action_key } => {
            matches!(state, State::AwaitingFeedback)
                && pending_feedback.is_some_and(|pending| &pending.action_key == action_key)
        }
        Action::Select {
            intent,
            request_id,
            mal_id,
        } => matches!(state,
            State::ChoosingAnime { intent: current, query, candidates, selected_mal_id }
            if intent == current && *request_id == query.request_id && candidates.contains(mal_id)
                && selected_mal_id.is_none_or(|selected| selected == *mal_id)),
        Action::AnimeScore { request_id, .. } => matches!(state,
            State::Selected { intent: super::state::AnimeIntent::Rate, selection }
            if *request_id == selection.query.request_id),
        Action::RecommendationScore { .. } | Action::RecommendationDescription { .. } => matches!(
            state,
            State::Selected {
                intent: super::state::AnimeIntent::Recommend,
                ..
            }
        ),
    }
}

pub struct SessionStore {
    inner: Mutex<HashMap<Actor, Arc<AsyncMutex<Session>>>>,
    token_source: Arc<dyn TokenSource>,
}
impl Default for SessionStore {
    fn default() -> Self {
        Self::with_token_source(Arc::new(RandomTokenSource))
    }
}
impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_token_source(token_source: Arc<dyn TokenSource>) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            token_source,
        }
    }
    pub fn get(&self, actor: Actor) -> Arc<AsyncMutex<Session>> {
        let mut map = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        map.entry(actor)
            .or_insert_with(|| {
                Arc::new(AsyncMutex::new(Session::with_token_source(
                    self.token_source.clone(),
                )))
            })
            .clone()
    }
    pub fn existing(&self, actor: Actor) -> Option<Arc<AsyncMutex<Session>>> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&actor)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokens_are_opaque_and_bound_to_generation_message_state() {
        let mut session = Session::default();
        let token = session.issue(Action::Recommend).unwrap();
        assert_eq!(token.len(), 35);
        assert!(callback::well_formed(&token));
        assert!(session.claim(&token, 9).is_none());
        session.activate(std::slice::from_ref(&token), 9);
        assert!(session.claim(&token, 10).is_none());
        assert!(session.claim(&token, 9).is_some());
        assert!(session.claim(&token, 9).is_none());
        session.finish(&token, false);
        assert!(session.claim(&token, 9).is_some());
        session.reset();
        assert!(session.claim(&token, 9).is_none());
        assert!(Session::default().claim(&token, 9).is_none());
        for bad in [
            "",
            "a1:",
            "a1:é",
            "a1:FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
            "a1:0000000000000000000000000000000g",
        ] {
            assert!(!callback::well_formed(bad));
        }
    }
    #[test]
    fn scores_and_state_are_checked_before_claim() {
        let mut session = Session::default();
        for action in [
            Action::AnimeScore {
                request_id: 1,
                score: 0,
            },
            Action::AnimeScore {
                request_id: 1,
                score: 11,
            },
            Action::RecommendationScore {
                position_id: 1,
                score: -1,
            },
            Action::RecommendationScore {
                position_id: 1,
                score: 6,
            },
        ] {
            let token = session.issue(action).unwrap();
            session.activate(std::slice::from_ref(&token), 42);
            assert!(session.claim(&token, 42).is_none());
        }
        let token = session
            .issue(Action::AnimeScore {
                request_id: 1,
                score: 5,
            })
            .unwrap();
        session.activate(std::slice::from_ref(&token), 42);
        assert!(session.claim(&token, 42).is_none());
    }

    #[test]
    fn selection_must_belong_to_query_candidates() {
        let mut s = Session {
            state: State::ChoosingAnime {
                intent: super::super::state::AnimeIntent::Rate,
                query: super::super::state::QueryContext {
                    request_id: 7,
                    raw_query: "x".into(),
                    action_key: "msg:1:1".into(),
                },
                candidates: vec![3],
                selected_mal_id: None,
            },
            ..Session::default()
        };
        let good = s
            .issue(Action::Select {
                intent: super::super::state::AnimeIntent::Rate,
                request_id: 7,
                mal_id: 3,
            })
            .unwrap();
        let bad = s
            .issue(Action::Select {
                intent: super::super::state::AnimeIntent::Rate,
                request_id: 7,
                mal_id: 4,
            })
            .unwrap();
        s.activate(&[good.clone(), bad.clone()], 4);
        assert!(s.claim(&bad, 4).is_none());
        assert!(s.claim(&good, 4).is_some());
    }
}
