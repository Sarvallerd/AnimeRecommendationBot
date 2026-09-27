mod support;
use bot::{
    db::{
        AnimeRatingEventId, DbError, DeliveredPosition, DeliveryInput, PositionId, RequestId,
        UserProfile, WriteOutcome,
    },
    dialogue::{
        context::AppContext,
        repository::{DbFuture, Repository},
        state::{Actor, AnimeIntent, QueryContext, State},
        storage::SessionStore,
    },
    handlers,
};
use serde_json::json;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use support::{callback, dispatch, message, FakeTelegram, FixedTokens};
use teloxide::{
    prelude::*,
    types::{CallbackQuery, Update, UpdateId, UpdateKind},
};
use tokio::sync::Notify;

#[derive(Default)]
struct FakeRepo {
    upserts: AtomicUsize,
    resolves: AtomicUsize,
    resolve_ok: AtomicBool,
    fail_upsert: AtomicBool,
    listings: AtomicUsize,
    positions: Mutex<Vec<DeliveredPosition>>,
    recommendation_scores: Mutex<Vec<(i64, PositionId, i16)>>,
    score_calls: AtomicUsize,
    block_resolve: AtomicBool,
    resolve_entered: Notify,
    resolve_release: Notify,
}
impl Repository for FakeRepo {
    fn upsert_user<'a>(&'a self, _p: &'a UserProfile) -> DbFuture<'a, ()> {
        self.upserts.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if self.fail_upsert.load(Ordering::SeqCst) {
                Err(DbError::DatabaseFailure)
            } else {
                Ok(())
            }
        })
    }
    fn record_query<'a>(
        &'a self,
        _: i64,
        _: &'a str,
        _: &'a str,
    ) -> DbFuture<'a, WriteOutcome<RequestId>> {
        Box::pin(async { panic!("query feature is not wired") })
    }
    fn resolve_request<'a>(
        &'a self,
        _: i64,
        _: RequestId,
        _: i32,
        _: &'a str,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        self.resolves.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if self.block_resolve.load(Ordering::SeqCst) {
                self.resolve_entered.notify_one();
                self.resolve_release.notified().await;
            }
            if self.resolve_ok.load(Ordering::SeqCst) {
                Ok(WriteOutcome::Created(()))
            } else {
                Err(DbError::DatabaseFailure)
            }
        })
    }
    fn record_delivery<'a>(
        &'a self,
        _: i64,
        _: RequestId,
        _: &'a DeliveryInput,
    ) -> DbFuture<'a, WriteOutcome<PositionId>> {
        Box::pin(async { panic!("unexpected delivery") })
    }
    fn list_delivered_positions<'a>(
        &'a self,
        _: i64,
        _: RequestId,
    ) -> DbFuture<'a, Vec<DeliveredPosition>> {
        self.listings.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(self.positions.lock().unwrap().clone()) })
    }
    fn rate_anime<'a>(
        &'a self,
        _: i64,
        _: RequestId,
        _: i16,
    ) -> DbFuture<'a, WriteOutcome<AnimeRatingEventId>> {
        Box::pin(async { panic!("unexpected rating") })
    }
    fn rate_recommendation<'a>(
        &'a self,
        user_id: i64,
        position_id: PositionId,
        score: i16,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        self.score_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let mut rows = self.recommendation_scores.lock().unwrap();
            if let Some((user, _, existing)) = rows.iter().find(|(_, id, _)| *id == position_id) {
                if *user != user_id || *existing != score {
                    return Err(DbError::Conflict);
                }
                return Ok(WriteOutcome::AlreadyRecorded(()));
            }
            rows.push((user_id, position_id, score));
            Ok(WriteOutcome::Created(()))
        })
    }
    fn save_feedback<'a>(
        &'a self,
        _: i64,
        _: &'a str,
        _: &'a str,
    ) -> DbFuture<'a, WriteOutcome<i64>> {
        Box::pin(async { panic!("unexpected feedback") })
    }
}

fn fixture(repo: Arc<FakeRepo>) -> Arc<AppContext> {
    support::fixture(
        repo,
        Arc::new(SessionStore::with_token_source(Arc::new(FixedTokens(
            AtomicUsize::new(0),
        )))),
    )
}
#[tokio::test]
async fn public_router_commands_and_opaque_callbacks() {
    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let bot = api.bot();
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    dispatch(&bot, &ctx, message(1, 1, 1, "/start")).await;
    assert_eq!(repo.upserts.load(Ordering::SeqCst), 1);
    let initial = api.snapshot();
    assert_eq!(initial.len(), 1);
    assert!(initial[0].0.ends_with("/SendMessage"), "{:?}", initial);
    assert!(initial[0].1.contains("reply_markup"));
    let session = ctx.sessions.get(actor);
    let token = {
        let guard = session.lock().await;
        guard
            .callbacks
            .iter()
            .find(|(_, record)| matches!(record.action, bot::dialogue::callback::Action::Recommend))
            .unwrap()
            .0
            .clone()
    };
    dispatch(&bot, &ctx, callback(2, 1, 2, 101, Some(&token))).await;
    let after = api.snapshot();
    assert!(after[1].0.ends_with("/AnswerCallbackQuery"));
    assert!(after[2].1.contains("%D0%AD%D1%82%D0%B0") || after[2].1.contains("Эта кнопка"));
    assert_eq!(repo.upserts.load(Ordering::SeqCst), 1);
    dispatch(&bot, &ctx, callback(3, 1, 1, 999, Some(&token))).await;
    dispatch(&bot, &ctx, callback(4, 1, 1, 101, Some("a1:bad"))).await;
    api.fail_ack.store(true, Ordering::SeqCst);
    let before_ack_failure = api.snapshot().len();
    dispatch(&bot, &ctx, callback(5, 1, 1, 101, Some(&token))).await;
    assert_eq!(api.snapshot().len(), before_ack_failure + 1);
    assert_eq!(session.lock().await.state, State::Idle);
    dispatch(&bot, &ctx, callback(6, 1, 1, 101, Some(&token))).await;
    assert!(matches!(
        session.lock().await.state,
        State::AwaitingQuery {
            intent: AnimeIntent::Recommend
        }
    ));
    let before_duplicate = api.snapshot().len();
    dispatch(&bot, &ctx, callback(7, 1, 1, 101, Some(&token))).await;
    assert_eq!(api.snapshot().len(), before_duplicate + 2);
    let after = api.snapshot();
    assert!(after
        .iter()
        .any(|(path, _)| path.ends_with("/AnswerCallbackQuery")));
    // All six commands are recognized in every state; /help preserves state.
    let states = [
        State::Idle,
        State::AwaitingQuery {
            intent: AnimeIntent::Recommend,
        },
        State::AwaitingFeedback,
        State::ChoosingAnime {
            intent: AnimeIntent::Rate,
            query: QueryContext {
                request_id: 1,
                raw_query: "a".into(),
                action_key: "msg:1:1".into(),
            },
            candidates: vec![1],
            selected_mal_id: None,
        },
        State::Selected {
            intent: AnimeIntent::Rate,
            selection: bot::dialogue::state::ResolvedSelection {
                query: QueryContext {
                    request_id: 1,
                    raw_query: "a".into(),
                    action_key: "msg:1:1".into(),
                },
                seed_mal_id: 1,
                bundle_id: ctx.bundle.identity().into(),
            },
        },
    ];
    let mut id = 10;
    for state in states {
        for command in [
            "/help",
            "/recommend",
            "/rate",
            "/feedback",
            "/cancel",
            "/start",
        ] {
            {
                session.lock().await.state = state.clone();
            }
            id += 1;
            dispatch(&bot, &ctx, message(id, 1, 1, command)).await;
            if command == "/help" {
                assert_eq!(session.lock().await.state, state);
            }
        }
    }
    assert_eq!(repo.upserts.load(Ordering::SeqCst), 6);
    dispatch(&bot, &ctx, message(99, -1, 1, "/start")).await;
    assert_eq!(repo.upserts.load(Ordering::SeqCst), 6);
}

#[tokio::test]
async fn failed_menu_send_never_activates_button_and_profile_failure_is_honest() {
    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let bot = api.bot();
    repo.fail_upsert.store(true, Ordering::SeqCst);
    dispatch(&bot, &ctx, message(1, 1, 1, "/start")).await;
    let requests = api.snapshot();
    assert!(requests
        .iter()
        .any(|(_, body)| body.contains("Не удалось подтвердить сохранение профиля")));
    repo.fail_upsert.store(false, Ordering::SeqCst);
    api.fail_send.store(true, Ordering::SeqCst);
    let result = handlers::schema()
        .dispatch(dptree::deps![
            bot.clone(),
            message(2, 1, 1, "/start"),
            ctx.clone()
        ])
        .await;
    assert!(matches!(result, std::ops::ControlFlow::Break(Err(_))));
    assert!(ctx
        .sessions
        .get(Actor {
            chat_id: 1,
            user_id: 1
        })
        .lock()
        .await
        .callbacks
        .is_empty());
}

#[tokio::test]
async fn failed_callback_notice_releases_claim_for_same_key_retry() {
    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let bot = api.bot();
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    let session = ctx.sessions.get(actor);
    let token = {
        let mut guard = session.lock().await;
        guard.state = State::ChoosingAnime {
            intent: AnimeIntent::Recommend,
            query: QueryContext {
                request_id: 7,
                raw_query: "anime".into(),
                action_key: "msg:1:1".into(),
            },
            candidates: vec![3],
            selected_mal_id: None,
        };
        let token = guard
            .issue(bot::dialogue::callback::Action::Select {
                intent: AnimeIntent::Recommend,
                request_id: 7,
                mal_id: 3,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&token), 111);
        token
    };
    api.fail_send.store(true, Ordering::SeqCst);
    for id in [1, 2] {
        let result = handlers::schema()
            .dispatch(dptree::deps![
                bot.clone(),
                callback(id, 1, 1, 111, Some(&token)),
                ctx.clone()
            ])
            .await;
        assert!(matches!(result, std::ops::ControlFlow::Break(Err(_))));
        let guard = session.lock().await;
        assert!(matches!(
            guard.state,
            State::ChoosingAnime {
                selected_mal_id: Some(3),
                ..
            }
        ));
        let record = guard.callbacks.get(&token).unwrap();
        assert_eq!(
            record.status,
            bot::dialogue::storage::CallbackStatus::Active
        );
        assert_eq!(record.action_key, format!("callback:{token}"));
    }
    assert_eq!(repo.resolves.load(Ordering::SeqCst), 2);
    let requests = api.snapshot();
    assert_eq!(requests.len(), 4);
    assert!(requests[0].0.ends_with("/AnswerCallbackQuery"));
    assert!(requests[1].0.ends_with("/SendMessage"));
}

#[tokio::test]
async fn malformed_missing_and_inaccessible_updates_do_not_enter_business_flow() {
    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let bot = api.bot();
    let mut no_sender = message(1, 1, 1, "/start");
    if let UpdateKind::Message(ref mut msg) = no_sender.kind {
        msg.from = None;
    }
    dispatch(&bot, &ctx, no_sender).await;
    assert!(api.snapshot().is_empty());
    let malformed = "а".repeat(1000);
    dispatch(&bot, &ctx, callback(2, 1, 1, 101, Some(&malformed))).await;
    dispatch(&bot, &ctx, callback(3, 1, 1, 101, None)).await;
    let inaccessible:CallbackQuery=serde_json::from_value(json!({"id":"q4","from":{"id":1,"is_bot":false,"first_name":"Тест"},
        "chat_instance":"x","data":"a1:00000000000000000000000000000001",
        "message":{"message_id":101,"date":0,"chat":{"id":1,"type":"private","first_name":"Test"}}})).unwrap();
    dispatch(
        &bot,
        &ctx,
        Update {
            id: UpdateId(4),
            kind: UpdateKind::CallbackQuery(inaccessible),
        },
    )
    .await;
    let inline:CallbackQuery=serde_json::from_value(json!({"id":"q5","from":{"id":1,"is_bot":false,"first_name":"Тест"},
        "chat_instance":"x","data":"a1:00000000000000000000000000000001","inline_message_id":"inline"})).unwrap();
    dispatch(
        &bot,
        &ctx,
        Update {
            id: UpdateId(5),
            kind: UpdateKind::CallbackQuery(inline),
        },
    )
    .await;
    let group: CallbackQuery = serde_json::from_value(
        json!({"id":"q6","from":{"id":1,"is_bot":false,"first_name":"Тест"},
        "chat_instance":"x","data":"a1:00000000000000000000000000000001",
        "message":{"message_id":101,"date":1,"chat":{"id":-1,"type":"group","title":"Group"}}}),
    )
    .unwrap();
    dispatch(
        &bot,
        &ctx,
        Update {
            id: UpdateId(6),
            kind: UpdateKind::CallbackQuery(group),
        },
    )
    .await;
    let requests = api.snapshot();
    assert!(requests
        .last()
        .unwrap()
        .1
        .contains("Напишите мне в личном чате"));
    assert_eq!(
        requests
            .iter()
            .filter(|(path, _)| path.ends_with("/AnswerCallbackQuery"))
            .count(),
        5
    );
    assert_eq!(
        requests
            .iter()
            .filter(|(path, _)| path.ends_with("/SendMessage"))
            .count(),
        2
    );
    assert_eq!(repo.upserts.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn restart_disposes_issued_tokens() {
    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    let first = fixture(repo.clone());
    let bot = api.bot();
    dispatch(&bot, &first, message(1, 1, 1, "/start")).await;
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    let token = first
        .sessions
        .get(actor)
        .lock()
        .await
        .callbacks
        .keys()
        .next()
        .unwrap()
        .clone();
    let second = fixture(repo.clone());
    dispatch(&bot, &second, callback(2, 1, 1, 101, Some(&token))).await;
    assert!(second.sessions.existing(actor).is_none());
    assert_eq!(repo.upserts.load(Ordering::SeqCst), 1);
    let requests = api.snapshot();
    assert!(requests[1].0.ends_with("/AnswerCallbackQuery"));
    assert!(requests[2].1.contains("Эта кнопка устарела"));
}

#[tokio::test]
async fn selection_send_failure_keeps_owned_query_for_retry() {
    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    repo.resolve_ok.store(true, Ordering::SeqCst);
    let ctx = fixture(repo.clone());
    let bot = api.bot();
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    let session = ctx.sessions.get(actor);
    let token = {
        let mut guard = session.lock().await;
        guard.state = State::ChoosingAnime {
            intent: AnimeIntent::Rate,
            query: QueryContext {
                request_id: 7,
                raw_query: "anime".into(),
                action_key: "msg:1:1".into(),
            },
            candidates: vec![3],
            selected_mal_id: None,
        };
        let token = guard
            .issue(bot::dialogue::callback::Action::Select {
                intent: AnimeIntent::Rate,
                request_id: 7,
                mal_id: 3,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&token), 111);
        token
    };
    api.fail_send.store(true, Ordering::SeqCst);
    let first = handlers::schema()
        .dispatch(dptree::deps![
            bot.clone(),
            callback(1, 1, 1, 111, Some(&token)),
            ctx.clone()
        ])
        .await;
    assert!(matches!(first, std::ops::ControlFlow::Break(Err(_))));
    {
        let guard = session.lock().await;
        assert!(matches!(guard.state, State::ChoosingAnime { .. }));
        assert_eq!(
            guard.callbacks.get(&token).unwrap().status,
            bot::dialogue::storage::CallbackStatus::Active
        );
    }
    dispatch(&bot, &ctx, callback(2, 1, 1, 111, Some(&token))).await;
    assert!(matches!(
        session.lock().await.state,
        State::Selected {
            intent: AnimeIntent::Rate,
            ..
        }
    ));
    assert_eq!(repo.resolves.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn recommendation_score_reaches_owned_position_check_and_handler() {
    use bot::dialogue::{callback::Action, state::ResolvedSelection, storage::CallbackStatus};

    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let bot = api.bot();
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    let session = ctx.sessions.get(actor);
    let token = {
        let mut guard = session.lock().await;
        guard.state = State::Selected {
            intent: AnimeIntent::Recommend,
            selection: ResolvedSelection {
                query: QueryContext {
                    request_id: 7,
                    raw_query: "anime".into(),
                    action_key: "msg:1:1".into(),
                },
                seed_mal_id: 3,
                bundle_id: ctx.bundle.identity().into(),
            },
        };
        let token = guard
            .issue(Action::RecommendationScore {
                position_id: 9,
                score: 0,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&token), 111);
        token
    };
    repo.positions.lock().unwrap().push(DeliveredPosition {
        id: 9,
        request_id: 7,
        rank: 1,
        mal_id: 3,
        chat_id: 1,
        message_id: 111,
    });
    dispatch(&bot, &ctx, callback(1, 1, 1, 111, Some(&token))).await;
    assert_eq!(repo.listings.load(Ordering::SeqCst), 1);
    assert_eq!(*repo.recommendation_scores.lock().unwrap(), vec![(1, 9, 0)]);
    assert!(api
        .snapshot()
        .iter()
        .any(|(_, body)| body.contains("Оценка полезности рекомендации")));
    assert_eq!(
        session.lock().await.callbacks[&token].status,
        CallbackStatus::Consumed
    );

    // A different chat or request cannot turn a server-issued score into an owned position.
    let bad = {
        let mut guard = session.lock().await;
        let token = guard
            .issue(Action::RecommendationScore {
                position_id: 10,
                score: 5,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&token), 112);
        token
    };
    repo.positions.lock().unwrap().push(DeliveredPosition {
        id: 10,
        request_id: 7,
        rank: 2,
        mal_id: 3,
        chat_id: 2,
        message_id: 112,
    });
    dispatch(&bot, &ctx, callback(2, 1, 1, 112, Some(&bad))).await;
    assert_eq!(repo.listings.load(Ordering::SeqCst), 2);
    let wrong_request = {
        let mut guard = session.lock().await;
        let token = guard
            .issue(Action::RecommendationScore {
                position_id: 11,
                score: 5,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&token), 113);
        token
    };
    repo.positions.lock().unwrap().push(DeliveredPosition {
        id: 11,
        request_id: 8,
        rank: 3,
        mal_id: 3,
        chat_id: 1,
        message_id: 113,
    });
    dispatch(&bot, &ctx, callback(3, 1, 1, 113, Some(&wrong_request))).await;
    assert_eq!(repo.listings.load(Ordering::SeqCst), 3);
    let wrong_message = {
        let mut guard = session.lock().await;
        let token = guard
            .issue(Action::RecommendationScore {
                position_id: 12,
                score: 5,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&token), 114);
        token
    };
    repo.positions.lock().unwrap().push(DeliveredPosition {
        id: 12,
        request_id: 7,
        rank: 4,
        mal_id: 3,
        chat_id: 1,
        message_id: 115,
    });
    dispatch(&bot, &ctx, callback(4, 1, 1, 114, Some(&wrong_message))).await;
    assert_eq!(repo.listings.load(Ordering::SeqCst), 4);
    assert_eq!(repo.score_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        api.snapshot()
            .iter()
            .filter(|(_, body)| body.contains("Оценка полезности рекомендации"))
            .count(),
        1
    );
}

#[tokio::test]
async fn failed_flow_callback_send_can_retry_the_same_token_and_key() {
    use bot::dialogue::{callback::Action, storage::CallbackStatus};

    for action in [
        Action::Recommend,
        Action::Rate,
        Action::Feedback,
        Action::Cancel,
    ] {
        let api = FakeTelegram::new();
        let repo = Arc::new(FakeRepo::default());
        let ctx = fixture(repo);
        let bot = api.bot();
        let actor = Actor {
            chat_id: 1,
            user_id: 1,
        };
        let session = ctx.sessions.get(actor);
        let previous = State::AwaitingQuery {
            intent: AnimeIntent::Rate,
        };
        let (token, generation) = {
            let mut guard = session.lock().await;
            guard.state = previous.clone();
            let token = guard.issue(action.clone()).unwrap();
            guard.activate(std::slice::from_ref(&token), 111);
            (token, guard.generation)
        };
        api.fail_send.store(true, Ordering::SeqCst);
        let first = handlers::schema()
            .dispatch(dptree::deps![
                bot.clone(),
                callback(1, 1, 1, 111, Some(&token)),
                ctx.clone()
            ])
            .await;
        assert!(
            matches!(first, std::ops::ControlFlow::Break(Err(_))),
            "{action:?}: {first:?}"
        );
        {
            let guard = session.lock().await;
            assert_eq!(guard.state, previous, "{action:?}");
            assert_eq!(guard.generation, generation, "{action:?}");
            assert_eq!(
                guard.callbacks[&token].status,
                CallbackStatus::Active,
                "{action:?}"
            );
            assert_eq!(
                guard.callbacks[&token].action_key,
                format!("callback:{token}")
            );
        }
        dispatch(&bot, &ctx, callback(2, 1, 1, 111, Some(&token))).await;
        {
            let guard = session.lock().await;
            assert_eq!(guard.generation, generation + 1, "{action:?}");
            assert!(!guard.callbacks.contains_key(&token), "{action:?}");
            let expected = match action {
                Action::Recommend => State::AwaitingQuery {
                    intent: AnimeIntent::Recommend,
                },
                Action::Rate => State::AwaitingQuery {
                    intent: AnimeIntent::Rate,
                },
                Action::Feedback => State::AwaitingFeedback,
                _ => State::Idle,
            };
            assert_eq!(guard.state, expected, "{action:?}");
        }
        let before = api.snapshot().len();
        dispatch(&bot, &ctx, callback(3, 1, 1, 111, Some(&token))).await;
        assert_eq!(api.snapshot().len(), before + 2, "{action:?}");
    }
}

#[tokio::test]
async fn valid_token_used_in_another_private_chat_has_no_effect() {
    use bot::dialogue::callback::Action;

    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let bot = api.bot();
    let owner = Actor {
        chat_id: 1,
        user_id: 1,
    };
    let other = Actor {
        chat_id: 2,
        user_id: 1,
    };
    let token = {
        let session = ctx.sessions.get(owner);
        let mut guard = session.lock().await;
        let token = guard.issue(Action::Recommend).unwrap();
        guard.activate(std::slice::from_ref(&token), 111);
        token
    };
    ctx.sessions.get(other).lock().await.state = State::Idle;
    dispatch(&bot, &ctx, callback(1, 2, 1, 111, Some(&token))).await;
    assert_eq!(ctx.sessions.get(owner).lock().await.state, State::Idle);
    assert_eq!(ctx.sessions.get(other).lock().await.state, State::Idle);
    assert_eq!(repo.upserts.load(Ordering::SeqCst), 0);
    assert_eq!(repo.resolves.load(Ordering::SeqCst), 0);
    assert!(api
        .snapshot()
        .iter()
        .any(|(_, body)| body.contains("Эта кнопка устарела")));
}

#[tokio::test]
async fn simultaneous_duplicate_callbacks_have_one_repository_effect() {
    use bot::dialogue::callback::Action;
    use tokio::time::timeout;

    let api = FakeTelegram::new();
    let repo = Arc::new(FakeRepo::default());
    repo.resolve_ok.store(true, Ordering::SeqCst);
    repo.block_resolve.store(true, Ordering::SeqCst);
    let ctx = fixture(repo.clone());
    let bot = api.bot();
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    let session = ctx.sessions.get(actor);
    let token = {
        let mut guard = session.lock().await;
        guard.state = State::ChoosingAnime {
            intent: AnimeIntent::Rate,
            query: QueryContext {
                request_id: 7,
                raw_query: "anime".into(),
                action_key: "msg:1:1".into(),
            },
            candidates: vec![3],
            selected_mal_id: None,
        };
        let token = guard
            .issue(Action::Select {
                intent: AnimeIntent::Rate,
                request_id: 7,
                mal_id: 3,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&token), 111);
        token
    };
    let entered = repo.resolve_entered.notified();
    let first = tokio::spawn({
        let bot = bot.clone();
        let ctx = ctx.clone();
        let token = token.clone();
        async move { dispatch(&bot, &ctx, callback(1, 1, 1, 111, Some(&token))).await }
    });
    timeout(Duration::from_secs(3), entered).await.unwrap();
    let second = tokio::spawn({
        let bot = bot.clone();
        let ctx = ctx.clone();
        let token = token.clone();
        async move { dispatch(&bot, &ctx, callback(2, 1, 1, 111, Some(&token))).await }
    });
    timeout(Duration::from_secs(3), async {
        while api
            .snapshot()
            .iter()
            .filter(|(path, _)| path.ends_with("/AnswerCallbackQuery"))
            .count()
            < 2
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(repo.resolves.load(Ordering::SeqCst), 1);
    repo.resolve_release.notify_one();
    timeout(Duration::from_secs(3), first)
        .await
        .unwrap()
        .unwrap();
    timeout(Duration::from_secs(3), second)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(repo.resolves.load(Ordering::SeqCst), 1);
    assert!(matches!(
        session.lock().await.state,
        State::Selected {
            intent: AnimeIntent::Rate,
            ..
        }
    ));
    assert_eq!(
        api.snapshot()
            .iter()
            .filter(|(_, body)| body.contains("Оцените «Небесный поезд» · MAL ID 3 от 1 до 10."))
            .count(),
        1
    );
}
