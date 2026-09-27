use bot::{
    catalog::Bundle,
    db::{
        AnimeRatingEventId, DbError, DeliveredPosition, DeliveryInput, PositionId, RequestId,
        UserProfile, WriteOutcome,
    },
    dialogue::{
        callback::TokenSource,
        context::AppContext,
        repository::{DbFuture, Repository},
        state::{Actor, AnimeIntent, QueryContext, State},
        storage::SessionStore,
    },
    handlers,
};
use serde_json::json;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};
use teloxide::{
    prelude::*,
    types::{CallbackQuery, Message, Update, UpdateId, UpdateKind},
};

#[derive(Default)]
struct FakeRepo {
    upserts: AtomicUsize,
    resolves: AtomicUsize,
    resolve_ok: AtomicBool,
    fail_upsert: AtomicBool,
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
        Box::pin(async { panic!("unexpected listing") })
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
        _: i64,
        _: PositionId,
        _: i16,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async { panic!("unexpected rating") })
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

struct FakeTelegram {
    url: String,
    requests: Arc<Mutex<Vec<(String, String)>>>,
    fail_send: Arc<AtomicBool>,
    fail_ack: Arc<AtomicBool>,
}
impl FakeTelegram {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let fail_send = Arc::new(AtomicBool::new(false));
        let fail_ack = Arc::new(AtomicBool::new(false));
        let seen = requests.clone();
        let failure = fail_send.clone();
        let ack_failure = fail_ack.clone();
        thread::spawn(move || {
            let mut next_id = 100;
            for incoming in listener.incoming() {
                let Ok(mut stream) = incoming else {
                    break;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0u8; 4096];
                while let Ok(n) = stream.read(&mut chunk) {
                    if n == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&chunk[..n]);
                    if let Some(start) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..start]);
                        let len = head
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|value| value.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= start + 4 + len {
                            break;
                        }
                    }
                }
                let request = String::from_utf8_lossy(&bytes);
                let path = request.split_whitespace().nth(1).unwrap_or("").to_owned();
                let body = request
                    .split_once("\r\n\r\n")
                    .map(|(_, body)| body.to_owned())
                    .unwrap_or_default();
                seen.lock().unwrap().push((path.clone(), body));
                let response = if path.ends_with("/AnswerCallbackQuery") {
                    if ack_failure.swap(false, Ordering::SeqCst) {
                        json!({"ok":false,"error_code":500,"description":"failed"})
                    } else {
                        json!({"ok":true,"result":true})
                    }
                } else if failure.swap(false, Ordering::SeqCst) {
                    json!({"ok":false,"error_code":500,"description":"failed"})
                } else {
                    next_id += 1;
                    json!({"ok":true,"result":{"message_id":next_id,"date":1,"chat":{"id":1,"type":"private","first_name":"Test"},"text":"ok"}})
                };
                let body = response.to_string();
                let wire=format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body);
                stream.write_all(wire.as_bytes()).unwrap();
            }
        });
        Self {
            url,
            requests,
            fail_send,
            fail_ack,
        }
    }
    fn bot(&self) -> Bot {
        Bot::new("123:test").set_api_url(self.url.parse().unwrap())
    }
    fn snapshot(&self) -> Vec<(String, String)> {
        self.requests.lock().unwrap().clone()
    }
}
struct FixedTokens(AtomicUsize);
impl TokenSource for FixedTokens {
    fn token(&self) -> Result<String, getrandom::Error> {
        Ok(format!(
            "a1:{:032x}",
            self.0.fetch_add(1, Ordering::SeqCst) + 1
        ))
    }
}

fn fixture(repo: Arc<FakeRepo>) -> Arc<AppContext> {
    let bundle = Bundle::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/bundle"
    ))
    .unwrap();
    Arc::new(AppContext {
        bundle: Arc::new(bundle),
        repository: repo,
        sessions: Arc::new(SessionStore::with_token_source(Arc::new(FixedTokens(
            AtomicUsize::new(0),
        )))),
    })
}
fn message(id: u32, chat: i64, user: u64, text: &str) -> Update {
    let ty = if chat > 0 { "private" } else { "group" };
    let value = json!({"update_id":id,"message":{"message_id":id,"date":1,
        "chat":{"id":chat,"type":ty,"first_name":"Test","title":"Group"},
        "from":{"id":user,"is_bot":false,"first_name":"Тест"},"text":text}});
    Update {
        id: UpdateId(id),
        kind: UpdateKind::Message(
            serde_json::from_value::<Message>(value["message"].clone()).unwrap(),
        ),
    }
}
fn callback(id: u32, chat: i64, user: u64, message_id: i32, data: Option<&str>) -> Update {
    let value = json!({"update_id":id,"callback_query":{"id":format!("q{id}"),
        "from":{"id":user,"is_bot":false,"first_name":"Тест"},"chat_instance":"x",
        "message":{"message_id":message_id,"date":1,"chat":{"id":chat,"type":"private","first_name":"Test"}},
        "data":data}});
    Update {
        id: UpdateId(id),
        kind: UpdateKind::CallbackQuery(
            serde_json::from_value::<CallbackQuery>(value["callback_query"].clone()).unwrap(),
        ),
    }
}
async fn dispatch(bot: &Bot, ctx: &Arc<AppContext>, update: Update) {
    let result = handlers::schema()
        .dispatch(dptree::deps![bot.clone(), update, ctx.clone()])
        .await;
    assert!(
        matches!(result, std::ops::ControlFlow::Break(Ok(()))),
        "{result:?}"
    );
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
            },
            candidates: vec![1],
        },
        State::Selected {
            intent: AnimeIntent::Rate,
            selection: bot::dialogue::state::ResolvedSelection {
                query: QueryContext {
                    request_id: 1,
                    raw_query: "a".into(),
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
            },
            candidates: vec![3],
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
            },
            candidates: vec![3],
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
