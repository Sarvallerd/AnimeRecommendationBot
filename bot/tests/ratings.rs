mod support;

use bot::{
    catalog::Bundle,
    db::{
        AnimeRatingEventId, Db, DbError, DeliveredPosition, DeliveryInput, PositionId, RequestId,
        UserProfile, WriteOutcome,
    },
    dialogue::{
        callback::Action,
        context::AppContext,
        repository::{DbFuture, Repository},
        state::{Actor, AnimeIntent, State},
        storage::{CallbackStatus, Session, SessionStore},
    },
    handlers,
};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use support::{callback, dispatch, message, FakeTelegram, FixedTokens};
use teloxide::prelude::*;
use tokio::sync::Notify;

type RequestRows = HashMap<(i64, String), (String, RequestId, Option<i32>)>;

#[derive(Default)]
struct FakeRepo {
    next_request: AtomicUsize,
    requests: Mutex<RequestRows>,
    calls: Mutex<Vec<(i64, RequestId, i16)>>,
    events: Mutex<HashMap<RequestId, (i16, AnimeRatingEventId)>>,
    failure: AtomicUsize,
    block_rating: AtomicBool,
    rating_entered: Notify,
    rating_release: Notify,
}
impl FakeRepo {
    fn set_failure(&self, mode: usize) {
        self.failure.store(mode, Ordering::SeqCst);
    }
}
impl Repository for FakeRepo {
    fn upsert_user<'a>(&'a self, _: &'a UserProfile) -> DbFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
    fn record_query<'a>(
        &'a self,
        user: i64,
        key: &'a str,
        query: &'a str,
    ) -> DbFuture<'a, WriteOutcome<RequestId>> {
        Box::pin(async move {
            let mut rows = self.requests.lock().unwrap();
            let row_key = (user, key.to_owned());
            if let Some((old, id, _)) = rows.get(&row_key) {
                return if old == query {
                    Ok(WriteOutcome::AlreadyRecorded(*id))
                } else {
                    Err(DbError::Conflict)
                };
            }
            let id = self.next_request.fetch_add(1, Ordering::SeqCst) as i64 + 1;
            rows.insert(row_key, (query.to_owned(), id, None));
            Ok(WriteOutcome::Created(id))
        })
    }
    fn resolve_request<'a>(
        &'a self,
        user: i64,
        request: RequestId,
        mal: i32,
        _: &'a str,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async move {
            let mut rows = self.requests.lock().unwrap();
            let (_, _, selected) = rows
                .iter_mut()
                .find(|((owner, _), (_, id, _))| *owner == user && *id == request)
                .map(|(_, row)| row)
                .ok_or(DbError::NotFound)?;
            match selected {
                Some(id) if *id != mal => Err(DbError::Conflict),
                Some(_) => Ok(WriteOutcome::AlreadyRecorded(())),
                None => {
                    *selected = Some(mal);
                    Ok(WriteOutcome::Created(()))
                }
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
        Box::pin(async { panic!("unexpected delivery lookup") })
    }
    fn rate_anime<'a>(
        &'a self,
        user: i64,
        request: RequestId,
        score: i16,
    ) -> DbFuture<'a, WriteOutcome<AnimeRatingEventId>> {
        Box::pin(async move {
            self.calls.lock().unwrap().push((user, request, score));
            if self.block_rating.swap(false, Ordering::SeqCst) {
                self.rating_entered.notify_one();
                self.rating_release.notified().await;
            }
            match self.failure.swap(0, Ordering::SeqCst) {
                1 => Err(DbError::DatabaseFailure),
                3 => Err(DbError::Conflict),
                4 => Err(DbError::NotFound),
                5 => Err(DbError::InvalidInput("test")),
                6 => Err(DbError::DatabaseFailure),
                mode => {
                    let mut events = self.events.lock().unwrap();
                    let outcome = if let Some((old, id)) = events.get(&request) {
                        if *old != score {
                            return Err(DbError::Conflict);
                        }
                        WriteOutcome::AlreadyRecorded(*id)
                    } else {
                        let id = events.len() as i64 + 1;
                        events.insert(request, (score, id));
                        WriteOutcome::Created(id)
                    };
                    if mode == 2 {
                        Err(DbError::DatabaseFailure)
                    } else {
                        Ok(outcome)
                    }
                }
            }
        })
    }
    fn rate_recommendation<'a>(
        &'a self,
        _: i64,
        _: PositionId,
        _: i16,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async { panic!("unexpected recommendation rating") })
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
fn bundle() -> Arc<Bundle> {
    Arc::new(
        Bundle::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/bundle"
        ))
        .unwrap(),
    )
}
fn fixture(repo: Arc<dyn Repository>) -> Arc<AppContext> {
    Arc::new(AppContext::new(
        bundle(),
        repo,
        Arc::new(SessionStore::with_token_source(Arc::new(FixedTokens(
            AtomicUsize::new(0),
        )))),
    ))
}
fn actor() -> Actor {
    Actor {
        chat_id: 73,
        user_id: 42,
    }
}
fn token(session: &Session, action: &Action) -> (String, i32) {
    session
        .callbacks
        .iter()
        .find_map(|(token, record)| {
            (&record.action == action).then(|| (token.clone(), record.message_id.unwrap()))
        })
        .unwrap()
}
async fn choose(bot: &Bot, ctx: &Arc<AppContext>, id: u32, mal: i32) -> (String, i32, RequestId) {
    let session = ctx.sessions.get(actor());
    let (choice, message_id, request) = {
        let guard = session.lock().await;
        let request = match &guard.state {
            State::ChoosingAnime { query, .. } => query.request_id,
            state => panic!("expected candidates, got {state:?}"),
        };
        let (choice, message_id) = token(
            &guard,
            &Action::Select {
                intent: AnimeIntent::Rate,
                request_id: request,
                mal_id: mal,
            },
        );
        (choice, message_id, request)
    };
    dispatch(bot, ctx, callback(id, 73, 42, message_id, Some(&choice))).await;
    (choice, message_id, request)
}
async fn search_and_choose(bot: &Bot, ctx: &Arc<AppContext>, start: u32) -> RequestId {
    dispatch(bot, ctx, message(start, 73, 42, "/rate")).await;
    dispatch(bot, ctx, message(start + 1, 73, 42, "Общее название")).await;
    choose(bot, ctx, start + 2, 2).await.2
}
async fn score_token(ctx: &Arc<AppContext>, request: RequestId, score: i16) -> (String, i32) {
    let session = ctx.sessions.get(actor());
    let guard = session.lock().await;
    token(
        &guard,
        &Action::AnimeScore {
            request_id: request,
            score,
        },
    )
}
async fn result(bot: &Bot, ctx: &Arc<AppContext>, update: teloxide::types::Update) -> bool {
    matches!(
        handlers::schema()
            .dispatch(dptree::deps![bot.clone(), update, ctx.clone()])
            .await,
        std::ops::ControlFlow::Break(Ok(()))
    )
}

#[tokio::test]
async fn full_flow_without_start_uses_selected_id_and_ten_bound_scores() {
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let request = search_and_choose(&bot, &ctx, 1).await;
    let guard = ctx.sessions.get(actor());
    let guard = guard.lock().await;
    assert!(
        matches!(&guard.state, State::Selected { selection, .. } if selection.seed_mal_id == 2)
    );
    let scores: Vec<_> = guard
        .callbacks
        .iter()
        .filter_map(|(token, record)| match &record.action {
            Action::AnimeScore { request_id, score } => {
                assert_eq!(*request_id, request);
                assert_eq!(record.status, CallbackStatus::Active);
                assert_eq!(record.message_id, Some(103));
                assert!(token.starts_with("a1:"));
                Some(*score)
            }
            _ => None,
        })
        .collect();
    assert_eq!(scores.len(), 10);
    assert!((1..=10).all(|score| scores.contains(&score)));
    assert!(repo.calls.lock().unwrap().is_empty());
    drop(guard);
    let keyboard = api
        .decoded()
        .into_iter()
        .filter_map(|(_, fields)| fields.get("reply_markup").cloned())
        .next_back()
        .unwrap();
    let markup: serde_json::Value = serde_json::from_str(&keyboard).unwrap();
    let rows = markup["inline_keyboard"].as_array().unwrap();
    assert_eq!(
        rows.iter()
            .map(|row| row.as_array().unwrap().len())
            .collect::<Vec<_>>(),
        [5, 5]
    );
    for (index, button) in rows
        .iter()
        .flat_map(|row| row.as_array().unwrap())
        .enumerate()
    {
        assert_eq!(button["text"], (index + 1).to_string());
        assert!(button["callback_data"].as_str().unwrap().starts_with("a1:"));
    }
    let (score, message_id) = score_token(&ctx, request, 4).await;
    dispatch(&bot, &ctx, callback(4, 73, 42, message_id, Some(&score))).await;
    assert_eq!(*repo.calls.lock().unwrap(), vec![(42, request, 4)]);
    assert_eq!(ctx.sessions.get(actor()).lock().await.state, State::Idle);
    assert!(api.decoded().iter().any(|(_, fields)| fields
        .get("text")
        .is_some_and(|text| text.contains("Оценка 4/10") && text.contains("MAL ID 2"))));
}

#[tokio::test]
async fn matched_alias_survives_rate_prompt_and_failed_write_retry() {
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    dispatch(&bot, &ctx, message(1, 73, 42, "/rate")).await;
    dispatch(&bot, &ctx, message(2, 73, 42, "second edition")).await;
    let request = choose(&bot, &ctx, 3, 2).await.2;
    let prompt = api
        .decoded()
        .into_iter()
        .filter_map(|(_, fields)| fields.get("text").cloned())
        .next_back()
        .unwrap();
    assert!(
        prompt.contains("Оцените «Second Edition» · MAL ID 2"),
        "{prompt}"
    );
    let (score, message_id) = score_token(&ctx, request, 7).await;
    repo.set_failure(2);
    assert!(!result(&bot, &ctx, callback(4, 73, 42, message_id, Some(&score))).await);
    assert!(ctx.sessions.get(actor()).lock().await.state != State::Idle);
    dispatch(&bot, &ctx, callback(5, 73, 42, message_id, Some(&score))).await;
    let confirmation = api
        .decoded()
        .into_iter()
        .filter_map(|(_, fields)| fields.get("text").cloned())
        .next_back()
        .unwrap();
    assert!(
        confirmation.contains("Оценка 7/10 для «Second Edition» · MAL ID 2 сохранена"),
        "{confirmation}"
    );
    assert_eq!(repo.events.lock().unwrap().len(), 1);
    assert_eq!(ctx.sessions.get(actor()).lock().await.state, State::Idle);
}

#[tokio::test]
async fn first_score_pins_and_uncertain_writes_retry_only_that_button() {
    for mode in [1, 2] {
        let api = FakeTelegram::new();
        let bot = api.bot();
        let repo = Arc::new(FakeRepo::default());
        let ctx = fixture(repo.clone());
        let request = search_and_choose(&bot, &ctx, 1).await;
        let (first, message_id) = score_token(&ctx, request, 4).await;
        let (sibling, _) = score_token(&ctx, request, 5).await;
        let generation = ctx.sessions.get(actor()).lock().await.generation;
        repo.set_failure(mode);
        assert!(!result(&bot, &ctx, callback(4, 73, 42, message_id, Some(&first))).await);
        {
            let session = ctx.sessions.get(actor());
            let guard = session.lock().await;
            assert_eq!(guard.generation, generation);
            assert!(
                matches!(&guard.state, State::Selected { selection, .. } if selection.query.request_id == request)
            );
            assert_eq!(
                guard.callbacks.get(&first).unwrap().status,
                CallbackStatus::Active
            );
            assert!(!guard.callbacks.contains_key(&sibling));
        }
        dispatch(&bot, &ctx, callback(5, 73, 42, message_id, Some(&sibling))).await;
        assert_eq!(repo.calls.lock().unwrap().len(), 1);
        dispatch(&bot, &ctx, callback(6, 73, 42, message_id, Some(&first))).await;
        assert_eq!(
            *repo.calls.lock().unwrap(),
            vec![(42, request, 4), (42, request, 4)]
        );
        assert_eq!(repo.events.lock().unwrap().len(), 1);
        assert_eq!(ctx.sessions.get(actor()).lock().await.state, State::Idle);
    }
}

#[tokio::test]
async fn failed_confirmation_or_failure_notice_keeps_exact_retry() {
    for database_failure in [false, true] {
        let api = FakeTelegram::new();
        let bot = api.bot();
        let repo = Arc::new(FakeRepo::default());
        let ctx = fixture(repo.clone());
        let request = search_and_choose(&bot, &ctx, 1).await;
        let (first, message_id) = score_token(&ctx, request, 7).await;
        let (sibling, _) = score_token(&ctx, request, 8).await;
        if database_failure {
            repo.set_failure(1);
        }
        api.fail_send.store(true, Ordering::SeqCst);
        assert!(!result(&bot, &ctx, callback(4, 73, 42, message_id, Some(&first))).await);
        {
            let session = ctx.sessions.get(actor());
            let guard = session.lock().await;
            assert!(matches!(guard.state, State::Selected { .. }));
            assert_eq!(
                guard.callbacks.get(&first).unwrap().status,
                CallbackStatus::Active
            );
            assert!(!guard.callbacks.contains_key(&sibling));
        }
        dispatch(&bot, &ctx, callback(5, 73, 42, message_id, Some(&first))).await;
        assert_eq!(repo.events.lock().unwrap().len(), 1);
        assert_eq!(ctx.sessions.get(actor()).lock().await.state, State::Idle);
    }
}

#[tokio::test]
async fn terminal_errors_reset_only_after_confirmed_notice() {
    for mode in [3, 4, 5] {
        let api = FakeTelegram::new();
        let bot = api.bot();
        let repo = Arc::new(FakeRepo::default());
        let ctx = fixture(repo.clone());
        let request = search_and_choose(&bot, &ctx, 1).await;
        let (first, message_id) = score_token(&ctx, request, 6).await;
        repo.set_failure(mode);
        api.fail_send.store(true, Ordering::SeqCst);
        assert!(!result(&bot, &ctx, callback(4, 73, 42, message_id, Some(&first))).await);
        assert!(matches!(
            ctx.sessions.get(actor()).lock().await.state,
            State::Selected { .. }
        ));
        repo.set_failure(mode);
        dispatch(&bot, &ctx, callback(5, 73, 42, message_id, Some(&first))).await;
        assert_eq!(ctx.sessions.get(actor()).lock().await.state, State::Idle);
        assert!(api.decoded().iter().any(|(_, fields)| fields
            .get("text")
            .is_some_and(|text| text.contains("Начните новый поиск: /rate"))));
    }
}

#[tokio::test]
async fn invalid_owner_message_generation_range_and_ack_do_not_write() {
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let request = search_and_choose(&bot, &ctx, 1).await;
    let (score, message_id) = score_token(&ctx, request, 3).await;
    dispatch(&bot, &ctx, callback(4, 73, 43, message_id, Some(&score))).await;
    dispatch(
        &bot,
        &ctx,
        callback(5, 73, 42, message_id + 1, Some(&score)),
    )
    .await;
    api.fail_ack.store(true, Ordering::SeqCst);
    dispatch(&bot, &ctx, callback(6, 73, 42, message_id, Some(&score))).await;
    {
        let session = ctx.sessions.get(actor());
        let mut guard = session.lock().await;
        let bad = guard
            .issue(Action::AnimeScore {
                request_id: request,
                score: 11,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&bad), message_id);
        assert!(guard.claim(&bad, message_id).is_none());
        let bad_request = guard
            .issue(Action::AnimeScore {
                request_id: request + 1,
                score: 3,
            })
            .unwrap();
        guard.activate(std::slice::from_ref(&bad_request), message_id);
        assert!(guard.claim(&bad_request, message_id).is_none());
    }
    assert!(repo.calls.lock().unwrap().is_empty());
    dispatch(&bot, &ctx, message(7, 73, 42, "/rate")).await;
    dispatch(&bot, &ctx, callback(8, 73, 42, message_id, Some(&score))).await;
    assert!(repo.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn failed_score_keyboard_returns_to_pinned_selection() {
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    dispatch(&bot, &ctx, message(1, 73, 42, "/rate")).await;
    dispatch(&bot, &ctx, message(2, 73, 42, "Общее название")).await;
    let (choice, message_id, request) = {
        let session = ctx.sessions.get(actor());
        let guard = session.lock().await;
        let request = match &guard.state {
            State::ChoosingAnime { query, .. } => query.request_id,
            _ => unreachable!(),
        };
        let (choice, message_id) = token(
            &guard,
            &Action::Select {
                intent: AnimeIntent::Rate,
                request_id: request,
                mal_id: 2,
            },
        );
        (choice, message_id, request)
    };
    api.fail_send.store(true, Ordering::SeqCst);
    assert!(!result(&bot, &ctx, callback(3, 73, 42, message_id, Some(&choice))).await);
    {
        let session = ctx.sessions.get(actor());
        let guard = session.lock().await;
        assert!(matches!(
            &guard.state,
            State::ChoosingAnime {
                selected_mal_id: Some(2),
                ..
            }
        ));
        assert!(!guard
            .callbacks
            .values()
            .any(|record| matches!(record.action, Action::AnimeScore { .. })));
    }
    dispatch(&bot, &ctx, callback(4, 73, 42, message_id, Some(&choice))).await;
    assert_eq!(score_token(&ctx, request, 10).await.1, 103);
    assert!(repo.calls.lock().unwrap().is_empty());
}

#[tokio::test]
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn postgres_public_router_preserves_history_and_current_rating() {
    use std::str::FromStr;
    use tokio_postgres::{Config, NoTls};
    static NEXT_SCHEMA: AtomicUsize = AtomicUsize::new(0);
    let url = std::env::var("ARB_TEST_DATABASE_URL").expect("set ARB_TEST_DATABASE_URL explicitly");
    let mut config = Config::from_str(&url).expect("valid test database URL");
    let name = config.get_dbname().expect("test database name");
    assert!(
        name == "arb_015_test" || name == "arb_ci_test",
        "unsafe ARB-015 test database name"
    );
    let (admin, connection) = config.connect(NoTls).await.unwrap();
    tokio::spawn(async move {
        connection.await.unwrap();
    });
    let schema = format!(
        "arb015_{}_{}",
        std::process::id(),
        NEXT_SCHEMA.fetch_add(1, Ordering::Relaxed)
    );
    admin
        .batch_execute(&format!("CREATE SCHEMA {schema}"))
        .await
        .unwrap();
    config.options(format!("-c search_path={schema}"));
    let (direct, connection) = config.connect(NoTls).await.unwrap();
    tokio::spawn(async move {
        connection.await.unwrap();
    });
    let db = Arc::new(Db::new(&config).await.unwrap());
    db.migrate().await.unwrap();
    let api = FakeTelegram::new();
    let bot = api.bot();
    let ctx = fixture(db.clone());
    let expected_bundle = ctx.bundle.identity().to_owned();
    let first_request = search_and_choose(&bot, &ctx, 1).await;
    let (first, first_message) = score_token(&ctx, first_request, 4).await;
    dispatch(&bot, &ctx, callback(4, 73, 42, first_message, Some(&first))).await;
    let second_request = search_and_choose(&bot, &ctx, 5).await;
    let (second, second_message) = score_token(&ctx, second_request, 9).await;
    dispatch(
        &bot,
        &ctx,
        callback(8, 73, 42, second_message, Some(&second)),
    )
    .await;
    assert_ne!(first_request, second_request);
    let events = direct
        .query(
            "SELECT e.id,e.request_id,e.tg_id,e.mal_id,e.score,e.created_at::text,
                r.action_key,r.raw_query,r.bundle_id,r.resolved_at::text
         FROM arb_anime_rating_events e JOIN arb_requests r ON r.id=e.request_id ORDER BY e.id",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(events.len(), 2);
    for (index, row) in events.iter().enumerate() {
        assert_eq!(row.get::<_, i64>(1), [first_request, second_request][index]);
        assert_eq!(row.get::<_, i64>(2), 42);
        assert_eq!(row.get::<_, i32>(3), 2);
        assert_eq!(row.get::<_, i16>(4), [4, 9][index]);
        assert!(!row.get::<_, String>(5).is_empty());
        assert_eq!(row.get::<_, String>(6), format!("msg:73:{}", [2, 6][index]));
        assert_eq!(row.get::<_, String>(7), "Общее название");
        assert_eq!(
            row.get::<_, Option<String>>(8).as_deref(),
            Some(expected_bundle.as_str())
        );
        assert!(row.get::<_, Option<String>>(9).is_some());
    }
    let current = direct.query_one(
        "SELECT current_event_id,created_at::text,updated_at::text FROM arb_anime_ratings WHERE tg_id=42 AND mal_id=2",
        &[],
    ).await.unwrap();
    let current_event: i64 = current.get(0);
    let updated_at: String = current.get(2);
    assert_eq!(current_event, events[1].get::<_, i64>(0));
    assert!(!current.get::<_, String>(1).is_empty());
    dispatch(&bot, &ctx, callback(9, 73, 42, first_message, Some(&first))).await;
    assert_eq!(
        db.rate_anime(42, first_request, 4).await.unwrap(),
        WriteOutcome::AlreadyRecorded(events[0].get::<_, i64>(0))
    );
    assert_eq!(
        db.rate_anime(42, first_request, 5).await,
        Err(DbError::Conflict)
    );
    let after = direct.query_one(
        "SELECT current_event_id,updated_at::text FROM arb_anime_ratings WHERE tg_id=42 AND mal_id=2",
        &[],
    ).await.unwrap();
    assert_eq!(after.get::<_, i64>(0), current_event);
    assert_eq!(after.get::<_, String>(1), updated_at);
    assert_eq!(
        direct
            .query_one("SELECT count(*) FROM arb_anime_rating_events", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        2
    );
    drop(ctx);
    drop(db);
    drop(direct);
    admin
        .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
        .await
        .unwrap();
}

#[tokio::test]
async fn simultaneous_score_callbacks_record_one_event() {
    use std::time::Duration;
    use tokio::time::timeout;
    for competing in [false, true] {
        let api = FakeTelegram::new();
        let bot = api.bot();
        let repo = Arc::new(FakeRepo::default());
        let ctx = fixture(repo.clone());
        let request = search_and_choose(&bot, &ctx, 1).await;
        let (first, message_id) = score_token(&ctx, request, 4).await;
        let (other, _) = score_token(&ctx, request, 5).await;
        repo.block_rating.store(true, Ordering::SeqCst);
        let entered = repo.rating_entered.notified();
        let first_task = tokio::spawn({
            let bot = bot.clone();
            let ctx = ctx.clone();
            let first = first.clone();
            async move { dispatch(&bot, &ctx, callback(4, 73, 42, message_id, Some(&first))).await }
        });
        timeout(Duration::from_secs(3), entered).await.unwrap();
        let second_task = tokio::spawn({
            let bot = bot.clone();
            let ctx = ctx.clone();
            let second = if competing { other } else { first };
            async move { dispatch(&bot, &ctx, callback(5, 73, 42, message_id, Some(&second))).await }
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
        assert_eq!(repo.calls.lock().unwrap().len(), 1);
        repo.rating_release.notify_one();
        timeout(Duration::from_secs(3), first_task)
            .await
            .unwrap()
            .unwrap();
        timeout(Duration::from_secs(3), second_task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(repo.calls.lock().unwrap().len(), 1);
        assert_eq!(repo.events.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn long_unicode_title_is_bounded_and_controls_are_plain_text() {
    use bot::dialogue::state::{QueryContext, ResolvedSelection};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    let dir = tempfile::tempdir().unwrap();
    let base = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bundle"
    ));
    let mut catalog: Value =
        serde_json::from_slice(&std::fs::read(base.join("catalog.json")).unwrap()).unwrap();
    catalog["anime"]["2"]["title"] = json!(format!("Название\n{}", "🌸".repeat(200)));
    let catalog_bytes = format!("{catalog}\n").into_bytes();
    std::fs::write(dir.path().join("catalog.json"), &catalog_bytes).unwrap();
    let neighbors_bytes = std::fs::read(base.join("neighbors.json")).unwrap();
    std::fs::write(dir.path().join("neighbors.json"), &neighbors_bytes).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(base.join("manifest.json")).unwrap()).unwrap();
    manifest["files"]["catalog.json"]["sha256"] =
        json!(format!("{:x}", Sha256::digest(&catalog_bytes)));
    std::fs::write(dir.path().join("manifest.json"), format!("{manifest}\n")).unwrap();
    let bundle = Arc::new(Bundle::load(dir.path()).unwrap());
    let repo = Arc::new(FakeRepo::default());
    let ctx = Arc::new(AppContext::new(
        bundle.clone(),
        repo,
        Arc::new(SessionStore::new()),
    ));
    let api = FakeTelegram::new();
    let bot = api.bot();
    let selection = ResolvedSelection {
        query: QueryContext {
            request_id: 1,
            raw_query: "Общее название".into(),
            action_key: "msg:73:1".into(),
        },
        seed_mal_id: 2,
        bundle_id: bundle.identity().into(),
    };
    let session = ctx.sessions.get(actor());
    let mut guard = session.lock().await;
    guard.state = State::Selected {
        intent: AnimeIntent::Rate,
        selection: selection.clone(),
    };
    handlers::ratings::begin(&bot, &ctx, actor(), &mut guard, &selection)
        .await
        .unwrap();
    drop(guard);
    let text = api
        .decoded()
        .into_iter()
        .filter_map(|(_, fields)| fields.get("text").cloned())
        .next_back()
        .unwrap();
    assert!(text.contains("MAL ID 2"));
    assert!(text.contains("Название 🌸"));
    assert!(!text.contains('\n') || text.contains("10.\nЧтобы"));
    let displayed = text.split('«').nth(1).unwrap().split('»').next().unwrap();
    assert!(displayed.encode_utf16().count() <= 240);
    assert!(!displayed.chars().any(char::is_control));
}
