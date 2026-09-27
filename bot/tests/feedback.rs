#[allow(dead_code)]
mod support;

use bot::{
    db::{
        AnimeRatingEventId, Db, DbError, DeliveredPosition, DeliveryInput, PositionId, RequestId,
        UserProfile, WriteOutcome,
    },
    dialogue::{
        callback::Action,
        context::AppContext,
        repository::{DbFuture, Repository},
        state::{Actor, State},
        storage::{CallbackStatus, SessionStore},
    },
    handlers,
};
use serde_json::json;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use support::{callback, message, FakeTelegram};
use teloxide::{
    prelude::*,
    types::{Message, Update, UpdateId, UpdateKind},
};

const ACTOR: Actor = Actor {
    chat_id: 73,
    user_id: 42,
};

#[derive(Default)]
struct FakeRepo {
    profiles: Mutex<Vec<UserProfile>>,
    rows: Mutex<HashMap<(i64, String), (String, i64)>>,
    saves: Mutex<Vec<(i64, String, String)>>,
    failure: AtomicUsize, // 1 profile, 2 before save, 3 after commit, 4 conflict
}
impl Repository for FakeRepo {
    fn upsert_user<'a>(&'a self, profile: &'a UserProfile) -> DbFuture<'a, ()> {
        Box::pin(async move {
            if self
                .failure
                .compare_exchange(1, 0, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Err(DbError::DatabaseFailure);
            }
            self.profiles.lock().unwrap().push(profile.clone());
            Ok(())
        })
    }
    fn record_query<'a>(
        &'a self,
        _: i64,
        _: &'a str,
        _: &'a str,
    ) -> DbFuture<'a, WriteOutcome<RequestId>> {
        Box::pin(async { panic!("unexpected search") })
    }
    fn resolve_request<'a>(
        &'a self,
        _: i64,
        _: RequestId,
        _: i32,
        _: &'a str,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async { panic!("unexpected selection") })
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
        Box::pin(async { panic!("unexpected delivery") })
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
        user: i64,
        key: &'a str,
        body: &'a str,
    ) -> DbFuture<'a, WriteOutcome<i64>> {
        Box::pin(async move {
            self.saves
                .lock()
                .unwrap()
                .push((user, key.into(), body.into()));
            let failure = self.failure.swap(0, Ordering::SeqCst);
            if failure == 2 {
                return Err(DbError::DatabaseFailure);
            }
            if failure == 4 {
                return Err(DbError::Conflict);
            }
            let mut rows = self.rows.lock().unwrap();
            let result = match rows.get(&(user, key.into())) {
                Some((old, id)) if old == body => WriteOutcome::AlreadyRecorded(*id),
                Some(_) => return Err(DbError::Conflict),
                None => {
                    let id = rows.len() as i64 + 1;
                    rows.insert((user, key.into()), (body.into(), id));
                    WriteOutcome::Created(id)
                }
            };
            if failure == 3 {
                Err(DbError::DatabaseFailure)
            } else {
                Ok(result)
            }
        })
    }
}
fn fixture(repo: Arc<dyn Repository>) -> Arc<AppContext> {
    support::fixture(repo, Arc::new(SessionStore::new()))
}
async fn run(bot: &Bot, ctx: &Arc<AppContext>, update: Update) -> bool {
    matches!(
        handlers::schema()
            .dispatch(dptree::deps![bot.clone(), update, ctx.clone()])
            .await,
        std::ops::ControlFlow::Break(Ok(()))
    )
}
fn text_updates(api: &FakeTelegram) -> Vec<String> {
    api.decoded()
        .into_iter()
        .filter(|(path, _)| path.ends_with("/SendMessage"))
        .map(|(_, fields)| fields.get("text").cloned().unwrap_or_default())
        .collect()
}
fn full_message(id: u32, body: Option<&str>) -> Update {
    let mut value = json!({"message_id":id,"date":1,"chat":{"id":73,"type":"private","first_name":"Test"},
        "from":{"id":42,"is_bot":false,"first_name":"Имя","last_name":"Фамилия","username":"handle","language_code":"ru"}});
    if let Some(body) = body {
        value["text"] = json!(body);
    } else {
        value["photo"] = json!([{"file_id":"a","file_unique_id":"b","width":1,"height":1}]);
    }
    Update {
        id: UpdateId(id),
        kind: UpdateKind::Message(serde_json::from_value::<Message>(value).unwrap()),
    }
}
async fn retry(ctx: &Arc<AppContext>) -> (String, i32) {
    let session = ctx.sessions.get(ACTOR);
    let guard = session.lock().await;
    guard
        .callbacks
        .iter()
        .find_map(|(token, record)| {
            matches!(record.action, Action::RetryFeedback { .. })
                .then(|| (token.clone(), record.message_id.unwrap()))
        })
        .unwrap()
}

#[tokio::test]
async fn direct_feedback_preserves_profile_and_exact_long_body() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    let body = format!(
        "  🌸\tстрока\n'é' ; DROP TABLE feedback;  {}  ",
        "длинный текст ".repeat(500)
    );
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    assert!(run(&bot, &ctx, full_message(2, Some(&body))).await);
    assert_eq!(
        repo.profiles.lock().unwrap().as_slice(),
        &[UserProfile {
            tg_id: 42,
            first_name: "Имя".into(),
            last_name: Some("Фамилия".into()),
            username: Some("handle".into()),
            language_code: Some("ru".into())
        }]
    );
    assert_eq!(
        repo.saves.lock().unwrap().as_slice(),
        &[(42, "msg:73:2".into(), body.clone())]
    );
    assert_eq!(
        repo.rows
            .lock()
            .unwrap()
            .get(&(42, "msg:73:2".into()))
            .unwrap()
            .0,
        body
    );
    assert_eq!(
        text_updates(&api).last().unwrap(),
        "Спасибо! Ваш отзыв сохранён."
    );
}

#[tokio::test]
async fn invalid_inputs_never_write_or_claim_success() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    for update in [
        full_message(2, None),
        full_message(3, Some(" \t\n ")),
        full_message(4, Some("bad\0body")),
    ] {
        assert!(run(&bot, &ctx, update).await);
    }
    assert!(repo.saves.lock().unwrap().is_empty());
    assert!(!text_updates(&api)
        .iter()
        .any(|text| text.contains("сохранён")));
}

#[tokio::test]
async fn failures_before_and_after_commit_retry_one_original_row() {
    for mode in [1, 2, 3] {
        let repo = Arc::new(FakeRepo::default());
        let ctx = fixture(repo.clone());
        let api = FakeTelegram::new();
        let bot = api.bot();
        assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
        repo.failure.store(mode, Ordering::SeqCst);
        assert!(!run(&bot, &ctx, full_message(2, Some("original"))).await);
        assert!(!text_updates(&api)
            .iter()
            .any(|text| text == "Спасибо! Ваш отзыв сохранён."));
        let (token, msg_id) = retry(&ctx).await;
        assert!(run(&bot, &ctx, callback(3, 73, 42, msg_id, Some(&token))).await);
        assert_eq!(repo.rows.lock().unwrap().len(), 1);
        assert_eq!(
            repo.rows
                .lock()
                .unwrap()
                .get(&(42, "msg:73:2".into()))
                .unwrap()
                .0,
            "original"
        );
        assert!(run(&bot, &ctx, callback(4, 73, 42, msg_id, Some(&token))).await);
        assert_eq!(repo.rows.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn confirmed_write_survives_confirmation_and_retry_ui_failures() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    api.fail_send_count.store(2, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, full_message(2, Some("persisted"))).await);
    assert_eq!(repo.saves.lock().unwrap().len(), 1);
    assert!(!ctx
        .sessions
        .get(ACTOR)
        .lock()
        .await
        .callbacks
        .values()
        .any(|r| matches!(r.action, Action::RetryFeedback { .. })
            && r.status == CallbackStatus::Active));
    repo.failure.store(1, Ordering::SeqCst);
    assert!(run(&bot, &ctx, message(3, 73, 42, "/feedback")).await);
    let (token, msg_id) = retry(&ctx).await;
    assert!(run(&bot, &ctx, callback(4, 73, 42, msg_id, Some(&token))).await);
    assert_eq!(repo.saves.lock().unwrap().len(), 1);
    assert_eq!(repo.profiles.lock().unwrap().len(), 1);
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn navigation_replay_and_callback_ownership_preserve_original() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    repo.failure.store(2, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, full_message(2, Some("original"))).await);
    let (old, old_message) = retry(&ctx).await;
    assert!(run(&bot, &ctx, full_message(2, Some("changed"))).await);
    assert!(run(&bot, &ctx, full_message(3, Some("fresh"))).await);
    assert_eq!(repo.saves.lock().unwrap().len(), 1);
    assert!(run(&bot, &ctx, callback(4, 73, 99, old_message, Some(&old))).await);
    assert!(run(&bot, &ctx, callback(5, 73, 42, old_message + 1, Some(&old))).await);
    assert_eq!(repo.saves.lock().unwrap().len(), 1);
    assert!(run(&bot, &ctx, message(6, 73, 42, "/help")).await);
    assert!(run(&bot, &ctx, message(7, 73, 42, "/cancel")).await);
    assert!(run(&bot, &ctx, message(8, 73, 42, "/recommend")).await);
    assert!(run(&bot, &ctx, callback(8, 73, 42, old_message, Some(&old))).await);
    assert_eq!(repo.saves.lock().unwrap().len(), 1);
    assert!(run(&bot, &ctx, full_message(2, Some("original"))).await);
    assert!(run(&bot, &ctx, full_message(2, Some("changed"))).await);
    assert!(run(&bot, &ctx, message(9, 73, 42, "/feedback")).await);
    let (new, new_message) = retry(&ctx).await;
    assert_ne!(old, new);
    assert!(run(&bot, &ctx, callback(10, 73, 42, new_message, Some(&new))).await);
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn terminal_conflict_requires_new_message() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    repo.failure.store(4, Ordering::SeqCst);
    assert!(run(&bot, &ctx, full_message(2, Some("old"))).await);
    assert!(matches!(
        ctx.sessions.get(ACTOR).lock().await.state,
        State::AwaitingFeedback
    ));
    assert!(repo.rows.lock().unwrap().is_empty());
    assert!(run(&bot, &ctx, full_message(3, Some("new"))).await);
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
}

#[cfg(test)]
async fn postgres_fixture(
    application_name: &str,
) -> (
    tokio_postgres::Client,
    tokio_postgres::Client,
    Arc<Db>,
    String,
) {
    use std::str::FromStr;
    use tokio_postgres::{Config, NoTls};
    static NEXT_SCHEMA: AtomicUsize = AtomicUsize::new(0);
    let url = std::env::var("ARB_TEST_DATABASE_URL").expect("set ARB_TEST_DATABASE_URL explicitly");
    let mut config = Config::from_str(&url).expect("valid test database URL");
    let name = config.get_dbname().expect("test database name");
    assert!(
        name == "arb_017_test" || name == "arb_ci_test",
        "unsafe ARB-017 test database name"
    );
    let (admin, connection) = config.connect(NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    let schema = format!(
        "arb017_{}_{}",
        std::process::id(),
        NEXT_SCHEMA.fetch_add(1, Ordering::Relaxed)
    );
    admin
        .batch_execute(&format!("CREATE SCHEMA {schema}"))
        .await
        .unwrap();
    config.options(format!("-c search_path={schema}"));
    config.application_name(application_name);
    let db = Arc::new(Db::new(&config).await.unwrap());
    db.migrate().await.unwrap();
    config.application_name(format!("{application_name}_direct"));
    let (direct, connection) = config.connect(NoTls).await.unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    (admin, direct, db, schema)
}

#[tokio::test]
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn arb017_postgres_exact_row_and_confirmation_retry() {
    let app = format!("arb017_persist_{}", std::process::id());
    let (admin, direct, db, schema) = postgres_fixture(&app).await;
    let ctx = fixture(db.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    let body = format!(
        " 🌸\tстрока\n'quoted'; SELECT 1; {} ",
        "очень длинный текст ".repeat(200)
    );
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    api.fail_send.store(true, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, full_message(2, Some(&body))).await);
    let before = direct.query_one("SELECT f.id,f.tg_id,f.action_key,f.body,u.first_name,u.last_name,u.username,u.language_code FROM arb_feedback f JOIN arb_users u ON u.tg_id=f.tg_id", &[]).await.unwrap();
    let row_id: i64 = before.get(0);
    assert_eq!(before.get::<_, i64>(1), 42);
    assert_eq!(before.get::<_, String>(2), "msg:73:2");
    assert_eq!(before.get::<_, String>(3), body);
    assert_eq!(before.get::<_, String>(4), "Имя");
    assert_eq!(before.get::<_, Option<String>>(5), Some("Фамилия".into()));
    assert_eq!(before.get::<_, Option<String>>(6), Some("handle".into()));
    assert_eq!(before.get::<_, Option<String>>(7), Some("ru".into()));
    let (token, message_id) = retry(&ctx).await;
    assert!(run(&bot, &ctx, callback(3, 73, 42, message_id, Some(&token))).await);
    let after = direct
        .query_one(
            "SELECT id,body FROM arb_feedback WHERE tg_id=42 AND action_key='msg:73:2'",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(after.get::<_, i64>(0), row_id);
    assert_eq!(after.get::<_, String>(1), body);
    assert_eq!(
        direct
            .query_one("SELECT count(*) FROM arb_feedback", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        1
    );
    assert_eq!(
        db.save_feedback(42, "msg:73:2", "changed").await,
        Err(DbError::Conflict)
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
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn arb017_postgres_interrupted_backend_reconnects_on_explicit_retry() {
    use tokio::time::{sleep, timeout, Duration, Instant};
    let app = format!("arb017_interrupt_{}", std::process::id());
    let (admin, direct, db, schema) = postgres_fixture(&app).await;
    let ctx = fixture(db);
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    direct
        .batch_execute("BEGIN; LOCK TABLE arb_feedback IN ACCESS EXCLUSIVE MODE")
        .await
        .unwrap();
    let running = tokio::spawn({
        let bot = bot.clone();
        let ctx = ctx.clone();
        async move { run(&bot, &ctx, full_message(2, Some("original body"))).await }
    });
    let deadline = Instant::now() + Duration::from_secs(8);
    let backend = loop {
        let rows = admin.query("SELECT pid FROM pg_stat_activity WHERE datname=current_database() AND application_name=$1 AND wait_event_type='Lock' AND query LIKE '%arb_feedback%'", &[&app]).await.unwrap();
        if let Some(row) = rows.first() {
            break row.get::<_, i32>(0);
        }
        assert!(
            Instant::now() < deadline,
            "feedback insert never waited on the private table lock"
        );
        sleep(Duration::from_millis(20)).await;
    };
    assert!(admin
        .query_one("SELECT pg_terminate_backend($1)", &[&backend])
        .await
        .unwrap()
        .get::<_, bool>(0));
    assert!(!timeout(Duration::from_secs(8), running)
        .await
        .unwrap()
        .unwrap());
    assert!(!text_updates(&api)
        .iter()
        .any(|text| text == "Спасибо! Ваш отзыв сохранён."));
    direct.batch_execute("ROLLBACK").await.unwrap();
    let (token, message_id) = retry(&ctx).await;
    assert!(run(&bot, &ctx, callback(3, 73, 42, message_id, Some(&token))).await);
    let rows = direct
        .query("SELECT tg_id,action_key,body FROM arb_feedback", &[])
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, i64>(0), 42);
    assert_eq!(rows[0].get::<_, String>(1), "msg:73:2");
    assert_eq!(rows[0].get::<_, String>(2), "original body");
    drop(ctx);
    drop(direct);
    admin
        .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
        .await
        .unwrap();
}

#[tokio::test]
async fn failed_retries_keep_one_live_button_and_failed_ack_does_not_write() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback extra words")).await);
    assert!(repo.saves.lock().unwrap().is_empty());
    repo.failure.store(2, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, full_message(2, Some("one"))).await);
    let (token, msg_id) = retry(&ctx).await;
    api.fail_ack.store(true, Ordering::SeqCst);
    assert!(run(&bot, &ctx, callback(3, 73, 42, msg_id, Some(&token))).await);
    assert_eq!(repo.saves.lock().unwrap().len(), 1);
    repo.failure.store(2, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, callback(4, 73, 42, msg_id, Some(&token))).await);
    {
        let session = ctx.sessions.get(ACTOR);
        let guard = session.lock().await;
        assert_eq!(
            guard
                .callbacks
                .values()
                .filter(|r| matches!(r.action, Action::RetryFeedback { .. })
                    && r.status == CallbackStatus::Active)
                .count(),
            1
        );
    }
    assert!(run(&bot, &ctx, callback(5, 73, 42, msg_id, Some(&token))).await);
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
    assert_eq!(repo.saves.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn restart_discards_old_callback_and_exact_replay_is_idempotent() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    api.fail_send.store(true, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, full_message(2, Some("persisted"))).await);
    let (old, msg_id) = retry(&ctx).await;
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
    let restarted = fixture(repo.clone());
    let sends_before_stale = text_updates(&api).len();
    assert!(run(&bot, &restarted, callback(3, 73, 42, msg_id, Some(&old))).await);
    assert_eq!(repo.saves.lock().unwrap().len(), 1);
    assert_eq!(text_updates(&api).len(), sends_before_stale + 1);
    assert_ne!(
        text_updates(&api).last().unwrap(),
        "Спасибо! Ваш отзыв сохранён."
    );
    assert!(run(&bot, &restarted, message(4, 73, 42, "/feedback")).await);
    assert!(run(&bot, &restarted, full_message(2, Some("persisted"))).await);
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
    assert_eq!(repo.saves.lock().unwrap().len(), 2);
    assert_eq!(
        text_updates(&api).last().unwrap(),
        "Спасибо! Ваш отзыв сохранён."
    );
}

#[tokio::test]
async fn simultaneous_callback_claims_write_one_row() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    repo.failure.store(2, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, full_message(2, Some("one"))).await);
    let (token, msg_id) = retry(&ctx).await;
    let first = run(&bot, &ctx, callback(3, 73, 42, msg_id, Some(&token)));
    let second = run(&bot, &ctx, callback(4, 73, 42, msg_id, Some(&token)));
    let (a, b) = tokio::join!(first, second);
    assert!(a && b);
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
    assert_eq!(repo.saves.lock().unwrap().len(), 2);
    assert_eq!(
        text_updates(&api)
            .iter()
            .filter(|text| *text == "Спасибо! Ваш отзыв сохранён.")
            .count(),
        1
    );
}

#[tokio::test]
async fn zero_source_id_and_failed_terminal_notice_do_not_claim_success() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    assert!(run(&bot, &ctx, full_message(0, Some("bad id"))).await);
    assert!(repo.saves.lock().unwrap().is_empty());
    repo.failure.store(4, Ordering::SeqCst);
    api.fail_send.store(true, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, full_message(2, Some("retryable"))).await);
    assert!(matches!(
        ctx.sessions.get(ACTOR).lock().await.state,
        State::AwaitingFeedback
    ));
    assert!(run(&bot, &ctx, message(3, 73, 42, "/feedback")).await);
    let (token, msg_id) = retry(&ctx).await;
    assert!(run(&bot, &ctx, callback(4, 73, 42, msg_id, Some(&token))).await);
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn exact_source_message_replay_retries_original_operation() {
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(repo.clone());
    let api = FakeTelegram::new();
    let bot = api.bot();
    assert!(run(&bot, &ctx, message(1, 73, 42, "/feedback")).await);
    repo.failure.store(2, Ordering::SeqCst);
    assert!(!run(&bot, &ctx, full_message(2, Some("точный текст"))).await);
    let (token, message_id) = retry(&ctx).await;
    assert!(run(&bot, &ctx, full_message(2, Some("точный текст"))).await);
    assert_eq!(
        repo.saves.lock().unwrap().as_slice(),
        &[
            (42, "msg:73:2".into(), "точный текст".into()),
            (42, "msg:73:2".into(), "точный текст".into())
        ]
    );
    assert_eq!(repo.rows.lock().unwrap().len(), 1);
    assert!(run(&bot, &ctx, callback(3, 73, 42, message_id, Some(&token))).await);
    assert_eq!(repo.saves.lock().unwrap().len(), 2);
}
