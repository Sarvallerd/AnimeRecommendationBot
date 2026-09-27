#[allow(dead_code)]
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
        state::{Actor, AnimeIntent, QueryContext, ResolvedSelection, State},
        storage::{Session, SessionStore},
    },
    handlers::recommendations,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        Arc, Mutex,
    },
};
use support::{callback, dispatch, message, FakeTelegram};

const ACTOR: Actor = Actor {
    chat_id: 73,
    user_id: 42,
};

fn bundle(count: usize, long: bool) -> (tempfile::TempDir, Arc<Bundle>) {
    let dir = tempfile::tempdir().unwrap();
    let base = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/bundle"
    ));
    let mut catalog: Value =
        serde_json::from_slice(&std::fs::read(base.join("catalog.json")).unwrap()).unwrap();
    let mut neighbors: Value =
        serde_json::from_slice(&std::fs::read(base.join("neighbors.json")).unwrap()).unwrap();
    neighbors["neighbors"]["1"] =
        json!(neighbors["neighbors"]["1"].as_array().unwrap()[..count].to_vec());
    if long {
        catalog["anime"]["1"]["title"] = json!("🌸".repeat(300));
        catalog["anime"]["10"]["title"] = json!("🌸".repeat(500));
        catalog["anime"]["10"]["type"] = json!("T".repeat(200));
        catalog["anime"]["10"]["genres"] = json!(["Ж".repeat(500)]);
        catalog["anime"]["10"]["synopsis"] = json!("🌸".repeat(6000));
        catalog["anime"]["10"]["episodes"] = json!("LONG_EPISODES_MARKER");
    }
    let catalog_bytes = format!("{catalog}\n")
        .replace("\"LONG_EPISODES_MARKER\"", &"9".repeat(300))
        .into_bytes();
    let neighbor_bytes = format!("{neighbors}\n").into_bytes();
    std::fs::write(dir.path().join("catalog.json"), &catalog_bytes).unwrap();
    std::fs::write(dir.path().join("neighbors.json"), &neighbor_bytes).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(base.join("manifest.json")).unwrap()).unwrap();
    manifest["files"]["catalog.json"]["sha256"] =
        json!(format!("{:x}", Sha256::digest(&catalog_bytes)));
    manifest["files"]["neighbors.json"]["sha256"] =
        json!(format!("{:x}", Sha256::digest(&neighbor_bytes)));
    std::fs::write(dir.path().join("manifest.json"), format!("{manifest}\n")).unwrap();
    let loaded = Arc::new(Bundle::load(dir.path()).unwrap());
    (dir, loaded)
}

fn selection(bundle: &Bundle, request: i64, seed: i32) -> ResolvedSelection {
    ResolvedSelection {
        query: QueryContext {
            request_id: request,
            raw_query: "Общее название".into(),
            action_key: format!("msg:73:{request}"),
        },
        seed_mal_id: seed,
        bundle_id: bundle.identity().to_owned(),
    }
}
fn session(sel: &ResolvedSelection) -> Session {
    let mut session = Session::default();
    session.state = State::Selected {
        intent: AnimeIntent::Recommend,
        selection: sel.clone(),
    };
    session
}
fn context(bundle: Arc<Bundle>, repo: Arc<dyn Repository>) -> AppContext {
    AppContext::new(bundle, repo, Arc::new(SessionStore::new()))
}
fn sent_texts(api: &FakeTelegram) -> Vec<String> {
    api.decoded()
        .into_iter()
        .filter(|(path, _)| path.ends_with("/SendMessage"))
        .map(|(_, fields)| fields.get("text").cloned().unwrap_or_default())
        .collect()
}

#[derive(Default)]
struct FakeRepo {
    rows: Mutex<Vec<DeliveredPosition>>,
    resolves: Mutex<Vec<(i64, i32, String)>>,
    next: Mutex<i64>,
    fail_list: AtomicBool,
    write_failure: AtomicU8, // 1 before commit, 2 after commit
    fail_send_after_record: Mutex<Option<Arc<AtomicBool>>>,
}
impl Repository for FakeRepo {
    fn upsert_user<'a>(&'a self, _: &'a UserProfile) -> DbFuture<'a, ()> {
        Box::pin(async { panic!("unexpected") })
    }
    fn record_query<'a>(
        &'a self,
        _: i64,
        _: &'a str,
        _: &'a str,
    ) -> DbFuture<'a, WriteOutcome<RequestId>> {
        Box::pin(async { panic!("unexpected") })
    }
    fn resolve_request<'a>(
        &'a self,
        _: i64,
        request: RequestId,
        seed: i32,
        bundle: &'a str,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async move {
            let mut rows = self.resolves.lock().unwrap();
            if let Some((_, old_seed, old_bundle)) = rows.iter().find(|(id, _, _)| *id == request) {
                return if *old_seed == seed && old_bundle == bundle {
                    Ok(WriteOutcome::AlreadyRecorded(()))
                } else {
                    Err(DbError::Conflict)
                };
            }
            rows.push((request, seed, bundle.to_owned()));
            Ok(WriteOutcome::Created(()))
        })
    }
    fn record_delivery<'a>(
        &'a self,
        _: i64,
        request: RequestId,
        input: &'a DeliveryInput,
    ) -> DbFuture<'a, WriteOutcome<PositionId>> {
        Box::pin(async move {
            let failure = self.write_failure.swap(0, Ordering::SeqCst);
            if failure == 1 {
                return Err(DbError::DatabaseFailure);
            }
            let mut rows = self.rows.lock().unwrap();
            let outcome = if let Some(row) = rows
                .iter()
                .find(|row| row.request_id == request && row.rank == input.rank)
            {
                if row.mal_id != input.mal_id
                    || row.chat_id != input.chat_id
                    || row.message_id != input.message_id
                {
                    return Err(DbError::Conflict);
                }
                WriteOutcome::AlreadyRecorded(row.id)
            } else {
                let mut next = self.next.lock().unwrap();
                *next += 1;
                let id = *next;
                rows.push(DeliveredPosition {
                    id,
                    request_id: request,
                    rank: input.rank,
                    mal_id: input.mal_id,
                    chat_id: input.chat_id,
                    message_id: input.message_id,
                });
                WriteOutcome::Created(id)
            };
            if let Some(flag) = self.fail_send_after_record.lock().unwrap().take() {
                flag.store(true, Ordering::SeqCst);
            }
            if failure == 2 {
                Err(DbError::DatabaseFailure)
            } else {
                Ok(outcome)
            }
        })
    }
    fn list_delivered_positions<'a>(
        &'a self,
        _: i64,
        request: RequestId,
    ) -> DbFuture<'a, Vec<DeliveredPosition>> {
        Box::pin(async move {
            if self.fail_list.swap(false, Ordering::SeqCst) {
                return Err(DbError::DatabaseFailure);
            }
            Ok(self
                .rows
                .lock()
                .unwrap()
                .iter()
                .filter(|row| row.request_id == request)
                .cloned()
                .collect())
        })
    }
    fn rate_anime<'a>(
        &'a self,
        _: i64,
        _: RequestId,
        _: i16,
    ) -> DbFuture<'a, WriteOutcome<AnimeRatingEventId>> {
        Box::pin(async { panic!("unexpected") })
    }
    fn rate_recommendation<'a>(
        &'a self,
        _: i64,
        _: PositionId,
        _: i16,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async { panic!("unexpected") })
    }
    fn save_feedback<'a>(
        &'a self,
        _: i64,
        _: &'a str,
        _: &'a str,
    ) -> DbFuture<'a, WriteOutcome<i64>> {
        Box::pin(async { panic!("unexpected") })
    }
}

#[tokio::test]
async fn canonical_five_two_and_empty_card_sets() {
    for count in [5, 2, 0] {
        let (_dir, bundle) = bundle(count, false);
        let repo = Arc::new(FakeRepo::default());
        let ctx = context(bundle.clone(), repo.clone());
        let api = FakeTelegram::new();
        let sel = selection(&bundle, 1, 1);
        let mut session = session(&sel);
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .unwrap();
        let texts = sent_texts(&api);
        let rows = repo.rows.lock().unwrap().clone();
        assert_eq!(rows.len(), count);
        assert_eq!(texts.len(), count.max(1));
        if count == 0 {
            assert_eq!(
                texts,
                vec!["Для этого аниме пока нет рекомендаций. Попробуйте другое: /recommend."]
            );
        } else {
            assert_eq!(
                rows.iter().map(|row| row.mal_id).collect::<Vec<_>>(),
                [10, 11, 12, 2, 3][..count]
            );
            for (index, (row, text)) in rows.iter().zip(&texts).enumerate() {
                assert_eq!(row.rank, (index + 1) as i16);
                assert_eq!(row.chat_id, ACTOR.chat_id);
                assert_eq!(row.message_id, 101 + index as i32);
                assert!(text.contains(&format!("Рекомендация {}/{}", index + 1, count)));
                assert!(text.contains(&format!("MAL ID: {}", row.mal_id)));
            }
            recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
                .await
                .unwrap();
            assert_eq!(sent_texts(&api).len(), count);
        }
    }
}

#[tokio::test]
async fn nulls_long_unicode_and_plain_text_stay_bounded() {
    let (_dir, bundle) = bundle(2, true);
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo);
    let api = FakeTelegram::new();
    let sel = selection(&bundle, 1, 1);
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session(&sel), &sel)
        .await
        .unwrap();
    let texts = sent_texts(&api);
    assert_eq!(texts.len(), 2);
    assert!(texts[0].encode_utf16().count() <= 4000);
    assert!(texts[0].contains("Название: "));
    assert!(texts[0].contains("Описание: "));
    assert!(texts[0].ends_with('…'));
    assert!(texts[1].contains("Оценка MAL: нет данных"));
    assert!(texts[1].contains("Описание: Описание отсутствует."));
    assert!(api
        .decoded()
        .iter()
        .all(|(_, fields)| !fields.contains_key("parse_mode")
            && !fields.contains_key("reply_markup")));
}

#[tokio::test]
async fn send_failures_stop_and_retry_skips_recorded_cards() {
    for fail_first in [true, false] {
        let (_dir, bundle) = bundle(2, false);
        let repo = Arc::new(FakeRepo::default());
        let ctx = context(bundle.clone(), repo.clone());
        let api = FakeTelegram::new();
        let sel = selection(&bundle, 1, 1);
        let mut session = session(&sel);
        if fail_first {
            api.fail_send.store(true, Ordering::SeqCst);
        } else {
            *repo.fail_send_after_record.lock().unwrap() = Some(api.fail_send.clone());
        }
        assert!(
            recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
                .await
                .is_err()
        );
        assert_eq!(
            repo.rows.lock().unwrap().len(),
            if fail_first { 0 } else { 1 }
        );
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .unwrap();
        let rows = repo.rows.lock().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rank, 1);
        assert_eq!(rows[1].rank, 2);
    }
}

#[tokio::test]
async fn uncertain_writes_reuse_original_coordinates_after_reset() {
    for failure in [1, 2] {
        let (_dir, bundle) = bundle(2, false);
        let repo = Arc::new(FakeRepo::default());
        let ctx = context(bundle.clone(), repo.clone());
        let api = FakeTelegram::new();
        let sel = selection(&bundle, 1, 1);
        let mut session = session(&sel);
        repo.write_failure.store(failure, Ordering::SeqCst);
        assert!(
            recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
                .await
                .is_err()
        );
        session.reset();
        session.state = State::Selected {
            intent: AnimeIntent::Recommend,
            selection: sel.clone(),
        };
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .unwrap();
        let rows = repo.rows.lock().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].message_id, 101);
        assert_eq!(rows[1].message_id, 103); // Retry notice is message 102.
        assert_eq!(
            sent_texts(&api)
                .iter()
                .filter(|text| text.starts_with("Рекомендация 1/"))
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn old_pending_keeps_its_request_when_navigation_starts_new_search() {
    let (_dir, bundle) = bundle(2, false);
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone());
    let api = FakeTelegram::new();
    let first = selection(&bundle, 1, 1);
    let second = selection(&bundle, 2, 2);
    let mut session = session(&first);
    repo.write_failure.store(1, Ordering::SeqCst);
    assert!(
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &first)
            .await
            .is_err()
    );
    session.reset();
    session.state = State::Selected {
        intent: AnimeIntent::Recommend,
        selection: second.clone(),
    };
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &second)
        .await
        .unwrap();
    let rows = repo.rows.lock().unwrap();
    assert_eq!(rows.iter().filter(|row| row.request_id == 1).count(), 1);
    assert_eq!(
        rows.iter()
            .find(|row| row.request_id == 1)
            .unwrap()
            .message_id,
        101
    );
    assert_eq!(rows.iter().filter(|row| row.request_id == 2).count(), 5);
    assert!(repo
        .resolves
        .lock()
        .unwrap()
        .contains(&(1, 1, first.bundle_id)));
}

#[tokio::test]
async fn listing_and_contradictory_history_fail_before_sends() {
    let (_dir, bundle) = bundle(2, false);
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone());
    let api = FakeTelegram::new();
    let sel = selection(&bundle, 1, 1);
    let mut session = session(&sel);
    repo.fail_list.store(true, Ordering::SeqCst);
    assert!(
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .is_err()
    );
    assert!(sent_texts(&api)
        .iter()
        .all(|text| !text.starts_with("Рекомендация")));
    repo.rows.lock().unwrap().push(DeliveredPosition {
        id: 1,
        request_id: 1,
        rank: 1,
        mal_id: 12,
        chat_id: 73,
        message_id: 101,
    });
    assert!(
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .is_err()
    );
    assert!(sent_texts(&api)
        .iter()
        .all(|text| !text.starts_with("Рекомендация")));
}

#[tokio::test]
async fn wrong_selection_and_actor_cannot_reassign_pending_delivery() {
    let (_dir, bundle) = bundle(2, false);
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone());
    let api = FakeTelegram::new();
    let sel = selection(&bundle, 1, 1);
    let mut session = session(&sel);
    let wrong = selection(&bundle, 1, 2);
    assert!(
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &wrong)
            .await
            .is_err()
    );
    repo.write_failure.store(1, Ordering::SeqCst);
    assert!(
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .is_err()
    );
    let foreign = Actor {
        chat_id: 74,
        user_id: 43,
    };
    assert!(
        recommendations::begin(&api.bot(), &ctx, foreign, &mut session, &sel)
            .await
            .is_err()
    );
    assert!(repo.rows.lock().unwrap().is_empty());
}

#[tokio::test]
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn postgres_records_real_coordinates_and_provenance() {
    use std::str::FromStr;
    use tokio_postgres::{Config, NoTls};
    let url = std::env::var("ARB_TEST_DATABASE_URL").expect("set ARB_TEST_DATABASE_URL");
    let mut config = Config::from_str(&url).unwrap();
    let name = config.get_dbname().unwrap();
    assert!(
        name == "arb_014_test" || name == "arb_ci_test",
        "unsafe ARB014 test database name"
    );
    let (admin, connection) = config.connect(NoTls).await.unwrap();
    tokio::spawn(async move {
        connection.await.unwrap();
    });
    let schema = format!("arb014_{}", std::process::id());
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
    let (_dir, bundle) = bundle(2, false);
    let ctx = Arc::new(context(bundle.clone(), db.clone()));
    let api = FakeTelegram::new();
    let bot = api.bot();
    dispatch(&bot, &ctx, message(1, 73, 42, "/recommend")).await;
    dispatch(&bot, &ctx, message(2, 73, 42, "Общее название")).await;
    let actor = ACTOR;
    let (token, candidate_message) = {
        let session = ctx.sessions.get(actor);
        let guard = session.lock().await;
        guard
            .callbacks
            .iter()
            .find_map(|(token, record)| {
                matches!(record.action, Action::Select { mal_id: 1, .. })
                    .then_some((token.clone(), record.message_id.unwrap()))
            })
            .unwrap()
    };
    dispatch(
        &bot,
        &ctx,
        callback(3, 73, 42, candidate_message, Some(&token)),
    )
    .await;
    dispatch(
        &bot,
        &ctx,
        callback(4, 73, 42, candidate_message, Some(&token)),
    )
    .await;
    let rows = direct.query("SELECT d.rank,d.mal_id,d.chat_id,d.message_id,r.raw_query,r.seed_mal_id,r.bundle_id,d.delivered_at FROM arb_delivered_positions d JOIN arb_requests r ON r.id=d.request_id ORDER BY d.rank", &[]).await.unwrap();
    assert_eq!(rows.len(), 2);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.get::<_, i16>(0), (index + 1) as i16);
        assert_eq!(row.get::<_, i32>(1), [10, 11][index]);
        assert_eq!(row.get::<_, i64>(2), 73);
        assert_eq!(row.get::<_, i32>(3), 103 + index as i32);
        assert_eq!(row.get::<_, String>(4), "Общее название");
        assert_eq!(row.get::<_, Option<i32>>(5), Some(1));
        assert_eq!(
            row.get::<_, Option<String>>(6).as_deref(),
            Some(bundle.identity())
        );
        assert!(row.get::<_, std::time::SystemTime>(7) <= std::time::SystemTime::now());
    }
    drop(ctx);
    drop(db);
    drop(direct);
    admin
        .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
        .await
        .unwrap();
}
