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
        storage::SessionStore,
    },
    handlers,
    search::{QueryError, SearchIndex},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use support::{callback, dispatch, message, FakeTelegram};
use teloxide::prelude::*;

fn small_bundle() -> Arc<Bundle> {
    Arc::new(
        Bundle::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/bundle"
        ))
        .unwrap(),
    )
}
fn synthetic_bundle() -> (tempfile::TempDir, Arc<Bundle>) {
    let dir = tempfile::tempdir().unwrap();
    let base = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bundle"
    ));
    let mut catalog: Value =
        serde_json::from_slice(&std::fs::read(base.join("catalog.json")).unwrap()).unwrap();
    let anime = catalog["anime"].as_object_mut().unwrap();
    for id in 101..=106 {
        anime.insert(id.to_string(), json!({"title":"Maou Gakuin", "aliases":["ＭＡＯＵ　ＧＡＫＵＩＮ", "Maou Gakuin"], "genres":[], "score":null, "year":null, "type":null, "episodes":null, "synopsis":null}));
    }
    anime.insert("107".into(), json!({"title":"カウボーイビバップ", "aliases":["Cowboy Bebop"], "genres":[], "score":null, "year":null, "type":null, "episodes":null, "synopsis":null}));
    anime.insert("108".into(), json!({"title":"X".repeat(1000), "aliases":[], "genres":[], "score":null, "year":null, "type":null, "episodes":null, "synopsis":null}));
    for (id, title, alias) in [
        (16498, "Shingeki no Kyojin", "Attack on Titan"),
        (
            25777,
            "Shingeki no Kyojin Season 2",
            "Attack on Titan Season 2",
        ),
        (
            35760,
            "Shingeki no Kyojin Season 3",
            "Attack on Titan Season 3",
        ),
        (
            36106,
            "Shingeki no Kyojin: Lost Girls",
            "Attack on Titan: Lost Girls",
        ),
        (
            31374,
            "Shingeki! Kyojin Chuugakkou",
            "Attack on Titan: Junior High",
        ),
    ] {
        let mut aliases = vec![alias.to_owned()];
        if id == 16498 {
            aliases.extend(["Ａｔｔａｃｋ　ｏｎ　Ｔｉｔａｎ".into(), "進撃の巨人".into()]);
        }
        anime.insert(id.to_string(), json!({"title":title, "aliases":aliases, "genres":[], "score":null, "year":2013, "type":"TV", "episodes":null, "synopsis":null}));
    }
    anime.insert("40000".into(), json!({"title":"Nova-X", "aliases":["Nova Y", "Nova Z", "Alias X", "Alias Y"], "genres":[], "score":null, "year":null, "type":null, "episodes":null, "synopsis":null}));
    let long_alias = format!("English {}", "A".repeat(250));
    anime.insert("40001".into(), json!({"title":"Short canonical", "aliases":[long_alias], "genres":[], "score":null, "year":null, "type":null, "episodes":null, "synopsis":null}));
    let mut neighbors: Value =
        serde_json::from_slice(&std::fs::read(base.join("neighbors.json")).unwrap()).unwrap();
    for id in (101..=108).chain([16498, 25777, 35760, 36106, 31374, 40000, 40001]) {
        neighbors["neighbors"][id.to_string()] = json!([]);
    }
    let catalog_bytes = format!("{}\n", catalog).into_bytes();
    let neighbors_bytes = format!("{}\n", neighbors).into_bytes();
    std::fs::write(dir.path().join("catalog.json"), &catalog_bytes).unwrap();
    std::fs::write(dir.path().join("neighbors.json"), &neighbors_bytes).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(base.join("manifest.json")).unwrap()).unwrap();
    manifest["files"]["catalog.json"]["sha256"] =
        json!(format!("{:x}", Sha256::digest(&catalog_bytes)));
    manifest["files"]["neighbors.json"]["sha256"] =
        json!(format!("{:x}", Sha256::digest(&neighbors_bytes)));
    std::fs::write(dir.path().join("manifest.json"), format!("{}\n", manifest)).unwrap();
    let bundle = Arc::new(Bundle::load(dir.path()).unwrap());
    (dir, bundle)
}

#[test]
fn unicode_ranking_limits_and_validation() {
    let (_dir, bundle) = synthetic_bundle();
    let index = SearchIndex::new(bundle.catalog());
    assert_eq!(
        index.find("ＭＡＯＵ---ＧＡＫＵＩＮ").unwrap().candidates,
        vec![101, 102, 103, 104, 105]
    );
    assert!(index.find("Maou Gakuin").unwrap().has_more);
    assert_eq!(index.find("カウボーイビバップ").unwrap().candidates[0], 107);
    assert_eq!(index.find("cowboy bebop").unwrap().candidates[0], 107);
    assert_eq!(index.find("Maou Gakuim").unwrap().candidates[0], 101);
    assert_eq!(index.find("ma").unwrap().candidates, Vec::<i32>::new());
    assert_eq!(index.find("  --  "), Err(QueryError::Empty));
    assert_eq!(index.find("a\0b"), Err(QueryError::InvalidText));
    assert_eq!(index.find(&"a".repeat(257)), Err(QueryError::TooLong));
    assert_eq!(index.find(&"x".repeat(1000)), Err(QueryError::TooLong));
}

#[test]
fn search_returns_the_original_best_matching_title_for_each_ranked_id() {
    let (_dir, bundle) = synthetic_bundle();
    let index = SearchIndex::new(bundle.catalog());
    let english = index.find("Attack on Titan").unwrap();
    assert_eq!(english.candidates, vec![16498, 25777, 35760, 36106, 31374]);
    assert_eq!(english.display_title(16498), Some("Attack on Titan"));
    assert_eq!(
        english.display_title(25777),
        Some("Attack on Titan Season 2")
    );
    assert_eq!(
        english.display_title(35760),
        Some("Attack on Titan Season 3")
    );
    assert_eq!(
        english.display_title(36106),
        Some("Attack on Titan: Lost Girls")
    );
    assert_eq!(
        english.display_title(31374),
        Some("Attack on Titan: Junior High")
    );
    assert_eq!(english.display_title(107), None);

    for query in ["ＡＴＴＡＣＫ—ＯＮ—ＴＩＴＡＮ", "Attack on", "on Titan"] {
        let matches = index.find(query).unwrap();
        assert_eq!(matches.candidates[0], 16498);
        assert_eq!(matches.display_title(16498), Some("Attack on Titan"));
    }
    let typo = index.find("Attack on Ttan").unwrap();
    assert_eq!(typo.candidates, vec![16498]);
    assert_eq!(typo.display_title(16498), Some("Attack on Titan"));
    let romaji = index.find("Shingeki no Kyojin").unwrap();
    assert_eq!(romaji.candidates[0], 16498);
    assert_eq!(romaji.display_title(16498), Some("Shingeki no Kyojin"));
    let japanese = index.find("進撃の巨人").unwrap();
    assert_eq!(japanese.candidates, vec![16498]);
    assert_eq!(japanese.display_title(16498), Some("進撃の巨人"));
    assert_eq!(
        index.find("Nova").unwrap().display_title(40000),
        Some("Nova-X")
    );
    assert_eq!(
        index.find("Alias").unwrap().display_title(40000),
        Some("Alias X")
    );
    assert_eq!(
        index.find("неттакогоаниме").unwrap().display_title(16498),
        None
    );
}

#[derive(Default)]
struct FakeRepo {
    profiles: Mutex<Vec<UserProfile>>,
    queries: Mutex<HashMap<(i64, String), (String, i64)>>,
    resolutions: Mutex<Vec<(i64, i64, i32, String)>>,
    deliveries: Mutex<Vec<DeliveredPosition>>,
    fail_upsert: AtomicBool,
    fail_query_after_commit: AtomicBool,
    fail_resolve: AtomicBool,
    next_id: AtomicUsize,
}
impl Repository for FakeRepo {
    fn upsert_user<'a>(&'a self, p: &'a UserProfile) -> DbFuture<'a, ()> {
        Box::pin(async move {
            if self.fail_upsert.swap(false, Ordering::SeqCst) {
                return Err(DbError::DatabaseFailure);
            }
            self.profiles.lock().unwrap().push(p.clone());
            Ok(())
        })
    }
    fn record_query<'a>(
        &'a self,
        user: i64,
        key: &'a str,
        query: &'a str,
    ) -> DbFuture<'a, WriteOutcome<RequestId>> {
        Box::pin(async move {
            let mut rows = self.queries.lock().unwrap();
            let row_key = (user, key.to_owned());
            let outcome = if let Some((old, id)) = rows.get(&row_key) {
                if old != query {
                    return Err(DbError::Conflict);
                }
                WriteOutcome::AlreadyRecorded(*id)
            } else {
                let id = self.next_id.fetch_add(1, Ordering::SeqCst) as i64 + 1;
                rows.insert(row_key, (query.to_owned(), id));
                WriteOutcome::Created(id)
            };
            if self.fail_query_after_commit.swap(false, Ordering::SeqCst) {
                Err(DbError::DatabaseFailure)
            } else {
                Ok(outcome)
            }
        })
    }
    fn resolve_request<'a>(
        &'a self,
        user: i64,
        request: RequestId,
        mal: i32,
        bundle: &'a str,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async move {
            if self.fail_resolve.swap(false, Ordering::SeqCst) {
                return Err(DbError::DatabaseFailure);
            }
            let mut rows = self.resolutions.lock().unwrap();
            if rows
                .iter()
                .any(|(_, id, chosen, _)| *id == request && *chosen != mal)
            {
                return Err(DbError::Conflict);
            }
            if rows
                .iter()
                .any(|(_, id, chosen, _)| *id == request && *chosen == mal)
            {
                return Ok(WriteOutcome::AlreadyRecorded(()));
            }
            rows.push((user, request, mal, bundle.to_owned()));
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
            let mut rows = self.deliveries.lock().unwrap();
            if let Some(row) = rows
                .iter()
                .find(|row| row.request_id == request && row.rank == input.rank)
            {
                return if row.mal_id == input.mal_id
                    && row.chat_id == input.chat_id
                    && row.message_id == input.message_id
                {
                    Ok(WriteOutcome::AlreadyRecorded(row.id))
                } else {
                    Err(DbError::Conflict)
                };
            }
            let id = self.next_id.fetch_add(1, Ordering::SeqCst) as i64 + 1;
            rows.push(DeliveredPosition {
                id,
                request_id: request,
                rank: input.rank,
                mal_id: input.mal_id,
                chat_id: input.chat_id,
                message_id: input.message_id,
            });
            Ok(WriteOutcome::Created(id))
        })
    }
    fn list_delivered_positions<'a>(
        &'a self,
        _: i64,
        request: RequestId,
    ) -> DbFuture<'a, Vec<DeliveredPosition>> {
        Box::pin(async move {
            Ok(self
                .deliveries
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
fn fixture(bundle: Arc<Bundle>, repo: Arc<dyn Repository>) -> Arc<AppContext> {
    Arc::new(AppContext::new(bundle, repo, Arc::new(SessionStore::new())))
}
fn chosen(session: &bot::dialogue::storage::Session, mal: i32) -> (String, i32) {
    session
        .callbacks
        .iter()
        .find_map(|(token, record)| {
            if matches!(record.action, Action::Select { mal_id, .. } if mal_id == mal) {
                Some((token.clone(), record.message_id.unwrap()))
            } else {
                None
            }
        })
        .unwrap()
}
fn last_keyboard(api: &FakeTelegram) -> (String, Vec<String>) {
    let (_, fields) = api
        .decoded()
        .into_iter()
        .rev()
        .find(|(path, fields)| {
            path.ends_with("/SendMessage") && fields.contains_key("reply_markup")
        })
        .unwrap();
    let markup: Value = serde_json::from_str(&fields["reply_markup"]).unwrap();
    let buttons = markup["inline_keyboard"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row[0]["text"].as_str().unwrap().to_owned())
        .collect();
    (fields["text"].clone(), buttons)
}
async fn dispatch_result(
    bot: &Bot,
    ctx: &Arc<AppContext>,
    update: teloxide::types::Update,
) -> bool {
    matches!(
        handlers::schema()
            .dispatch(dptree::deps![bot.clone(), update, ctx.clone()])
            .await,
        std::ops::ControlFlow::Break(Ok(()))
    )
}

#[tokio::test]
async fn public_router_registers_actual_profile_and_requires_confirmation() {
    for command in ["/recommend", "/rate"] {
        let api = FakeTelegram::new();
        let bot = api.bot();
        let repo = Arc::new(FakeRepo::default());
        let ctx = fixture(small_bundle(), repo.clone());
        dispatch(&bot, &ctx, message(1, 7, 42, command)).await;
        dispatch(&bot, &ctx, message(2, 7, 42, "  Общее  название  ")).await;
        assert_eq!(repo.profiles.lock().unwrap()[0].tg_id, 42);
        {
            let queries = repo.queries.lock().unwrap();
            assert_eq!(queries[&(42, "msg:7:2".into())].0, "  Общее  название  ");
        }
        let actor = Actor {
            chat_id: 7,
            user_id: 42,
        };
        let session = ctx.sessions.get(actor);
        let guard = session.lock().await;
        assert!(
            matches!(guard.state, State::ChoosingAnime { ref candidates, selected_mal_id: None, .. } if candidates == &vec![1,2])
        );
        let (token, msg_id) = chosen(&guard, 1);
        drop(guard);
        assert!(api.decoded().iter().any(|(_, fields)| fields
            .get("reply_markup")
            .is_some_and(|v| v.contains("MAL ID"))));
        assert!(repo.resolutions.lock().unwrap().is_empty());
        dispatch(&bot, &ctx, callback(3, 7, 42, msg_id, Some(&token))).await;
        assert_eq!(repo.resolutions.lock().unwrap().len(), 1);
        assert_eq!(repo.resolutions.lock().unwrap()[0].2, 1);
        assert_eq!(repo.resolutions.lock().unwrap()[0].3, ctx.bundle.identity());
        dispatch(&bot, &ctx, callback(4, 7, 42, msg_id, Some(&token))).await;
        assert_eq!(repo.resolutions.lock().unwrap().len(), 1);
        if command == "/recommend" {
            let rows = repo.deliveries.lock().unwrap().clone();
            assert_eq!(
                rows.iter().map(|row| row.mal_id).collect::<Vec<_>>(),
                vec![10, 11, 12, 2, 3]
            );
            assert!(rows.iter().enumerate().all(|(index, row)| {
                row.rank == (index + 1) as i16 && row.chat_id == 7 && row.message_id > 0
            }));
            dispatch(&bot, &ctx, callback(5, 7, 43, msg_id, Some(&token))).await;
            dispatch(&bot, &ctx, message(6, 7, 42, "/cancel")).await;
            dispatch(&bot, &ctx, callback(7, 7, 42, msg_id, Some(&token))).await;
            assert_eq!(repo.deliveries.lock().unwrap().as_slice(), rows.as_slice());
        }
    }
}

#[tokio::test]
async fn both_intents_confirm_with_the_matched_title_and_stable_id() {
    for command in ["/recommend", "/rate"] {
        for (query, expected_ids, expected_titles) in [
            (
                "Attack on titan",
                vec![16498, 25777, 35760, 36106, 31374],
                vec![
                    "Attack on Titan",
                    "Attack on Titan Season 2",
                    "Attack on Titan Season 3",
                    "Attack on Titan: Lost Girls",
                    "Attack on Titan: Junior High",
                ],
            ),
            (
                "Shingeki no Kyojin",
                vec![16498, 25777, 35760, 36106],
                vec![
                    "Shingeki no Kyojin",
                    "Shingeki no Kyojin Season 2",
                    "Shingeki no Kyojin Season 3",
                    "Shingeki no Kyojin: Lost Girls",
                ],
            ),
            ("進撃の巨人", vec![16498], vec!["進撃の巨人"]),
        ] {
            let (_dir, bundle) = synthetic_bundle();
            let api = FakeTelegram::new();
            let bot = api.bot();
            let repo = Arc::new(FakeRepo::default());
            let ctx = fixture(bundle, repo.clone());
            dispatch(&bot, &ctx, message(1, 7, 42, command)).await;
            dispatch(&bot, &ctx, message(2, 7, 42, query)).await;
            let (text, buttons) = last_keyboard(&api);
            assert!(text.starts_with("Подтвердите аниме:\n"));
            assert_eq!(buttons.len(), expected_ids.len());
            for (index, ((mal_id, title), button)) in expected_ids
                .iter()
                .zip(expected_titles.iter())
                .zip(buttons.iter())
                .enumerate()
            {
                assert!(
                    text.contains(&format!(
                        "{}. {} (2013, TV) · MAL ID {mal_id}",
                        index + 1,
                        title
                    )),
                    "{text}"
                );
                assert_eq!(
                    button,
                    &format!("{}. {} · MAL ID {mal_id}", index + 1, title)
                );
            }
            let actor = Actor {
                chat_id: 7,
                user_id: 42,
            };
            let session = ctx.sessions.get(actor);
            let guard = session.lock().await;
            assert!(
                matches!(&guard.state, State::ChoosingAnime { intent, query: context, candidates, selected_mal_id: None }
                if *intent == if command == "/recommend" { AnimeIntent::Recommend } else { AnimeIntent::Rate }
                && context.raw_query == query && candidates == &expected_ids)
            );
            let selected_id = if query == "Attack on titan" {
                25777
            } else {
                16498
            };
            let (token, message_id) = chosen(&guard, selected_id);
            drop(guard);
            assert!(repo.resolutions.lock().unwrap().is_empty());
            dispatch(&bot, &ctx, callback(3, 7, 42, message_id, Some(&token))).await;
            let resolutions = repo.resolutions.lock().unwrap();
            assert_eq!(resolutions.len(), 1);
            assert_eq!(resolutions[0].2, selected_id);
            assert_eq!(resolutions[0].3, ctx.bundle.identity());
        }
    }
}

#[tokio::test]
async fn long_alias_has_independent_message_and_button_bounds() {
    let (_dir, bundle) = synthetic_bundle();
    let api = FakeTelegram::new();
    let bot = api.bot();
    let ctx = fixture(bundle, Arc::new(FakeRepo::default()));
    dispatch(&bot, &ctx, message(1, 7, 42, "/rate")).await;
    dispatch(
        &bot,
        &ctx,
        message(2, 7, 42, &format!("English {}", "A".repeat(240))),
    )
    .await;
    let (text, buttons) = last_keyboard(&api);
    assert!(text.contains(&format!(
        "1. English {}… (—, —) · MAL ID 40001",
        "A".repeat(231)
    )));
    assert_eq!(
        buttons,
        vec![format!("1. English {}… · MAL ID 40001", "A".repeat(41))]
    );
    let matches = ctx
        .search
        .find(&format!("English {}", "A".repeat(240)))
        .unwrap();
    assert_eq!(
        matches.display_title(40001).unwrap(),
        format!("English {}", "A".repeat(250))
    );
}

#[tokio::test]
async fn failed_candidate_send_retries_with_the_original_alias_and_request() {
    let (_dir, bundle) = synthetic_bundle();
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(bundle, repo.clone());
    dispatch(&bot, &ctx, message(1, 7, 42, "/recommend")).await;
    api.fail_send.store(true, Ordering::SeqCst);
    assert!(!dispatch_result(&bot, &ctx, message(2, 7, 42, "Attack on Titan")).await);
    assert_eq!(repo.queries.lock().unwrap().len(), 1);
    let session = ctx.sessions.get(Actor {
        chat_id: 7,
        user_id: 42,
    });
    let (token, message_id) = {
        let guard = session.lock().await;
        assert!(matches!(guard.state, State::PendingQuery { .. }));
        guard
            .callbacks
            .iter()
            .find_map(|(token, record)| {
                matches!(record.action, Action::RetryQuery { .. })
                    .then_some((token.clone(), record.message_id.unwrap()))
            })
            .unwrap()
    };
    dispatch(&bot, &ctx, callback(3, 7, 42, message_id, Some(&token))).await;
    assert_eq!(repo.queries.lock().unwrap().len(), 1);
    let (text, buttons) = last_keyboard(&api);
    assert!(text.contains("1. Attack on Titan (2013, TV) · MAL ID 16498"));
    assert_eq!(buttons[0], "1. Attack on Titan · MAL ID 16498");
}

#[tokio::test]
async fn uncertain_query_retry_reuses_original_message_and_pin_survives_send_failure() {
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(small_bundle(), repo.clone());
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    dispatch(&bot, &ctx, message(1, 1, 1, "/rate")).await;
    repo.fail_query_after_commit.store(true, Ordering::SeqCst);
    assert!(!dispatch_result(&bot, &ctx, message(2, 1, 1, "Общее название")).await);
    let session = ctx.sessions.get(actor);
    let (retry, retry_id) = {
        let guard = session.lock().await;
        assert!(matches!(guard.state, State::PendingQuery { .. }));
        guard
            .callbacks
            .iter()
            .find_map(|(token, rec)| {
                matches!(rec.action, Action::RetryQuery { .. })
                    .then_some((token.clone(), rec.message_id.unwrap()))
            })
            .unwrap()
    };
    dispatch(&bot, &ctx, callback(3, 1, 1, retry_id, Some(&retry))).await;
    assert_eq!(repo.queries.lock().unwrap().len(), 1);
    let (first, first_id, second) = {
        let guard = session.lock().await;
        let (first, first_id) = chosen(&guard, 1);
        let (second, _) = chosen(&guard, 2);
        (first, first_id, second)
    };
    api.fail_send.store(true, Ordering::SeqCst);
    assert!(!dispatch_result(&bot, &ctx, callback(4, 1, 1, first_id, Some(&first))).await);
    assert!(matches!(
        session.lock().await.state,
        State::ChoosingAnime {
            selected_mal_id: Some(1),
            ..
        }
    ));
    dispatch(&bot, &ctx, callback(5, 1, 1, first_id, Some(&second))).await;
    assert_eq!(repo.resolutions.lock().unwrap().len(), 1);
    assert_eq!(repo.resolutions.lock().unwrap()[0].2, 1);
    dispatch(&bot, &ctx, callback(6, 1, 1, first_id, Some(&first))).await;
    assert!(matches!(
        session.lock().await.state,
        State::Selected {
            intent: AnimeIntent::Rate,
            ..
        }
    ));
}

#[tokio::test]
async fn failed_profile_and_candidate_send_recover_with_same_query() {
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(small_bundle(), repo.clone());
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    dispatch(&bot, &ctx, message(1, 1, 1, "/recommend")).await;
    repo.fail_upsert.store(true, Ordering::SeqCst);
    assert!(!dispatch_result(&bot, &ctx, message(2, 1, 1, "Общее название")).await);
    assert!(repo.queries.lock().unwrap().is_empty());
    let session = ctx.sessions.get(actor);
    let (retry, retry_id) = {
        let guard = session.lock().await;
        guard
            .callbacks
            .iter()
            .find_map(|(token, rec)| {
                matches!(rec.action, Action::RetryQuery { .. })
                    .then_some((token.clone(), rec.message_id.unwrap()))
            })
            .unwrap()
    };
    repo.fail_upsert.store(true, Ordering::SeqCst);
    assert!(!dispatch_result(&bot, &ctx, callback(3, 1, 1, retry_id, Some(&retry))).await);
    {
        let guard = session.lock().await;
        assert_eq!(
            guard
                .callbacks
                .values()
                .filter(|rec| matches!(rec.action, Action::RetryQuery { .. }))
                .count(),
            1
        );
        assert_eq!(
            guard.callbacks[&retry].status,
            bot::dialogue::storage::CallbackStatus::Active
        );
    }
    assert!(repo.queries.lock().unwrap().is_empty());
    api.fail_send.store(true, Ordering::SeqCst);
    assert!(!dispatch_result(&bot, &ctx, callback(4, 1, 1, retry_id, Some(&retry))).await);
    assert!(matches!(
        session.lock().await.state,
        State::PendingQuery { .. }
    ));
    assert_eq!(
        session
            .lock()
            .await
            .callbacks
            .values()
            .filter(|rec| matches!(rec.action, Action::Select { .. }))
            .count(),
        0
    );
    dispatch(&bot, &ctx, callback(5, 1, 1, retry_id, Some(&retry))).await;
    assert_eq!(repo.queries.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn invalid_no_match_help_cancel_and_fresh_query_are_recoverable() {
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(small_bundle(), repo.clone());
    let actor = Actor {
        chat_id: 1,
        user_id: 1,
    };
    dispatch(&bot, &ctx, message(1, 1, 1, "/rate")).await;
    for (id, query) in [(2, "  !!! ".to_owned()), (3, "a".repeat(257))] {
        dispatch(&bot, &ctx, message(id, 1, 1, &query)).await;
    }
    assert!(repo.queries.lock().unwrap().is_empty());
    dispatch(&bot, &ctx, message(4, 1, 1, "неттакогоаниме")).await;
    assert!(matches!(
        ctx.sessions.get(actor).lock().await.state,
        State::AwaitingQuery { .. }
    ));
    dispatch(&bot, &ctx, message(5, 1, 1, "Общее название")).await;
    let old = {
        let guard = ctx.sessions.get(actor);
        let guard = guard.lock().await;
        chosen(&guard, 1)
    };
    dispatch(&bot, &ctx, message(6, 1, 1, "/help")).await;
    assert!(matches!(
        ctx.sessions.get(actor).lock().await.state,
        State::ChoosingAnime { .. }
    ));
    dispatch(&bot, &ctx, message(7, 1, 1, "Северный ветер")).await;
    dispatch(&bot, &ctx, callback(8, 1, 1, old.1, Some(&old.0))).await;
    assert!(repo.resolutions.lock().unwrap().is_empty());
    dispatch(&bot, &ctx, message(9, 1, 1, "/cancel")).await;
    assert!(matches!(
        ctx.sessions.get(actor).lock().await.state,
        State::Idle
    ));
}

#[tokio::test]
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn postgres_public_router_records_profile_query_and_stable_resolution() {
    use std::str::FromStr;
    use tokio_postgres::{Config, NoTls};
    let url = std::env::var("ARB_TEST_DATABASE_URL").expect("set ARB_TEST_DATABASE_URL explicitly");
    let mut config = Config::from_str(&url).expect("test database URL");
    let name = config.get_dbname().expect("test database name");
    assert!(
        name == "arb_013_test" || name == "arb_ci_test",
        "unsafe ARB013 test database name"
    );
    let (admin, connection) = config.connect(NoTls).await.unwrap();
    tokio::spawn(async move {
        connection.await.unwrap();
    });
    let schema = format!("arb013_{}", std::process::id());
    admin
        .batch_execute(&format!("CREATE SCHEMA {schema}"))
        .await
        .unwrap();
    config.options(format!("-c search_path={schema}"));
    let (direct, connection) = config.connect(NoTls).await.unwrap();
    tokio::spawn(async move {
        connection.await.unwrap();
    });
    let db = Db::new(&config).await.unwrap();
    db.migrate().await.unwrap();
    let api = FakeTelegram::new();
    let bot = api.bot();
    let bundle = small_bundle();
    let expected_bundle = bundle.identity().to_owned();
    let ctx = fixture(bundle, Arc::new(db));
    dispatch(&bot, &ctx, message(1, 73, 42, "/rate")).await;
    let mut query_update = message(2, 73, 42, "  Общее название  ");
    if let teloxide::types::UpdateKind::Message(ref mut message) = query_update.kind {
        let user = message.from.as_mut().unwrap();
        user.first_name = "Настоящее имя".into();
        user.last_name = Some("Фамилия".into());
        user.username = Some("anime42".into());
        user.language_code = Some("ru".into());
    }
    dispatch(&bot, &ctx, query_update.clone()).await;
    let actor = Actor {
        chat_id: 73,
        user_id: 42,
    };
    let session = ctx.sessions.get(actor);
    let (first, msg_id, second) = {
        let guard = session.lock().await;
        let (first, msg_id) = chosen(&guard, 1);
        let (second, _) = chosen(&guard, 2);
        (first, msg_id, second)
    };
    dispatch(&bot, &ctx, query_update).await;
    let row = direct
        .query_one(
            "SELECT first_name,last_name,username,language_code FROM arb_users WHERE tg_id=42",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, String>(0), "Настоящее имя");
    assert_eq!(row.get::<_, Option<String>>(1).as_deref(), Some("Фамилия"));
    assert_eq!(row.get::<_, Option<String>>(2).as_deref(), Some("anime42"));
    assert_eq!(row.get::<_, Option<String>>(3).as_deref(), Some("ru"));
    let rows = direct
        .query(
            "SELECT action_key,raw_query,seed_mal_id,bundle_id FROM arb_requests WHERE tg_id=42",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, String>(0), "msg:73:2");
    assert_eq!(rows[0].get::<_, String>(1), "  Общее название  ");
    assert_eq!(rows[0].get::<_, Option<i32>>(2), None);
    api.fail_send.store(true, Ordering::SeqCst);
    assert!(!dispatch_result(&bot, &ctx, callback(3, 73, 42, msg_id, Some(&first))).await);
    dispatch(&bot, &ctx, callback(4, 73, 42, msg_id, Some(&second))).await;
    dispatch(&bot, &ctx, callback(5, 73, 42, msg_id, Some(&first))).await;
    let row = direct
        .query_one(
            "SELECT seed_mal_id,bundle_id FROM arb_requests WHERE tg_id=42",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, Option<i32>>(0), Some(1));
    assert_eq!(
        row.get::<_, Option<String>>(1).as_deref(),
        Some(expected_bundle.as_str())
    );
    assert_eq!(
        direct
            .query_one("SELECT count(*) FROM arb_requests WHERE tg_id=42", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        1
    );
    drop(ctx);
    drop(direct);
    admin
        .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
        .await
        .unwrap();
}

#[tokio::test]
async fn duplicate_titles_limit_to_five_and_singleton_still_requires_a_button() {
    let (_dir, bundle) = synthetic_bundle();
    let api = FakeTelegram::new();
    let bot = api.bot();
    let repo = Arc::new(FakeRepo::default());
    let ctx = fixture(bundle, repo.clone());
    let actor = Actor {
        chat_id: 5,
        user_id: 6,
    };
    dispatch(&bot, &ctx, message(1, 5, 6, "/recommend")).await;
    dispatch(&bot, &ctx, message(2, 5, 6, "Maou Gakuin")).await;
    let session = ctx.sessions.get(actor);
    let (old, old_msg) = {
        let guard = session.lock().await;
        assert!(
            matches!(guard.state, State::ChoosingAnime { ref candidates, .. } if candidates == &vec![101,102,103,104,105])
        );
        chosen(&guard, 101)
    };
    assert!(api.decoded().iter().any(|(_, fields)| fields
        .get("text")
        .is_some_and(|text| text.contains("Показаны первые 5 совпадений"))));
    dispatch(&bot, &ctx, message(2, 5, 6, "changed text")).await;
    assert!(matches!(
        session.lock().await.state,
        State::ChoosingAnime { .. }
    ));
    assert_eq!(repo.queries.lock().unwrap().len(), 1);
    dispatch(&bot, &ctx, message(3, 5, 6, "カウボーイビバップ")).await;
    assert!(
        matches!(session.lock().await.state, State::ChoosingAnime { ref candidates, .. } if candidates == &vec![107])
    );
    assert!(repo.resolutions.lock().unwrap().is_empty());
    dispatch(&bot, &ctx, callback(4, 5, 6, old_msg, Some(&old))).await;
    assert!(repo.resolutions.lock().unwrap().is_empty());
    let (single, message_id) = {
        let guard = session.lock().await;
        chosen(&guard, 107)
    };
    dispatch(&bot, &ctx, callback(5, 5, 7, message_id, Some(&single))).await;
    assert!(repo.resolutions.lock().unwrap().is_empty());
    dispatch(&bot, &ctx, callback(6, 5, 6, message_id, Some(&single))).await;
    assert_eq!(repo.resolutions.lock().unwrap()[0].2, 107);
}
