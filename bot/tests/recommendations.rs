#[allow(dead_code)]
mod support;
use bot::{
    catalog::Bundle,
    db::{
        AnimeRatingEventId, Db, DbError, DeliveredPosition, DeliveryInput, PositionId, RequestId,
        UserProfile, WriteOutcome,
    },
    dialogue::{
        callback::{Action, DescriptionView},
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

struct FixedCovers;
impl bot::covers::CoverProvider for FixedCovers {
    fn lookup(&self, id: i32) -> bot::covers::CoverFuture<'_> {
        Box::pin(async move {
            reqwest::Url::parse(&format!(
                "https://cdn.myanimelist.net/images/anime/{id}.jpg"
            ))
            .ok()
        })
    }
}

const ACTOR: Actor = Actor {
    chat_id: 73,
    user_id: 42,
};

fn bundle(count: usize, long: bool) -> (tempfile::TempDir, Arc<Bundle>) {
    bundle_with(count, long, |_| {})
}

fn bundle_with(
    count: usize,
    long: bool,
    edit_catalog: impl FnOnce(&mut Value),
) -> (tempfile::TempDir, Arc<Bundle>) {
    let dir = tempfile::tempdir().unwrap();
    let base = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bundle"
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
    edit_catalog(&mut catalog);
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

fn english_bundle() -> (tempfile::TempDir, Arc<Bundle>) {
    bundle_with(5, false, |catalog| {
        catalog["anime"]["1"]["title"] = json!("Shingeki no Kyojin");
        catalog["anime"]["1"]["aliases"] = json!([
            "Attack on Titan",
            "Ａｔｔａｃｋ　ｏｎ　Ｔｉｔａｎ",
            "進撃の巨人"
        ]);
        catalog["anime"]["10"]["title"] = json!("Koutetsujou no Kabaneri");
        catalog["anime"]["10"]["aliases"] = json!(["Kabaneri of the Iron Fortress"]);
        catalog["anime"]["11"]["title"] = json!("Karas");
        catalog["anime"]["11"]["aliases"] = json!(["鴉 -KARAS-", "Crow"]);
        catalog["anime"]["12"]["title"] = json!("Naruto");
        catalog["anime"]["12"]["aliases"] = json!([]);
        catalog["anime"]["2"]["title"] = json!("Romaji Two");
        catalog["anime"]["2"]["aliases"] = json!(["日本語", "English Two"]);
        catalog["anime"]["3"]["title"] = json!("No Alias");
        catalog["anime"]["3"]["aliases"] = json!([]);
    })
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
    scores: Mutex<Vec<(i64, PositionId, i16)>>,
    score_calls: Mutex<Vec<(i64, PositionId, i16)>>,
    score_failure: AtomicU8, // 1 before commit, 2 after commit, 3 terminal
    fail_list: AtomicBool,
    write_failure: AtomicU8, // 1 before commit, 2 after commit
    fail_send_after_record: Mutex<Option<Arc<AtomicBool>>>,
    feedback: Mutex<Vec<(i64, String, String)>>,
    allow_feedback: AtomicBool,
}
impl Repository for FakeRepo {
    fn upsert_user<'a>(&'a self, _: &'a UserProfile) -> DbFuture<'a, ()> {
        Box::pin(async {
            assert!(
                self.allow_feedback.load(Ordering::SeqCst),
                "unexpected profile upsert"
            );
            Ok(())
        })
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
        user: i64,
        position: PositionId,
        score: i16,
    ) -> DbFuture<'a, WriteOutcome<()>> {
        Box::pin(async move {
            self.score_calls
                .lock()
                .unwrap()
                .push((user, position, score));
            let failure = self.score_failure.swap(0, Ordering::SeqCst);
            if failure == 1 {
                return Err(DbError::DatabaseFailure);
            }
            if failure == 3 {
                return Err(DbError::NotFound);
            }
            if !(0..=5).contains(&score) {
                return Err(DbError::InvalidInput("score"));
            }
            let mut rows = self.scores.lock().unwrap();
            let result = if let Some((old_user, _, old_score)) =
                rows.iter().find(|(_, id, _)| *id == position)
            {
                if *old_user != user || *old_score != score {
                    return Err(DbError::Conflict);
                }
                WriteOutcome::AlreadyRecorded(())
            } else {
                rows.push((user, position, score));
                WriteOutcome::Created(())
            };
            if failure == 2 {
                Err(DbError::DatabaseFailure)
            } else {
                Ok(result)
            }
        })
    }
    fn save_feedback<'a>(
        &'a self,
        user: i64,
        key: &'a str,
        body: &'a str,
    ) -> DbFuture<'a, WriteOutcome<i64>> {
        Box::pin(async move {
            assert!(
                self.allow_feedback.load(Ordering::SeqCst),
                "unexpected feedback"
            );
            let mut rows = self.feedback.lock().unwrap();
            if let Some((index, (_, _, previous))) = rows
                .iter()
                .enumerate()
                .find(|(_, (u, k, _))| *u == user && k == key)
            {
                return if previous == body {
                    Ok(WriteOutcome::AlreadyRecorded(index as i64 + 1))
                } else {
                    Err(DbError::Conflict)
                };
            }
            rows.push((user, key.to_owned(), body.to_owned()));
            Ok(WriteOutcome::Created(rows.len() as i64))
        })
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
            let edits: Vec<_> = api
                .decoded()
                .into_iter()
                .filter(|(path, _)| path.ends_with("/EditMessageReplyMarkup"))
                .collect();
            assert_eq!(edits.len(), count);
            assert!(repo.scores.lock().unwrap().is_empty());
            for (index, (row, text)) in rows.iter().zip(&texts).enumerate() {
                let fields = &edits[index].1;
                assert_eq!(fields.get("chat_id").unwrap(), &row.chat_id.to_string());
                assert_eq!(
                    fields.get("message_id").unwrap(),
                    &row.message_id.to_string()
                );
                let markup: Value =
                    serde_json::from_str(fields.get("reply_markup").unwrap()).unwrap();
                let buttons = markup["inline_keyboard"][0].as_array().unwrap();
                assert_eq!(buttons.len(), 6);
                for (score, button) in buttons.iter().enumerate() {
                    assert_eq!(button["text"], score.to_string());
                    let token = button["callback_data"].as_str().unwrap();
                    assert!(token.starts_with("a1:") && token.len() == 35);
                    let record = session.callbacks.get(token).unwrap();
                    assert_eq!(record.message_id, Some(row.message_id));
                    assert!(
                        matches!(record.action, Action::RecommendationScore { position_id, score: value } if position_id == row.id && value == score as i16)
                    );
                }
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
            assert_eq!(
                api.decoded()
                    .iter()
                    .filter(|(path, _)| path.ends_with("/EditMessageReplyMarkup"))
                    .count(),
                count
            );
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
        .filter(|(path, _)| path.ends_with("/SendMessage"))
        .all(|(_, fields)| !fields.contains_key("parse_mode")
            && !fields.contains_key("reply_markup")));
}

#[tokio::test]
async fn selected_spelling_controls_seed_and_neighbor_titles() {
    for (query, seed, target) in [
        (
            "Attack on Titan",
            "Attack on Titan",
            "Kabaneri of the Iron Fortress",
        ),
        (
            "attack on titan",
            "Attack on Titan",
            "Kabaneri of the Iron Fortress",
        ),
        (
            "ＡＴＴＡＣＫ—ＯＮ—ＴＩＴＡＮ",
            "Attack on Titan",
            "Kabaneri of the Iron Fortress",
        ),
        (
            "Attack on",
            "Attack on Titan",
            "Kabaneri of the Iron Fortress",
        ),
        (
            "Attack on Ttan",
            "Attack on Titan",
            "Kabaneri of the Iron Fortress",
        ),
        (
            "Shingeki no Kyojin",
            "Shingeki no Kyojin",
            "Koutetsujou no Kabaneri",
        ),
        ("進撃の巨人", "進撃の巨人", "Koutetsujou no Kabaneri"),
        (
            "unmatched query",
            "Shingeki no Kyojin",
            "Koutetsujou no Kabaneri",
        ),
    ] {
        let (_dir, bundle) = english_bundle();
        let ctx = context(bundle.clone(), Arc::new(FakeRepo::default()));
        let api = FakeTelegram::new();
        let mut sel = selection(&bundle, 1, 1);
        sel.query.raw_query = query.into();
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session(&sel), &sel)
            .await
            .unwrap();
        let cards = sent_texts(&api);
        assert_eq!(cards.len(), 5, "{query}");
        assert!(
            cards[0].contains(&format!("По запросу: {seed}\nНазвание: {target}\n")),
            "{query}: {}",
            cards[0]
        );
        assert!(cards[1].contains("Название: Karas\n"), "{query}");
        assert!(cards[2].contains("Название: Naruto\n"), "{query}");
        assert!(cards[3].contains("Название: Romaji Two\n"), "{query}");
        assert!(cards[4].contains("Название: No Alias\n"), "{query}");
        assert!(cards.iter().all(|card| card.encode_utf16().count() <= 4000));
    }
}

#[tokio::test]
async fn english_canonical_without_alias_enables_neighbor_preference() {
    let (_dir, bundle) = bundle_with(1, false, |catalog| {
        catalog["anime"]["1"]["title"] = json!("Cowboy Bebop");
        catalog["anime"]["1"]["aliases"] = json!(["カウボーイビバップ"]);
        catalog["anime"]["10"]["title"] = json!("Koutetsujou no Kabaneri");
        catalog["anime"]["10"]["aliases"] = json!(["Kabaneri of the Iron Fortress"]);
    });
    let ctx = context(bundle.clone(), Arc::new(FakeRepo::default()));
    let api = FakeTelegram::new();
    let mut sel = selection(&bundle, 1, 1);
    sel.query.raw_query = "Cowboy Bebop".into();
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session(&sel), &sel)
        .await
        .unwrap();
    assert!(sent_texts(&api)[0]
        .contains("По запросу: Cowboy Bebop\nНазвание: Kabaneri of the Iron Fortress"));
}

#[tokio::test]
async fn duplicate_normalized_spellings_and_ambiguous_latin_canonical_are_deterministic() {
    for (canonical, aliases, expected_seed) in [
        (
            "Ａｔｔａｃｋ　ｏｎ　Ｔｉｔａｎ",
            json!(["Attack on Titan"]),
            "Ａｔｔａｃｋ　ｏｎ　Ｔｉｔａｎ",
        ),
        ("Attack on Titan", json!([]), "Attack on Titan"),
    ] {
        let (_dir, bundle) = bundle_with(1, false, |catalog| {
            catalog["anime"]["1"]["title"] = json!(canonical);
            catalog["anime"]["1"]["aliases"] = aliases;
            catalog["anime"]["10"]["title"] = json!("Koutetsujou no Kabaneri");
            catalog["anime"]["10"]["aliases"] = json!(["Kabaneri of the Iron Fortress"]);
        });
        let ctx = context(bundle.clone(), Arc::new(FakeRepo::default()));
        let api = FakeTelegram::new();
        let mut sel = selection(&bundle, 1, 1);
        sel.query.raw_query = "Attack on Titan".into();
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session(&sel), &sel)
            .await
            .unwrap();
        let card = &sent_texts(&api)[0];
        assert!(
            card.contains(&format!(
                "По запросу: {expected_seed}\nНазвание: Kabaneri of the Iron Fortress"
            )),
            "{card}"
        );
    }
}

#[tokio::test]
async fn long_matched_seed_and_neighbor_aliases_keep_independent_bounds() {
    let seed_alias = format!("English {}", "A".repeat(240));
    let neighbor_alias = format!("Target {}", "B".repeat(400));
    let (_dir, bundle) = bundle_with(1, false, |catalog| {
        catalog["anime"]["1"]["title"] = json!("Seed Romaji");
        catalog["anime"]["1"]["aliases"] = json!([seed_alias]);
        catalog["anime"]["10"]["aliases"] = json!([neighbor_alias]);
    });
    let ctx = context(bundle.clone(), Arc::new(FakeRepo::default()));
    let api = FakeTelegram::new();
    let mut sel = selection(&bundle, 1, 1);
    sel.query.raw_query = seed_alias.clone();
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session(&sel), &sel)
        .await
        .unwrap();
    let card = &sent_texts(&api)[0];
    let shown_seed = card
        .lines()
        .find_map(|line| line.strip_prefix("По запросу: "))
        .unwrap();
    let shown_target = card
        .lines()
        .find_map(|line| line.strip_prefix("Название: "))
        .unwrap();
    assert!(shown_seed.encode_utf16().count() <= 160);
    assert!(shown_target.encode_utf16().count() <= 240);
    assert!(shown_seed.starts_with("English A") && shown_seed.ends_with('…'));
    assert!(shown_target.starts_with("Target B") && shown_target.ends_with('…'));
    assert_eq!(sel.query.raw_query, seed_alias);
    assert!(card.encode_utf16().count() <= 4000);
}

#[tokio::test]
async fn usefulness_retry_retains_english_target_title() {
    let (_dir, bundle) = english_bundle();
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone());
    let api = FakeTelegram::new();
    let mut sel = selection(&bundle, 1, 1);
    sel.query.raw_query = "Attack on Titan".into();
    let mut session = session(&sel);
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
        .await
        .unwrap();
    let row = repo.rows.lock().unwrap()[0].clone();
    let token = score_token(&session, row.id, 4);
    let key = session.claim(&token, row.message_id).unwrap().1;
    repo.score_failure.store(2, Ordering::SeqCst);
    let action = || recommendations::ScoreAction {
        position: &row,
        score: 4,
        action_key: &key,
    };
    assert!(
        recommendations::on_score(&api.bot(), &ctx, ACTOR, &mut session, &sel, action())
            .await
            .is_err()
    );
    session.finish(&token, false);
    assert!(session.claim(&token, row.message_id).is_some());
    recommendations::on_score(&api.bot(), &ctx, ACTOR, &mut session, &sel, action())
        .await
        .unwrap();
    let confirmation = sent_texts(&api).last().unwrap().clone();
    assert!(confirmation.contains("Kabaneri of the Iron Fortress"));
    assert!(!confirmation.contains("Koutetsujou"));
    assert_eq!(repo.scores.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn usefulness_confirmation_handles_missing_seed_or_target() {
    let (_dir, bundle) = english_bundle();
    let ctx = context(bundle.clone(), Arc::new(FakeRepo::default()));
    for (seed_mal_id, target_mal_id, expected_title) in
        [(1, 999, "Аниме"), (999, 10, "Koutetsujou no Kabaneri")]
    {
        let api = FakeTelegram::new();
        let mut sel = selection(&bundle, 1, seed_mal_id);
        sel.query.raw_query = "Attack on Titan".into();
        let row = DeliveredPosition {
            id: i64::from(target_mal_id),
            request_id: 1,
            rank: 1,
            mal_id: target_mal_id,
            chat_id: ACTOR.chat_id,
            message_id: 101,
        };
        recommendations::on_score(
            &api.bot(),
            &ctx,
            ACTOR,
            &mut session(&sel),
            &sel,
            recommendations::ScoreAction {
                position: &row,
                score: 3,
                action_key: "test",
            },
        )
        .await
        .unwrap();
        assert!(sent_texts(&api)[0].contains(&format!("«{expected_title}»")));
    }
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
        assert!(session
            .callbacks
            .values()
            .all(|record| !matches!(record.action, Action::RecommendationScore { .. })));
        assert!(api
            .decoded()
            .iter()
            .all(|(path, _)| !path.ends_with("/EditMessageReplyMarkup")));
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
    assert!(session
        .callbacks
        .values()
        .filter_map(|record| {
            if let Action::RecommendationScore { position_id, .. } = record.action {
                Some(position_id)
            } else {
                None
            }
        })
        .all(|position_id| rows
            .iter()
            .any(|row| row.id == position_id && row.request_id == 2)));
    assert!(repo
        .resolves
        .lock()
        .unwrap()
        .contains(&(1, 1, first.bundle_id)));
}

#[tokio::test]
async fn feedback_navigation_preserves_confirmed_recommendation_coordinates() {
    let (_dir, bundle) = bundle(2, false);
    let repo = Arc::new(FakeRepo::default());
    repo.allow_feedback.store(true, Ordering::SeqCst);
    let ctx = Arc::new(context(bundle.clone(), repo.clone()));
    let api = FakeTelegram::new();
    let bot = api.bot();
    let sel = selection(&bundle, 1, 1);
    let session = ctx.sessions.get(ACTOR);
    {
        let mut guard = session.lock().await;
        guard.state = State::Selected {
            intent: AnimeIntent::Recommend,
            selection: sel.clone(),
        };
        repo.write_failure.store(1, Ordering::SeqCst);
        assert!(recommendations::begin(&bot, &ctx, ACTOR, &mut guard, &sel)
            .await
            .is_err());
    }
    dispatch(&bot, &ctx, message(10, 73, 42, "/feedback")).await;
    dispatch(&bot, &ctx, message(11, 73, 42, "Отзыв после карточки")).await;
    assert_eq!(
        repo.feedback.lock().unwrap().as_slice(),
        &[(42, "msg:73:11".into(), "Отзыв после карточки".into())]
    );
    {
        let mut guard = session.lock().await;
        guard.state = State::Selected {
            intent: AnimeIntent::Recommend,
            selection: sel.clone(),
        };
        recommendations::begin(&bot, &ctx, ACTOR, &mut guard, &sel)
            .await
            .unwrap();
    }
    let rows = repo.rows.lock().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].message_id, 101);
    assert_eq!(
        sent_texts(&api)
            .iter()
            .filter(|text| text.starts_with("Рекомендация 1/"))
            .count(),
        1
    );
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

fn score_token(session: &Session, position: PositionId, score: i16) -> String {
    session
        .callbacks
        .iter()
        .find_map(|(token, record)| {
            matches!(record.action, Action::RecommendationScore { position_id, score: value }
            if position_id == position && value == score)
            .then(|| token.clone())
        })
        .unwrap()
}

#[tokio::test]
async fn markup_failure_keeps_durable_card_and_retries_edit() {
    let (_dir, bundle) = bundle(2, false);
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone());
    let api = FakeTelegram::new();
    let sel = selection(&bundle, 1, 1);
    let mut session = session(&sel);
    api.fail_edit.store(true, Ordering::SeqCst);
    assert!(
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .is_err()
    );
    let first = repo.rows.lock().unwrap()[0].clone();
    assert_eq!(first.message_id, 101);
    assert!(session
        .callbacks
        .values()
        .all(|record| !matches!(record.action, Action::RecommendationScore { .. })));
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
        .await
        .unwrap();
    let rows = repo.rows.lock().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], first);
    assert_eq!(rows[1].message_id, 103); // Retry notice is 102.
    assert_eq!(
        sent_texts(&api)
            .iter()
            .filter(|text| text.starts_with("Рекомендация 1/"))
            .count(),
        1
    );
    let edits: Vec<_> = api
        .decoded()
        .into_iter()
        .filter(|(path, _)| path.ends_with("/EditMessageReplyMarkup"))
        .collect();
    assert_eq!(edits.len(), 3);
    assert_eq!(edits[0].1["message_id"], "101");
    assert_eq!(edits[1].1["message_id"], "101");
    assert_ne!(edits[0].1["reply_markup"], edits[1].1["reply_markup"]);
}

#[tokio::test]
async fn later_delivery_failure_preserves_earlier_controls_and_scores_wait_for_selection() {
    let (_dir, bundle) = bundle(2, false);
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone());
    let api = FakeTelegram::new();
    let sel = selection(&bundle, 1, 1);
    let mut session = session(&sel);
    *repo.fail_send_after_record.lock().unwrap() = Some(api.fail_send.clone());
    assert!(
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .is_err()
    );
    let row = repo.rows.lock().unwrap()[0].clone();
    let first_token = score_token(&session, row.id, 5);
    let edit_count = api
        .decoded()
        .iter()
        .filter(|(path, _)| path.ends_with("/EditMessageReplyMarkup"))
        .count();
    session.state = State::ChoosingAnime {
        intent: AnimeIntent::Recommend,
        query: sel.query.clone(),
        candidates: vec![sel.seed_mal_id],
        selected_mal_id: Some(sel.seed_mal_id),
    };
    assert!(session.claim(&first_token, row.message_id).is_none());
    session.state = State::Selected {
        intent: AnimeIntent::Recommend,
        selection: sel.clone(),
    };
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
        .await
        .unwrap();
    assert_eq!(score_token(&session, row.id, 5), first_token);
    assert_eq!(
        api.decoded()
            .iter()
            .filter(|(path, _)| path.ends_with("/EditMessageReplyMarkup"))
            .count(),
        edit_count + 1
    );
    assert!(session.claim(&first_token, row.message_id).is_some());
}

#[tokio::test]
async fn all_scores_and_independent_positions_are_immutable() {
    for score in 0..=5 {
        let (_dir, bundle) = bundle(2, false);
        let repo = Arc::new(FakeRepo::default());
        let ctx = context(bundle.clone(), repo.clone());
        let api = FakeTelegram::new();
        let sel = selection(&bundle, 1, 1);
        let mut session = session(&sel);
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .unwrap();
        let rows = repo.rows.lock().unwrap().clone();
        let token = score_token(&session, rows[0].id, score);
        let other = score_token(&session, rows[1].id, 5);
        let action_key = session.claim(&token, rows[0].message_id).unwrap().1;
        recommendations::on_score(
            &api.bot(),
            &ctx,
            ACTOR,
            &mut session,
            &sel,
            recommendations::ScoreAction {
                position: &rows[0],
                score,
                action_key: &action_key,
            },
        )
        .await
        .unwrap();
        session.finish(&token, true);
        assert_eq!(
            *repo.scores.lock().unwrap(),
            vec![(ACTOR.user_id, rows[0].id, score)]
        );
        assert!(session
            .claim(
                &score_token(&session, rows[0].id, score),
                rows[0].message_id
            )
            .is_none());
        assert!(session.claim(&other, rows[1].message_id).is_some());
        let second_key = session.callbacks[&other].action_key.clone();
        recommendations::on_score(
            &api.bot(),
            &ctx,
            ACTOR,
            &mut session,
            &sel,
            recommendations::ScoreAction {
                position: &rows[1],
                score: 5,
                action_key: &second_key,
            },
        )
        .await
        .unwrap();
        session.finish(&other, true);
        assert_eq!(repo.scores.lock().unwrap().len(), 2);
    }
}

#[tokio::test]
async fn database_and_notice_failures_pin_first_score_for_same_button_retry() {
    for failure in [1, 2, 3] {
        let (_dir, bundle) = bundle(2, false);
        let repo = Arc::new(FakeRepo::default());
        let ctx = context(bundle.clone(), repo.clone());
        let api = FakeTelegram::new();
        let sel = selection(&bundle, 1, 1);
        let mut session = session(&sel);
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .unwrap();
        let rows = repo.rows.lock().unwrap().clone();
        let token = score_token(&session, rows[0].id, 4);
        let other = score_token(&session, rows[1].id, 3);
        let key = session.claim(&token, rows[0].message_id).unwrap().1;
        repo.score_failure.store(failure, Ordering::SeqCst);
        let result = recommendations::on_score(
            &api.bot(),
            &ctx,
            ACTOR,
            &mut session,
            &sel,
            recommendations::ScoreAction {
                position: &rows[0],
                score: 4,
                action_key: &key,
            },
        )
        .await;
        assert_eq!(result.is_err(), failure != 3);
        session.finish(&token, result.is_ok());
        assert!(session.callbacks.values().all(|record| {
            !matches!(record.action, Action::RecommendationScore { position_id, score }
                if position_id == rows[0].id && score != 4)
        }));
        assert!(session.callbacks.contains_key(&other));
        if failure == 3 {
            assert!(session.claim(&token, rows[0].message_id).is_none());
        } else {
            assert!(session.claim(&token, rows[0].message_id).is_some());
            let result = recommendations::on_score(
                &api.bot(),
                &ctx,
                ACTOR,
                &mut session,
                &sel,
                recommendations::ScoreAction {
                    position: &rows[0],
                    score: 4,
                    action_key: &key,
                },
            )
            .await;
            assert!(result.is_ok());
            session.finish(&token, true);
            assert_eq!(
                *repo.scores.lock().unwrap(),
                vec![(ACTOR.user_id, rows[0].id, 4)]
            );
        }
    }
    for terminal in [false, true] {
        let (_dir, bundle) = bundle(2, false);
        let repo = Arc::new(FakeRepo::default());
        let ctx = context(bundle.clone(), repo.clone());
        let api = FakeTelegram::new();
        let sel = selection(&bundle, 1, 1);
        let mut session = session(&sel);
        recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
            .await
            .unwrap();
        let row = repo.rows.lock().unwrap()[0].clone();
        let token = score_token(&session, row.id, 2);
        let key = session.claim(&token, row.message_id).unwrap().1;
        if terminal {
            repo.score_failure.store(3, Ordering::SeqCst);
        }
        api.fail_send.store(true, Ordering::SeqCst);
        assert!(recommendations::on_score(
            &api.bot(),
            &ctx,
            ACTOR,
            &mut session,
            &sel,
            recommendations::ScoreAction {
                position: &row,
                score: 2,
                action_key: &key
            }
        )
        .await
        .is_err());
        session.finish(&token, false);
        assert!(session.claim(&token, row.message_id).is_some());
    }
}

#[tokio::test]
async fn synopsis_pages_are_lossless_and_keep_score_tokens() {
    let original = format!("Первое. Второе! Третье? {}", "🌸word\n".repeat(1100));
    let (_dir, bundle) = bundle_with(1, false, |catalog| {
        catalog["anime"]["10"]["synopsis"] = json!(original.clone());
    });
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone());
    let api = FakeTelegram::new();
    let sel = selection(&bundle, 1, 1);
    let mut session = session(&sel);
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
        .await
        .unwrap();
    let row = repo.rows.lock().unwrap()[0].clone();
    let preview = &sent_texts(&api)[0];
    assert!(preview.ends_with("Первое. Второе! Третье?"));
    assert!(!preview.contains("🌸word"));
    let score = score_token(&session, row.id, 3);
    let mut recovered = String::new();
    for page in 0..10 {
        let result = recommendations::on_description(
            &api.bot(),
            &ctx,
            ACTOR,
            &mut session,
            &sel,
            &row,
            DescriptionView::Page(page),
        )
        .await;
        if result.is_err() {
            break;
        }
        let edits: Vec<_> = api
            .decoded()
            .into_iter()
            .filter(|(path, _)| path.ends_with("/EditMessageText"))
            .collect();
        let fields = &edits.last().unwrap().1;
        assert_eq!(fields["message_id"], row.message_id.to_string());
        assert!(fields["text"].encode_utf16().count() <= 4000);
        recovered.push_str(fields["text"].split_once("):\n").unwrap().1);
        let markup: Value = serde_json::from_str(&fields["reply_markup"]).unwrap();
        assert_eq!(markup["inline_keyboard"][0][3]["callback_data"], score);
    }
    assert_eq!(recovered, original);
    recommendations::on_description(
        &api.bot(),
        &ctx,
        ACTOR,
        &mut session,
        &sel,
        &row,
        DescriptionView::Summary,
    )
    .await
    .unwrap();
    let last = api
        .decoded()
        .into_iter()
        .rfind(|(path, _)| path.ends_with("/EditMessageText"))
        .unwrap()
        .1;
    assert_eq!(last["text"], *preview);
    assert!(repo.scores.lock().unwrap().is_empty());
    let key = session.claim(&score, row.message_id).unwrap().1;
    recommendations::on_score(
        &api.bot(),
        &ctx,
        ACTOR,
        &mut session,
        &sel,
        recommendations::ScoreAction {
            position: &row,
            score: 3,
            action_key: &key,
        },
    )
    .await
    .unwrap();
    session.finish(&score, true);
    recommendations::on_description(
        &api.bot(),
        &ctx,
        ACTOR,
        &mut session,
        &sel,
        &row,
        DescriptionView::Page(0),
    )
    .await
    .unwrap();
    let last = api
        .decoded()
        .into_iter()
        .rfind(|(path, _)| path.ends_with("/EditMessageText"))
        .unwrap()
        .1;
    let markup: Value = serde_json::from_str(&last["reply_markup"]).unwrap();
    assert_eq!(markup["inline_keyboard"][0].as_array().unwrap().len(), 1);
    assert_eq!(markup["inline_keyboard"][0][0]["callback_data"], score);
}

#[tokio::test]
async fn description_callbacks_are_reusable_owned_and_retry_failed_edits() {
    use bot::dialogue::storage::CallbackStatus;
    let (_dir, bundle) = bundle_with(1, false, |catalog| {
        catalog["anime"]["10"]["synopsis"] = json!("One. Two. Three. Four.");
    });
    let repo = Arc::new(FakeRepo::default());
    let ctx = Arc::new(context(bundle.clone(), repo.clone()));
    let api = FakeTelegram::new();
    let bot = api.bot();
    let sel = selection(&bundle, 1, 1);
    let session = ctx.sessions.get(ACTOR);
    {
        let mut guard = session.lock().await;
        guard.state = State::Selected {
            intent: AnimeIntent::Recommend,
            selection: sel.clone(),
        };
        recommendations::begin(&bot, &ctx, ACTOR, &mut guard, &sel)
            .await
            .unwrap();
    }
    let row = repo.rows.lock().unwrap()[0].clone();
    let token = {
        let guard = session.lock().await;
        guard.callbacks.iter().find(|(_, record)| matches!(record.action,
            Action::RecommendationDescription { position_id, view: DescriptionView::Page(0) } if position_id == row.id))
            .unwrap().0.clone()
    };
    dispatch(
        &bot,
        &ctx,
        callback(1000, ACTOR.chat_id, 999, row.message_id, Some(&token)),
    )
    .await;
    dispatch(
        &bot,
        &ctx,
        callback(
            1001,
            ACTOR.chat_id + 1,
            ACTOR.user_id as u64,
            row.message_id,
            Some(&token),
        ),
    )
    .await;
    dispatch(
        &bot,
        &ctx,
        callback(
            1002,
            ACTOR.chat_id,
            ACTOR.user_id as u64,
            row.message_id + 1,
            Some(&token),
        ),
    )
    .await;
    assert_eq!(
        api.decoded()
            .iter()
            .filter(|(path, _)| path.ends_with("/EditMessageText"))
            .count(),
        0
    );
    api.fail_edit_text.store(true, Ordering::SeqCst);
    let failed = bot::handlers::schema()
        .dispatch(teloxide::dptree::deps![
            bot.clone(),
            callback(
                1003,
                ACTOR.chat_id,
                ACTOR.user_id as u64,
                row.message_id,
                Some(&token)
            ),
            ctx.clone()
        ])
        .await;
    assert!(matches!(failed, std::ops::ControlFlow::Break(Err(_))));
    assert_eq!(
        session.lock().await.callbacks[&token].status,
        CallbackStatus::Active
    );
    dispatch(
        &bot,
        &ctx,
        callback(
            1004,
            ACTOR.chat_id,
            ACTOR.user_id as u64,
            row.message_id,
            Some(&token),
        ),
    )
    .await;
    api.edit_not_modified.store(true, Ordering::SeqCst);
    dispatch(
        &bot,
        &ctx,
        callback(
            1005,
            ACTOR.chat_id,
            ACTOR.user_id as u64,
            row.message_id,
            Some(&token),
        ),
    )
    .await;
    assert_eq!(
        session.lock().await.callbacks[&token].status,
        CallbackStatus::Active
    );
    assert_eq!(
        api.decoded()
            .iter()
            .filter(|(path, _)| path.ends_with("/EditMessageText"))
            .count(),
        3
    );
    assert!(repo.scores.lock().unwrap().is_empty());
    {
        let mut guard = session.lock().await;
        guard.reset();
    }
    dispatch(
        &bot,
        &ctx,
        callback(
            1006,
            ACTOR.chat_id,
            ACTOR.user_id as u64,
            row.message_id,
            Some(&token),
        ),
    )
    .await;
    assert_eq!(
        api.decoded()
            .iter()
            .filter(|(path, _)| path.ends_with("/EditMessageText"))
            .count(),
        3
    );
}

#[tokio::test]
async fn covers_reply_to_durable_text_and_failed_photo_does_not_repeat_on_retry() {
    let (_dir, bundle) = bundle(2, false);
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone()).with_covers(Arc::new(FixedCovers));
    let api = FakeTelegram::new();
    let sel = selection(&bundle, 1, 1);
    let mut session = session(&sel);
    api.fail_photo.store(true, Ordering::SeqCst);
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
        .await
        .unwrap();
    let first = api.decoded();
    let photos: Vec<_> = first
        .iter()
        .filter(|(path, _)| path.ends_with("/SendPhoto"))
        .collect();
    assert_eq!(photos.len(), 1);
    let rows = repo.rows.lock().unwrap().clone();
    assert_eq!(rows.len(), 2);
    assert!(
        photos[0]
            .1
            .get("reply_parameters")
            .is_some_and(|value| value.contains(&rows[0].message_id.to_string())),
        "{:?}",
        photos[0].1
    );
    assert_eq!(
        first
            .iter()
            .filter(|(path, _)| path.ends_with("/SendMessage"))
            .count(),
        2
    );
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
        .await
        .unwrap();
    assert_eq!(
        api.decoded()
            .iter()
            .filter(|(path, _)| path.ends_with("/SendPhoto"))
            .count(),
        1
    );
}

#[tokio::test]
async fn five_covers_follow_the_five_durable_cards_in_order() {
    let (_dir, bundle) = bundle(5, false);
    let repo = Arc::new(FakeRepo::default());
    let ctx = context(bundle.clone(), repo.clone()).with_covers(Arc::new(FixedCovers));
    let api = FakeTelegram::new();
    let sel = selection(&bundle, 1, 1);
    let mut session = session(&sel);
    recommendations::begin(&api.bot(), &ctx, ACTOR, &mut session, &sel)
        .await
        .unwrap();
    let rows = repo.rows.lock().unwrap().clone();
    let requests = api.decoded();
    let photos: Vec<_> = requests
        .iter()
        .filter(|(path, _)| path.ends_with("/SendPhoto"))
        .collect();
    assert_eq!(photos.len(), 5);
    for (photo, row) in photos.iter().zip(rows) {
        assert!(photo.1["reply_parameters"].contains(&row.message_id.to_string()));
        assert!(photo.1["photo"].contains(&format!("/{}.jpg", row.mal_id)));
        assert_eq!(photo.1["disable_notification"], "true");
    }
    assert_eq!(
        requests
            .iter()
            .filter(|(path, _)| path.ends_with("/SendMessage"))
            .count(),
        5
    );
}

#[tokio::test]
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn arb016_postgres_public_router_persists_scores_with_provenance() {
    use std::str::FromStr;
    use tokio_postgres::{Config, NoTls};
    let url = std::env::var("ARB_TEST_DATABASE_URL").expect("set ARB_TEST_DATABASE_URL");
    let mut config = Config::from_str(&url).unwrap();
    let name = config.get_dbname().unwrap();
    assert!(
        name == "arb_016_test" || name == "arb_ci_test",
        "unsafe ARB016 test database name"
    );
    let (admin, connection) = config.connect(NoTls).await.unwrap();
    tokio::spawn(async move {
        connection.await.unwrap();
    });
    let schema = format!("arb016_{}", std::process::id());
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
    let (select, candidate_message) = {
        let session = ctx.sessions.get(ACTOR);
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
        callback(3, 73, 42, candidate_message, Some(&select)),
    )
    .await;
    let (positions, scores) = {
        let rows = db.list_delivered_positions(42, 1).await.unwrap();
        let session = ctx.sessions.get(ACTOR);
        let guard = session.lock().await;
        let scores = rows
            .iter()
            .zip([0, 5])
            .map(|(row, score)| {
                (
                    score_token(&guard, row.id, score),
                    row.message_id,
                    row.id,
                    score,
                )
            })
            .collect::<Vec<_>>();
        (rows, scores)
    };
    assert_eq!(positions.len(), 2);
    dispatch(
        &bot,
        &ctx,
        callback(4, 73, 42, scores[0].1, Some(&scores[0].0)),
    )
    .await;
    dispatch(
        &bot,
        &ctx,
        callback(5, 73, 42, scores[1].1, Some(&scores[1].0)),
    )
    .await;
    dispatch(
        &bot,
        &ctx,
        callback(6, 73, 42, scores[0].1, Some(&scores[0].0)),
    )
    .await;
    dispatch(
        &bot,
        &ctx,
        callback(7, 74, 43, scores[0].1, Some(&scores[0].0)),
    )
    .await;
    let rows = direct.query(
        "SELECT d.rank,d.mal_id,d.chat_id,d.message_id,d.tg_id,r.raw_query,r.seed_mal_id,r.bundle_id,s.score,d.delivered_at,s.created_at FROM arb_recommendation_ratings s JOIN arb_delivered_positions d ON d.id=s.position_id JOIN arb_requests r ON r.id=d.request_id ORDER BY d.rank",
        &[]).await.unwrap();
    assert_eq!(rows.len(), 2);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.get::<_, i16>(0), (index + 1) as i16);
        assert_eq!(row.get::<_, i32>(1), [10, 11][index]);
        assert_eq!(row.get::<_, i64>(2), 73);
        assert_eq!(row.get::<_, i32>(3), positions[index].message_id);
        assert_eq!(row.get::<_, i64>(4), 42);
        assert_eq!(row.get::<_, String>(5), "Общее название");
        assert_eq!(row.get::<_, Option<i32>>(6), Some(1));
        assert_eq!(
            row.get::<_, Option<String>>(7).as_deref(),
            Some(bundle.identity())
        );
        assert_eq!(row.get::<_, i16>(8), [0, 5][index]);
        assert!(row.get::<_, std::time::SystemTime>(9) <= std::time::SystemTime::now());
        assert!(row.get::<_, std::time::SystemTime>(10) <= std::time::SystemTime::now());
    }
    assert_eq!(
        db.rate_recommendation(42, scores[0].2, 0).await.unwrap(),
        WriteOutcome::AlreadyRecorded(())
    );
    assert_eq!(
        db.rate_recommendation(42, scores[0].2, 5).await,
        Err(DbError::Conflict)
    );
    let restarted = Arc::new(context(bundle, db.clone()));
    dispatch(
        &bot,
        &restarted,
        callback(8, 73, 42, scores[0].1, Some(&scores[0].0)),
    )
    .await;
    assert_eq!(
        direct
            .query_one("SELECT count(*) FROM arb_recommendation_ratings", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        2
    );
    drop(restarted);
    drop(ctx);
    drop(db);
    drop(direct);
    admin
        .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
        .await
        .unwrap();
}
