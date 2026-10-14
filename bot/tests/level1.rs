#[path = "support/level1.rs"]
mod level1_support;
mod support;

use bot::dialogue::state::{Actor, State};
use level1_support::{changed_bundle, fixture_bundle, DbFixture, Stand, CHAT, USER};
use serde_json::json;
use std::{sync::atomic::Ordering, time::Duration};
use support::message;
use teloxide::prelude::*;

fn actor_rows(table: &str) -> String {
    assert!([
        "arb_requests",
        "arb_delivered_positions",
        "arb_anime_rating_events",
        "arb_recommendation_ratings",
        "arb_feedback"
    ]
    .contains(&table));
    format!("SELECT count(*) FROM {table} WHERE tg_id={USER}")
}
async fn count(db: &DbFixture, table: &str) -> i64 {
    db.direct
        .query_one(&actor_rows(table), &[])
        .await
        .unwrap()
        .get(0)
}
async fn stale(stand: &mut Stand, button: &(i32, String), db: &DbFixture) {
    let before = db.business_snapshot().await;
    assert!(stand.press(button).await);
    assert_eq!(
        stand.last_text(),
        "Эта кнопка устарела. Начните заново: /start"
    );
    assert_eq!(db.business_snapshot().await, before);
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_recommendations_alias() {
    let db = DbFixture::new("recommend").await;
    let bundle = fixture_bundle();
    let mut stand = Stand::new(db.db.clone(), bundle.clone());
    assert!(stand.say("/recommend").await);
    assert!(stand.say("Shared Name").await);
    assert_eq!(count(&db, "arb_requests").await, 1);
    assert_eq!(count(&db, "arb_delivered_positions").await, 0);
    let pending = db
        .direct
        .query_one(
            "SELECT raw_query,seed_mal_id,bundle_id FROM arb_requests WHERE tg_id=$1",
            &[&(USER as i64)],
        )
        .await
        .unwrap();
    assert_eq!(pending.get::<_, String>(0), "Shared Name");
    assert_eq!(pending.get::<_, Option<i32>>(1), None);
    assert_eq!(pending.get::<_, Option<String>>(2), None);
    let selected = stand.button("MAL ID 1");
    assert!(stand.press(&selected).await);
    let rows = db.direct.query("SELECT d.rank,d.mal_id,d.chat_id,d.message_id,r.raw_query,r.seed_mal_id,r.bundle_id FROM arb_delivered_positions d JOIN arb_requests r ON r.id=d.request_id WHERE d.tg_id=$1 ORDER BY d.rank", &[&(USER as i64)]).await.unwrap();
    assert_eq!(rows.len(), 5);
    let cards = stand.successful_sends();
    for (index, row) in rows.iter().enumerate() {
        let mal = [10, 11, 12, 2, 3][index];
        assert_eq!(row.get::<_, i16>(0), (index + 1) as i16);
        assert_eq!(row.get::<_, i32>(1), mal);
        assert_eq!(row.get::<_, i64>(2), CHAT);
        let id = row.get::<_, i32>(3);
        let rendered = cards
            .iter()
            .find(|exchange| exchange.response["result"]["message_id"] == id)
            .unwrap();
        assert!(rendered
            .fields
            .get("text")
            .or(rendered.fields.get("caption"))
            .unwrap()
            .contains(&format!("MAL ID: {mal}")));
        assert_eq!(row.get::<_, String>(4), "Shared Name");
        assert_eq!(row.get::<_, Option<i32>>(5), Some(1));
        assert_eq!(row.get::<_, String>(6), bundle.identity());
    }
    stale(&mut stand, &selected, &db).await;
    drop(stand);
    drop(bundle);
    db.close().await;
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_empty() {
    let db = DbFixture::new("empty").await;
    let mut stand = Stand::new(db.db.clone(), fixture_bundle());
    stand.recommend("Тихий ноль", 99).await;
    assert_eq!(
        stand.last_text(),
        "Для этого аниме пока нет рекомендаций. Попробуйте другое: /recommend."
    );
    let row = db
        .direct
        .query_one(
            "SELECT raw_query,seed_mal_id,bundle_id FROM arb_requests WHERE tg_id=$1",
            &[&(USER as i64)],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, String>(0), "Тихий ноль");
    assert_eq!(row.get::<_, Option<i32>>(1), Some(99));
    assert_eq!(row.get::<_, String>(2), stand.ctx.bundle.identity());
    assert_eq!(count(&db, "arb_delivered_positions").await, 0);
    drop(stand);
    db.close().await;
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_recommendation_scores() {
    let db = DbFixture::new("recscores").await;
    let mut stand = Stand::new(db.db.clone(), fixture_bundle());
    let mut rated = Vec::new();
    for score in 0..=5_i16 {
        stand.recommend("Общее название", 1).await;
        let row = db.direct.query_one("SELECT id,message_id FROM arb_delivered_positions WHERE tg_id=$1 ORDER BY id DESC LIMIT 1 OFFSET 4", &[&(USER as i64)]).await.unwrap();
        let position: i64 = row.get(0);
        let message_id: i32 = row.get(1);
        let labels: Vec<_> = stand
            .buttons()
            .into_iter()
            .filter(|(_, id, _)| *id == message_id)
            .map(|(label, _, _)| label)
            .collect();
        for option in 0..=5 {
            assert!(labels.contains(&option.to_string()));
        }
        let selected = stand.button_on(message_id, &score.to_string());
        assert!(stand.press(&selected).await);
        rated.push((position, score, selected));
    }
    let rows = db.direct.query("SELECT position_id,score FROM arb_recommendation_ratings WHERE tg_id=$1 ORDER BY position_id", &[&(USER as i64)]).await.unwrap();
    assert_eq!(rows.len(), 6);
    for (row, (position, score, _)) in rows.iter().zip(&rated) {
        assert_eq!(row.get::<_, i64>(0), *position);
        assert_eq!(row.get::<_, i16>(1), *score);
    }
    stale(&mut stand, &rated[5].2, &db).await;
    drop(stand);
    db.close().await;
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_anime_scores() {
    let db = DbFixture::new("animescores").await;
    let mut stand = Stand::new(db.db.clone(), fixture_bundle());
    let mut last = None;
    for score in 1..=10_i16 {
        stand.rate("Общее название", 1).await;
        let keyboard = stand
            .successful_sends()
            .into_iter()
            .rev()
            .find(|exchange| {
                exchange
                    .fields
                    .get("text")
                    .is_some_and(|text| text.contains("от 1 до 10"))
            })
            .unwrap();
        let id = keyboard.response["result"]["message_id"].as_i64().unwrap() as i32;
        let labels: Vec<_> = stand
            .buttons()
            .into_iter()
            .filter(|(_, mid, _)| *mid == id)
            .map(|(label, _, _)| label)
            .collect();
        for option in 1..=10 {
            assert!(labels.contains(&option.to_string()));
        }
        let button = stand.button_on(id, &score.to_string());
        assert!(stand.press(&button).await);
        last = Some(button);
    }
    let rows = db.direct.query("SELECT e.score,e.id,r.current_event_id FROM arb_anime_rating_events e JOIN arb_anime_ratings r ON r.tg_id=e.tg_id AND r.mal_id=e.mal_id WHERE e.tg_id=$1 ORDER BY e.id", &[&(USER as i64)]).await.unwrap();
    assert_eq!(rows.len(), 10);
    let current: i64 = rows[9].get(2);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.get::<_, i16>(0), (index + 1) as i16);
        assert_eq!(row.get::<_, i64>(2), current);
    }
    assert_eq!(rows[9].get::<_, i64>(1), current);
    stale(&mut stand, &last.unwrap(), &db).await;
    drop(stand);
    db.close().await;
}

async fn state(stand: &Stand) -> State {
    let session = stand.ctx.sessions.get(Actor {
        chat_id: CHAT,
        user_id: USER as i64,
    });
    let current = session.lock().await.state.clone();
    current
}
async fn setup_stage(stand: &mut Stand, stage: &str) -> Option<(i32, String)> {
    match stage {
        "query" => {
            assert!(stand.say("/recommend").await);
            None
        }
        "choice" => {
            assert!(stand.say("/recommend").await);
            assert!(stand.say("Shared Name").await);
            Some(stand.button("MAL ID 1"))
        }
        "grade" => {
            stand.rate("Shared Name", 1).await;
            Some(stand.button("10"))
        }
        "feedback" => {
            assert!(stand.say("/feedback").await);
            None
        }
        _ => panic!("unknown stage"),
    }
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_navigation_all_stages() {
    let db = DbFixture::new("navigation").await;
    for stage in ["query", "choice", "grade", "feedback"] {
        for reset in ["/cancel", "/start"] {
            let mut stand = Stand::new(db.db.clone(), fixture_bundle());
            let old = setup_stage(&mut stand, stage).await;
            let before = db.business_snapshot().await;
            let prior = state(&stand).await;
            assert!(stand.say("/help").await);
            assert!(stand.last_text().contains("Команды: /start"));
            assert_eq!(state(&stand).await, prior);
            assert_eq!(db.business_snapshot().await, before);
            assert!(stand.say(reset).await);
            if reset == "/cancel" {
                assert!(stand.last_text().contains("Действие отменено"));
            } else {
                assert!(stand.last_text().contains("Привет!"));
                assert!(stand
                    .buttons()
                    .iter()
                    .any(|(label, _, _)| label == "Рекомендации"));
            }
            assert_eq!(state(&stand).await, State::Idle);
            assert_eq!(db.business_snapshot().await, before);
            if let Some(button) = old {
                stale(&mut stand, &button, &db).await;
            }
        }
    }
    db.close().await;
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_description_pages() {
    let synopsis = "🌸word\n".repeat(1100);
    let (_directory, bundle) = changed_bundle(|catalog| {
        catalog["anime"]["10"]["synopsis"] = json!(synopsis.clone());
    });
    let db = DbFixture::new("description").await;
    let mut stand = Stand::new(db.db.clone(), bundle);
    stand.recommend("Shared Name", 1).await;
    let row = db
        .direct
        .query_one(
            "SELECT message_id FROM arb_delivered_positions WHERE tg_id=$1 AND rank=1",
            &[&(USER as i64)],
        )
        .await
        .unwrap();
    let id: i32 = row.get(0);
    let original = stand
        .successful_sends()
        .into_iter()
        .find(|exchange| exchange.response["result"]["message_id"] == id)
        .unwrap()
        .fields["text"]
        .clone();
    let before = db.business_snapshot().await;
    let score = stand.button_on(id, "3");
    let expand = stand.button_on(id, "Развернуть описание");
    assert!(stand.press(&expand).await);
    assert!(stand.last_text().contains("🌸word"));
    let first = stand.last_text();
    let mut reconstructed = String::new();
    let mut pages = 0;
    loop {
        pages += 1;
        assert!(pages < 12, "description pagination did not terminate");
        let page = stand.last_text();
        assert!(page.encode_utf16().count() <= 4000);
        reconstructed.push_str(page.split_once("):\n").unwrap().1);
        assert_eq!(stand.button_on(id, "3"), score);
        if !stand
            .current_buttons_on(id)
            .iter()
            .any(|(label, _, _)| label == "Далее ▶")
        {
            break;
        }
        let forward = stand.button_on(id, "Далее ▶");
        assert!(stand.press(&forward).await);
    }
    assert!(pages >= 2);
    assert_eq!(reconstructed, synopsis);
    while stand
        .current_buttons_on(id)
        .iter()
        .any(|(label, _, _)| label == "◀ Назад")
    {
        let back = stand.button_on(id, "◀ Назад");
        assert!(stand.press(&back).await);
    }
    assert_eq!(stand.last_text(), first);
    let collapse = stand.button_on(id, "Свернуть описание");
    assert!(stand.press(&collapse).await);
    assert_eq!(stand.last_text(), original);
    assert_eq!(db.business_snapshot().await, before);
    assert_eq!(count(&db, "arb_recommendation_ratings").await, 0);
    drop(stand);
    db.close().await;
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_feedback_unicode() {
    let db = DbFixture::new("unicode").await;
    let mut stand = Stand::new(db.db.clone(), fixture_bundle());
    let body = format!(
        " Ёж, e\u{301}, 漢字, 🌸, 'quote', \\backslash; %_ <>&\n{}\nКонец.",
        "длинная строка с переносом. ".repeat(20)
    );
    assert!(body.len() > 255);
    assert!(stand.say("/feedback").await);
    assert!(stand.say(&body).await);
    assert_eq!(stand.last_text(), "Спасибо! Ваш отзыв сохранён.");
    let row = db
        .direct
        .query_one(
            "SELECT body,action_key FROM arb_feedback WHERE tg_id=$1",
            &[&(USER as i64)],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, String>(0), body);
    assert_eq!(row.get::<_, String>(1), format!("msg:{CHAT}:2"));
    let before = db.business_snapshot().await;
    assert!(stand.route(message(2, CHAT, USER, &body)).await);
    assert_eq!(db.business_snapshot().await, before);
    drop(stand);
    db.close().await;
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_context_reset_bundle_change() {
    let db = DbFixture::new("reset").await;
    let original = fixture_bundle();
    let mut stand = Stand::new(db.db.clone(), original.clone());
    assert!(stand.say("/recommend").await);
    assert!(stand.say("Shared Name").await);
    let old = stand.button("MAL ID 1");
    let before = db.business_snapshot().await;
    stand.reset_context(db.db.clone(), original);
    stale(&mut stand, &old, &db).await;
    assert_eq!(db.business_snapshot().await, before);
    let (_directory, changed) = changed_bundle(|catalog| {
        catalog["anime"]["10"]["title"] = json!("Новый северный ветер");
    });
    let changed_id = changed.identity().to_owned();
    stand.reset_context(db.db.clone(), changed);
    stale(&mut stand, &old, &db).await;
    stand.recommend("Shared Name", 1).await;
    let row = db.direct.query_one("SELECT bundle_id FROM arb_requests WHERE tg_id=$1 AND resolved_at IS NOT NULL ORDER BY id DESC LIMIT 1", &[&(USER as i64)]).await.unwrap();
    assert_eq!(row.get::<_, String>(0), changed_id);
    assert!(stand
        .successful_texts()
        .iter()
        .any(|text| text.contains("Новый северный ветер")));
    drop(stand);
    db.close().await;
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_transport_failure() {
    let db = DbFixture::new("transport").await;
    let mut stand = Stand::new(db.db.clone(), fixture_bundle());
    assert!(stand.say("/recommend").await);
    stand.api.fail_send.store(true, Ordering::SeqCst);
    assert!(!stand.say("Shared Name").await);
    assert_eq!(count(&db, "arb_requests").await, 1);
    assert_eq!(count(&db, "arb_delivered_positions").await, 0);
    assert!(stand.route(message(2, CHAT, USER, "Shared Name")).await);
    let choice = stand.button("MAL ID 1");
    assert!(stand.press(&choice).await);
    assert_eq!(count(&db, "arb_delivered_positions").await, 5);
    let row = db
        .direct
        .query_one(
            "SELECT message_id FROM arb_delivered_positions WHERE tg_id=$1 AND rank=1",
            &[&(USER as i64)],
        )
        .await
        .unwrap();
    let id: i32 = row.get(0);
    let score = stand.button_on(id, "5");
    stand.api.fail_ack.store(true, Ordering::SeqCst);
    assert!(stand.press(&score).await);
    assert_eq!(count(&db, "arb_recommendation_ratings").await, 0);
    assert!(stand.press(&score).await);
    assert_eq!(count(&db, "arb_recommendation_ratings").await, 1);
    drop(stand);
    db.close().await;
}

#[tokio::test]
#[ignore = "requires isolated ARB_TEST_DATABASE_URL"]
async fn level1_feedback_failure_navigation_retry() {
    let db = DbFixture::new("fbfailure").await;
    let mut stand = Stand::new(db.db.clone(), fixture_bundle());
    let body = "Первый отзыв 🌸\ne\u{301} 漢字. ".repeat(20);
    assert!(stand.say("/feedback").await);
    db.direct
        .batch_execute("BEGIN; LOCK TABLE arb_feedback IN ACCESS EXCLUSIVE MODE")
        .await
        .unwrap();
    let bot = stand.bot.clone();
    let ctx = stand.ctx.clone();
    let failed_message = message(2, CHAT, USER, &body);
    let running = tokio::spawn(async move {
        bot::handlers::schema()
            .dispatch(dptree::deps![bot, failed_message, ctx])
            .await
    });
    let app = format!("arb035_fbfailure_{}", std::process::id());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let backend = loop {
        let rows = db.admin.query("SELECT pid FROM pg_stat_activity WHERE datname=current_database() AND application_name=$1 AND wait_event_type='Lock' AND query LIKE '%arb_feedback%'", &[&app]).await.unwrap();
        if let Some(row) = rows.first() {
            break row.get::<_, i32>(0);
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "feedback insert did not reach the locked table"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(db
        .admin
        .query_one("SELECT pg_terminate_backend($1)", &[&backend])
        .await
        .unwrap()
        .get::<_, bool>(0));
    let result = tokio::time::timeout(Duration::from_secs(10), running)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(result, std::ops::ControlFlow::Break(Err(_))));
    db.direct.batch_execute("ROLLBACK").await.unwrap();
    assert_eq!(count(&db, "arb_feedback").await, 0);
    assert!(stand
        .successful_texts()
        .iter()
        .any(|text| text.contains("Не удалось подтвердить сохранение отзыва")));
    let obsolete = stand.button("Повторить отзыв");
    assert!(stand.say("/help").await);
    assert!(stand.say("/cancel").await);
    assert!(stand.say("/start").await);
    stale(&mut stand, &obsolete, &db).await;
    assert!(stand.say("/feedback").await);
    let retry = stand.button("Повторить отзыв");
    assert_ne!(retry.1, obsolete.1);
    assert!(stand.say("replacement body").await);
    assert!(stand
        .last_text()
        .contains("Сначала завершите предыдущий отзыв"));
    assert_eq!(count(&db, "arb_feedback").await, 0);
    assert!(stand.press(&retry).await);
    assert_eq!(stand.last_text(), "Спасибо! Ваш отзыв сохранён.");
    let row = db
        .direct
        .query_one(
            "SELECT action_key,body FROM arb_feedback WHERE tg_id=$1",
            &[&(USER as i64)],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, String>(0), format!("msg:{CHAT}:2"));
    assert_eq!(row.get::<_, String>(1), body);
    assert_eq!(count(&db, "arb_feedback").await, 1);
    stale(&mut stand, &retry, &db).await;
    drop(stand);
    db.close().await;
}
