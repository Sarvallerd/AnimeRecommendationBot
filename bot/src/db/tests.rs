use super::*;
use std::{
    env,
    str::FromStr,
    sync::atomic::{AtomicU64, Ordering},
};
use tokio_postgres::NoTls;

static NEXT_SCHEMA: AtomicU64 = AtomicU64::new(0);
const BUNDLE_A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const BUNDLE_B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

#[test]
fn validates_public_inputs_without_database() {
    assert!(legacy_id("2147483648").is_ok());
    assert_eq!(
        legacy_id("0"),
        Err(DbError::InvalidInput("ID must be positive"))
    );
    assert!(bundle(BUNDLE_A).is_ok());
    assert!(bundle(&format!("sha256:{}", "A".repeat(64))).is_err());
}

struct Fixture {
    db: Db,
    direct: Client,
    admin: Client,
    config: Config,
    schema: String,
}
async fn connect(config: &Config) -> Client {
    let (client, connection) = config
        .connect(NoTls)
        .await
        .expect("test database connection");
    tokio::spawn(async move {
        connection.await.expect("test connection task");
    });
    client
}
fn test_config() -> Config {
    let value = env::var("ARB_TEST_DATABASE_URL").expect("set ARB_TEST_DATABASE_URL explicitly");
    let config = Config::from_str(&value).expect("valid test PostgreSQL URL");
    let name = config.get_dbname().expect("test database name required");
    let numbered = name
        .strip_prefix("arb_")
        .and_then(|s| s.strip_suffix("_test"))
        .is_some_and(|digits| digits.len() == 3 && digits.bytes().all(|b| b.is_ascii_digit()));
    assert!(
        numbered || name == "arb_ci_test",
        "unsafe test database name"
    );
    config
}
async fn fixture(migrate: bool) -> Fixture {
    let base = test_config();
    let admin = connect(&base).await;
    let suffix = NEXT_SCHEMA.fetch_add(1, Ordering::Relaxed);
    let schema = format!("arbtest_{}_{}", std::process::id(), suffix);
    admin
        .batch_execute(&format!("CREATE SCHEMA {schema}"))
        .await
        .expect("create isolated schema");
    let mut config = base;
    config.options(format!("-c search_path={schema}"));
    let direct = connect(&config).await;
    let db = Db::new(&config).await.expect("db");
    if migrate {
        db.migrate().await.expect("migrate");
    }
    Fixture {
        db,
        direct,
        admin,
        config,
        schema,
    }
}
async fn cleanup(f: Fixture) {
    f.admin
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", f.schema))
        .await
        .expect("drop isolated schema");
}
fn created<T>(outcome: WriteOutcome<T>) -> T {
    match outcome {
        WriteOutcome::Created(value) => value,
        _ => panic!("expected created"),
    }
}
fn profile(id: i64) -> UserProfile {
    UserProfile {
        tg_id: id,
        first_name: "Иван '🌸'".into(),
        language_code: Some("ru".into()),
        last_name: None,
        username: Some("anime_fan".into()),
    }
}
async fn user_and_request(f: &Fixture, id: i64, key: &str, bundle_id: &str) -> i64 {
    f.db.upsert_user(&profile(id)).await.unwrap();
    let request = created(f.db.record_query(id, key, "Наруто ' \n 🌸").await.unwrap());
    assert_eq!(
        f.db.resolve_request(id, request, 42, bundle_id)
            .await
            .unwrap(),
        WriteOutcome::Created(())
    );
    request
}

#[tokio::test]
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn migration_preserves_legacy_and_rejects_unknown_versions() {
    let f = fixture(false).await;
    f.direct
        .batch_execute(
            "CREATE TABLE test_users (tg_id SERIAL PRIMARY KEY, language_code VARCHAR,
         first_name VARCHAR, last_name VARCHAR, username VARCHAR);
         CREATE TABLE test_request (tg_id INTEGER, msg VARCHAR);
         CREATE TABLE test_feedback (tg_id INTEGER, msg VARCHAR);
         INSERT INTO test_users (tg_id,language_code,first_name,last_name,username)
         VALUES (7,'ru','Иван''🌸',NULL,'name');
         INSERT INTO test_request VALUES (7,'line 1
         line 2 '' 🌸');
         INSERT INTO test_feedback VALUES (7,'отзыв '' 🌸');",
        )
        .await
        .unwrap();
    f.db.migrate().await.unwrap();
    f.db.migrate().await.unwrap();
    let row = f
        .direct
        .query_one(
            "SELECT tg_id,language_code,first_name,last_name,username FROM test_users",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, i64>(0), 7);
    assert_eq!(row.get::<_, String>(2), "Иван'🌸");
    assert_eq!(row.get::<_, Option<String>>(3), None);
    assert_eq!(
        f.direct
            .query_one("SELECT msg FROM test_request", &[])
            .await
            .unwrap()
            .get::<_, String>(0),
        "line 1\n         line 2 ' 🌸"
    );
    assert_eq!(
        f.direct
            .query_one("SELECT msg FROM test_feedback", &[])
            .await
            .unwrap()
            .get::<_, String>(0),
        "отзыв ' 🌸"
    );
    for table in ["test_users", "test_request", "test_feedback"] {
        let row = f
            .direct
            .query_one(
                "SELECT a.atttypid='bigint'::regtype FROM pg_attribute a
             WHERE a.attrelid=$1::text::regclass AND a.attname='tg_id'",
                &[&table],
            )
            .await
            .unwrap();
        assert!(row.get::<_, bool>(0), "{table}");
    }
    assert_eq!(
        f.direct
            .query_one("SELECT count(*) FROM arb_schema_migrations", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        1
    );
    f.db.insert_user(
        "2147483648".into(),
        "ru".into(),
        "Большой".into(),
        "NULL".into(),
        "quote'".into(),
    )
    .await
    .unwrap();
    f.db.insert_msg("request", "2147483648".into(), "O'Brien 🌸\nnext".into())
        .await
        .unwrap();
    f.db.insert_msg("feedback", "2147483648".into(), "отзыв".into())
        .await
        .unwrap();
    assert_eq!(
        f.direct
            .query_one(
                "SELECT msg FROM test_request WHERE tg_id=$1",
                &[&2147483648_i64]
            )
            .await
            .unwrap()
            .get::<_, String>(0),
        "O'Brien 🌸\nnext"
    );
    assert_eq!(
        f.db.insert_msg("other", "7".into(), "x".into()).await,
        Err(DbError::InvalidInput("unknown message kind"))
    );
    f.direct
        .execute(
            "INSERT INTO arb_schema_migrations(version) VALUES (99)",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(f.db.migrate().await, Err(DbError::Conflict));
    cleanup(f).await;
}

#[tokio::test]
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn records_replays_and_provenance() {
    let f = fixture(true).await;
    let id = 2147483648_i64;
    f.db.upsert_user(&profile(id)).await.unwrap();
    f.db.upsert_user(&profile(id + 1)).await.unwrap();
    let query = "Наруто ' 🌸\nsecond line";
    let request = created(f.db.record_query(id, "message:1", query).await.unwrap());
    assert_eq!(
        f.db.record_query(id, "message:1", query).await.unwrap(),
        WriteOutcome::AlreadyRecorded(request)
    );
    assert_eq!(
        f.db.record_query(id, "message:1", "changed").await,
        Err(DbError::Conflict)
    );
    assert_eq!(
        f.direct
            .query_one(
                "SELECT raw_query FROM arb_requests WHERE id=$1",
                &[&request]
            )
            .await
            .unwrap()
            .get::<_, String>(0),
        query
    );
    assert!(f
        .direct
        .execute(
            "UPDATE arb_requests SET seed_mal_id=42, resolved_at=now() WHERE id=$1",
            &[&request],
        )
        .await
        .is_err());
    let input = DeliveryInput {
        rank: 1,
        mal_id: 100,
        chat_id: -5,
        message_id: 7,
    };
    assert_eq!(
        f.db.record_delivery(id, request, &input).await,
        Err(DbError::Conflict)
    );
    assert_eq!(
        f.db.record_delivery(id + 1, request, &input).await,
        Err(DbError::NotFound)
    );
    assert_eq!(
        f.db.record_delivery(id, 999999, &input).await,
        Err(DbError::NotFound)
    );
    assert_eq!(
        f.db.resolve_request(id, request, 42, BUNDLE_A)
            .await
            .unwrap(),
        WriteOutcome::Created(())
    );
    assert_eq!(
        f.db.resolve_request(id, request, 42, BUNDLE_A)
            .await
            .unwrap(),
        WriteOutcome::AlreadyRecorded(())
    );
    assert_eq!(
        f.db.resolve_request(id, request, 43, BUNDLE_A).await,
        Err(DbError::Conflict)
    );
    assert_eq!(
        f.db.record_delivery(
            id,
            request,
            &DeliveryInput {
                mal_id: 42,
                ..input.clone()
            }
        )
        .await,
        Err(DbError::Conflict)
    );
    let position = created(f.db.record_delivery(id, request, &input).await.unwrap());
    assert_eq!(
        f.db.record_delivery(id, request, &input).await.unwrap(),
        WriteOutcome::AlreadyRecorded(position)
    );
    assert_eq!(
        f.db.record_delivery(
            id,
            request,
            &DeliveryInput {
                message_id: 8,
                ..input.clone()
            }
        )
        .await,
        Err(DbError::Conflict)
    );
    assert_eq!(
        f.db.record_delivery(
            id,
            request,
            &DeliveryInput {
                rank: 2,
                ..input.clone()
            }
        )
        .await,
        Err(DbError::Conflict)
    );
    let second = DeliveryInput {
        rank: 2,
        mal_id: 101,
        ..input
    };
    f.db.record_delivery(id, request, &second).await.unwrap();
    assert_eq!(
        f.db.list_delivered_positions(id, request)
            .await
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        f.db.list_delivered_positions(id + 1, request).await,
        Err(DbError::NotFound)
    );
    assert_eq!(
        f.db.rate_recommendation(id + 1, position, 5).await,
        Err(DbError::NotFound)
    );
    assert_eq!(
        f.db.rate_recommendation(id, position, 0).await.unwrap(),
        WriteOutcome::Created(())
    );
    assert_eq!(
        f.db.rate_recommendation(id, position, 0).await.unwrap(),
        WriteOutcome::AlreadyRecorded(())
    );
    assert_eq!(
        f.db.rate_recommendation(id, position, 5).await,
        Err(DbError::Conflict)
    );
    let body = "Полный отзыв: ' 🌸\n".to_owned() + &"длинный текст ".repeat(1000);
    let feedback = created(f.db.save_feedback(id, "message:2", &body).await.unwrap());
    assert_eq!(
        f.db.save_feedback(id, "message:2", &body).await.unwrap(),
        WriteOutcome::AlreadyRecorded(feedback)
    );
    assert_eq!(
        f.db.save_feedback(id, "message:2", "other").await,
        Err(DbError::Conflict)
    );
    assert_eq!(
        f.direct
            .query_one("SELECT body FROM arb_feedback WHERE id=$1", &[&feedback])
            .await
            .unwrap()
            .get::<_, String>(0),
        body
    );

    let event_a = created(f.db.rate_anime(id, request, 4).await.unwrap());
    let original = f.direct.query_one(
        "SELECT created_at::text,updated_at::text FROM arb_anime_ratings WHERE tg_id=$1 AND mal_id=42",
        &[&id],
    ).await.unwrap();
    let request_b = created(
        f.db.record_query(id, "message:3", "другой поиск")
            .await
            .unwrap(),
    );
    f.db.resolve_request(id, request_b, 42, BUNDLE_B)
        .await
        .unwrap();
    let event_b = created(f.db.rate_anime(id, request_b, 9).await.unwrap());
    assert!(event_b > event_a);
    let newer = f
        .direct
        .query_one(
            "SELECT current_event_id,created_at::text,updated_at::text
         FROM arb_anime_ratings WHERE tg_id=$1 AND mal_id=42",
            &[&id],
        )
        .await
        .unwrap();
    assert_eq!(newer.get::<_, i64>(0), event_b);
    assert_eq!(newer.get::<_, String>(1), original.get::<_, String>(0));
    assert_eq!(
        f.db.rate_anime(id, request, 4).await.unwrap(),
        WriteOutcome::AlreadyRecorded(event_a)
    );
    assert_eq!(
        f.db.rate_anime(id, request, 5).await,
        Err(DbError::Conflict)
    );
    let after = f.direct.query_one(
        "SELECT current_event_id,updated_at::text FROM arb_anime_ratings WHERE tg_id=$1 AND mal_id=42",
        &[&id],
    ).await.unwrap();
    assert_eq!(after.get::<_, i64>(0), event_b);
    assert_eq!(after.get::<_, String>(1), newer.get::<_, String>(2));
    let request_c = created(
        f.db.record_query(id, "message:4", "same score")
            .await
            .unwrap(),
    );
    f.db.resolve_request(id, request_c, 42, BUNDLE_B)
        .await
        .unwrap();
    let event_c = created(f.db.rate_anime(id, request_c, 9).await.unwrap());
    assert!(event_c > event_b);
    let rows = f
        .direct
        .query(
            "SELECT e.id,e.score,r.raw_query,r.bundle_id,r.seed_mal_id,e.created_at::text,
                r.created_at::text,r.resolved_at::text
         FROM arb_anime_rating_events e JOIN arb_requests r ON r.id=e.request_id
         ORDER BY e.id",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].get::<_, String>(2), query);
    assert_eq!(rows[0].get::<_, String>(3), BUNDLE_A);
    assert_eq!(rows[0].get::<_, i32>(4), 42);
    assert!(!rows[0].get::<_, String>(5).is_empty());
    assert!(!rows[0].get::<_, String>(6).is_empty());
    assert!(!rows[0].get::<_, String>(7).is_empty());
    assert_eq!(rows[1].get::<_, String>(3), BUNDLE_B);
    assert_eq!(rows[2].get::<_, i16>(1), 9);
    let request_d = created(
        f.db.record_query(id, "message:5", "foreign key check")
            .await
            .unwrap(),
    );
    f.db.resolve_request(id, request_d, 42, BUNDLE_A)
        .await
        .unwrap();
    assert!(f.direct.execute(
        "INSERT INTO arb_anime_rating_events(request_id,tg_id,mal_id,score) VALUES ($1,$2,99,5)",
        &[&request_d, &id],
    ).await.is_err());
    f.db.migrate().await.unwrap();
    let current = f
        .direct
        .query_one(
            "SELECT current_event_id FROM arb_anime_ratings WHERE tg_id=$1 AND mal_id=42",
            &[&id],
        )
        .await
        .unwrap()
        .get::<_, i64>(0);
    assert_eq!(current, event_c);
    assert_eq!(
        f.direct
            .query_one("SELECT count(*) FROM arb_anime_rating_events", &[])
            .await
            .unwrap()
            .get::<_, i64>(0),
        3
    );
    cleanup(f).await;
}

#[tokio::test]
#[ignore = "requires explicit ARB_TEST_DATABASE_URL"]
async fn boundaries_concurrency_and_reconnect() {
    let f = fixture(true).await;
    let id = 123_i64;
    let request = user_and_request(&f, id, "a", BUNDLE_A).await;
    assert!(matches!(
        f.db.rate_anime(id, request, 0).await,
        Err(DbError::InvalidInput(_))
    ));
    assert!(matches!(
        f.db.rate_anime(id, request, 11).await,
        Err(DbError::InvalidInput(_))
    ));
    assert!(matches!(
        f.db.resolve_request(id, request, 0, BUNDLE_A).await,
        Err(DbError::InvalidInput(_))
    ));
    assert!(matches!(
        f.db.resolve_request(id, request, 42, "bad").await,
        Err(DbError::InvalidInput(_))
    ));
    for rank in [0, 6] {
        assert!(matches!(
            f.db.record_delivery(
                id,
                request,
                &DeliveryInput {
                    rank,
                    mal_id: 99,
                    chat_id: 1,
                    message_id: 1
                }
            )
            .await,
            Err(DbError::InvalidInput(_))
        ));
    }
    let position = created(
        f.db.record_delivery(
            id,
            request,
            &DeliveryInput {
                rank: 1,
                mal_id: i32::MAX,
                chat_id: 1,
                message_id: 1,
            },
        )
        .await
        .unwrap(),
    );
    for score in [-1, 6] {
        assert!(matches!(
            f.db.rate_recommendation(id, position, score).await,
            Err(DbError::InvalidInput(_))
        ));
    }
    let db2 = Db::new(&f.config).await.unwrap();
    let (a, b) = tokio::join!(
        f.db.rate_anime(id, request, 4),
        db2.rate_anime(id, request, 4)
    );
    let outcomes = [a.unwrap(), b.unwrap()];
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| matches!(o, WriteOutcome::Created(_)))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| matches!(o, WriteOutcome::AlreadyRecorded(_)))
            .count(),
        1
    );
    let (a, b) = tokio::join!(
        f.db.rate_recommendation(id, position, 5),
        db2.rate_recommendation(id, position, 5)
    );
    assert_eq!(
        [a.unwrap(), b.unwrap()]
            .iter()
            .filter(|o| matches!(o, WriteOutcome::Created(())))
            .count(),
        1
    );
    let other = user_and_request(&f, id, "b", BUNDLE_B).await;
    let db3 = Db::new(&f.config).await.unwrap();
    let (a, b) = tokio::join!(db2.rate_anime(id, other, 5), db3.rate_anime(id, other, 6));
    assert_eq!([a, b].iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        f.direct
            .query_one(
                "SELECT count(*) FROM arb_anime_rating_events WHERE request_id=$1",
                &[&other],
            )
            .await
            .unwrap()
            .get::<_, i64>(0),
        1
    );
    let third = user_and_request(&f, id, "c", BUNDLE_B).await;
    let fourth = user_and_request(&f, id, "d", BUNDLE_B).await;
    let (a, b) = tokio::join!(db2.rate_anime(id, third, 7), db3.rate_anime(id, fourth, 8));
    let a = created(a.unwrap());
    let b = created(b.unwrap());
    assert_eq!(
        f.direct
            .query_one(
                "SELECT current_event_id FROM arb_anime_ratings WHERE tg_id=$1 AND mal_id=42",
                &[&id],
            )
            .await
            .unwrap()
            .get::<_, i64>(0),
        a.max(b)
    );

    let backend = {
        let client = f.db.client().await.unwrap();
        client
            .query_one("SELECT pg_backend_pid()", &[])
            .await
            .unwrap()
            .get::<_, i32>(0)
    };
    f.admin
        .query_one("SELECT pg_terminate_backend($1)", &[&backend])
        .await
        .unwrap();
    for _ in 0..1000 {
        if f.db.client.lock().await.is_closed() {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(f.db.client.lock().await.is_closed());
    assert!(matches!(
        f.db.save_feedback(id, "after-reconnect", "ok").await,
        Ok(WriteOutcome::Created(_))
    ));
    cleanup(f).await;
}
