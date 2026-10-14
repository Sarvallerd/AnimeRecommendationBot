use crate::support::{callback, message, Exchange, FakeTelegram};
use bot::{
    catalog::Bundle,
    db::Db,
    dialogue::{context::AppContext, storage::SessionStore},
    handlers,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    str::FromStr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use teloxide::{prelude::*, types::Update};
use tokio_postgres::{Client, Config, NoTls};

pub const CHAT: i64 = 7771;
pub const USER: u64 = 7771;

pub struct DbFixture {
    pub admin: Client,
    pub direct: Client,
    pub db: Arc<Db>,
    pub schema: String,
}
impl DbFixture {
    pub async fn new(label: &str) -> Self {
        static NEXT_SCHEMA: AtomicUsize = AtomicUsize::new(0);
        let url = std::env::var("ARB_TEST_DATABASE_URL").expect("explicit test DB URL");
        let mut config = Config::from_str(&url).expect("test database URL");
        assert_eq!(config.get_dbname(), Some("arb_ci_test"));
        assert_eq!(config.get_hosts().len(), 1);
        let (admin, connection) = config.connect(NoTls).await.unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let schema = format!(
            "arb035_{}_{}",
            std::process::id(),
            NEXT_SCHEMA.fetch_add(1, Ordering::SeqCst)
        );
        admin
            .batch_execute(&format!("CREATE SCHEMA {schema}"))
            .await
            .unwrap();
        config.options(format!(
            "-c search_path={schema} -c statement_timeout=10000"
        ));
        config.application_name(format!("arb035_{label}_{}", std::process::id()));
        let (seed, connection) = config.connect(NoTls).await.unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        seed.batch_execute(include_str!("../../../deploy/tests/legacy.sql"))
            .await
            .unwrap();
        let db = Arc::new(Db::new(&config).await.unwrap());
        db.migrate().await.unwrap();
        seed.batch_execute(include_str!("../../../deploy/tests/history.sql"))
            .await
            .unwrap();
        Self {
            admin,
            direct: seed,
            db,
            schema,
        }
    }
    pub async fn business_snapshot(&self) -> String {
        self.direct.query_one("SELECT jsonb_build_object(
          'legacy_users',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id) FROM test_users t),
          'legacy_requests',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id,msg) FROM test_request t),
          'legacy_feedback',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id,msg) FROM test_feedback t),
          'requests',(SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_requests t),
          'delivered',(SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_delivered_positions t),
          'anime_events',(SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_anime_rating_events t),
          'anime_current',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id,mal_id) FROM arb_anime_ratings t),
          'recommendation_ratings',(SELECT jsonb_agg(to_jsonb(t) ORDER BY position_id) FROM arb_recommendation_ratings t),
          'feedback',(SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_feedback t),
          'migrations',(SELECT jsonb_agg(to_jsonb(t) ORDER BY version) FROM arb_schema_migrations t)
          )::text", &[]).await.unwrap().get(0)
    }
    pub async fn close(self) {
        let schema = self.schema.clone();
        drop(self.db);
        drop(self.direct);
        self.admin
            .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
            .await
            .unwrap();
    }
}

pub fn fixture_bundle() -> Arc<Bundle> {
    Arc::new(
        Bundle::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/bundle"
        ))
        .unwrap(),
    )
}
pub fn changed_bundle(mut transform: impl FnMut(&mut Value)) -> (tempfile::TempDir, Arc<Bundle>) {
    let directory = tempfile::tempdir().unwrap();
    let base = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bundle"
    ));
    let mut catalog: Value =
        serde_json::from_slice(&std::fs::read(base.join("catalog.json")).unwrap()).unwrap();
    transform(&mut catalog);
    let catalog_bytes = format!("{catalog}\n").into_bytes();
    let neighbors_bytes = std::fs::read(base.join("neighbors.json")).unwrap();
    std::fs::write(directory.path().join("catalog.json"), &catalog_bytes).unwrap();
    std::fs::write(directory.path().join("neighbors.json"), &neighbors_bytes).unwrap();
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(base.join("manifest.json")).unwrap()).unwrap();
    manifest["files"]["catalog.json"]["sha256"] =
        json!(format!("{:x}", Sha256::digest(&catalog_bytes)));
    manifest["files"]["neighbors.json"]["sha256"] =
        json!(format!("{:x}", Sha256::digest(&neighbors_bytes)));
    std::fs::write(
        directory.path().join("manifest.json"),
        format!("{manifest}\n"),
    )
    .unwrap();
    let bundle = Arc::new(Bundle::load(directory.path()).unwrap());
    (directory, bundle)
}

pub struct Stand {
    pub api: FakeTelegram,
    pub bot: Bot,
    pub ctx: Arc<AppContext>,
    next_update: u32,
}
impl Stand {
    pub fn new(db: Arc<Db>, bundle: Arc<Bundle>) -> Self {
        let api = FakeTelegram::new();
        let bot = api.bot();
        let ctx = Arc::new(AppContext::new(bundle, db, Arc::new(SessionStore::new())));
        Self {
            api,
            bot,
            ctx,
            next_update: 1,
        }
    }
    pub fn reset_context(&mut self, db: Arc<Db>, bundle: Arc<Bundle>) {
        self.ctx = Arc::new(AppContext::new(bundle, db, Arc::new(SessionStore::new())));
    }
    pub async fn route(&mut self, update: Update) -> bool {
        matches!(
            tokio::time::timeout(
                Duration::from_secs(12),
                handlers::schema().dispatch(dptree::deps![
                    self.bot.clone(),
                    update,
                    self.ctx.clone()
                ])
            )
            .await
            .expect("router timed out"),
            std::ops::ControlFlow::Break(Ok(()))
        )
    }
    pub async fn say(&mut self, text: &str) -> bool {
        let id = self.next_update;
        self.next_update += 1;
        self.route(message(id, CHAT, USER, text)).await
    }
    pub async fn press(&mut self, button: &(i32, String)) -> bool {
        let id = self.next_update;
        self.next_update += 1;
        self.route(callback(id, CHAT, USER, button.0, Some(&button.1)))
            .await
    }
    pub fn buttons(&self) -> Vec<(String, i32, String)> {
        self.api
            .exchanges()
            .into_iter()
            .filter(|exchange| exchange.response["ok"] == true)
            .filter_map(|exchange| {
                let markup: Value =
                    serde_json::from_str(exchange.fields.get("reply_markup")?).ok()?;
                let id = exchange.response["result"]["message_id"].as_i64()? as i32;
                let rows = markup["inline_keyboard"].as_array()?;
                Some(
                    rows.iter()
                        .flat_map(|row| row.as_array().into_iter().flatten())
                        .filter_map(move |item| {
                            Some((
                                item["text"].as_str()?.to_owned(),
                                id,
                                item["callback_data"].as_str()?.to_owned(),
                            ))
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .flatten()
            .collect()
    }
    pub fn button(&self, label: &str) -> (i32, String) {
        let matching: Vec<_> = self
            .buttons()
            .into_iter()
            .filter(|(text, _, _)| text.contains(label))
            .collect();
        let (_, id, data) = matching
            .last()
            .unwrap_or_else(|| panic!("missing rendered button {label}"));
        (*id, data.clone())
    }
    pub fn button_on(&self, message_id: i32, label: &str) -> (i32, String) {
        let matching: Vec<_> = self
            .current_buttons_on(message_id)
            .into_iter()
            .filter(|(text, _, _)| text == label)
            .collect();
        let (_, id, data) = matching
            .last()
            .unwrap_or_else(|| panic!("missing rendered button {label} on {message_id}"));
        (*id, data.clone())
    }
    pub fn current_buttons_on(&self, message_id: i32) -> Vec<(String, i32, String)> {
        self.api
            .exchanges()
            .into_iter()
            .rev()
            .filter(|exchange| {
                exchange.response["ok"] == true
                    && exchange.response["result"]["message_id"] == message_id
            })
            .find_map(|exchange| {
                let markup: Value =
                    serde_json::from_str(exchange.fields.get("reply_markup")?).ok()?;
                Some(
                    markup["inline_keyboard"]
                        .as_array()?
                        .iter()
                        .flat_map(|row| row.as_array().into_iter().flatten())
                        .filter_map(|item| {
                            Some((
                                item["text"].as_str()?.to_owned(),
                                message_id,
                                item["callback_data"].as_str()?.to_owned(),
                            ))
                        })
                        .collect(),
                )
            })
            .unwrap_or_default()
    }
    pub fn successful_texts(&self) -> Vec<String> {
        self.api
            .exchanges()
            .into_iter()
            .filter(|exchange| {
                exchange.response["ok"] == true
                    && (exchange.path.ends_with("/SendMessage")
                        || exchange.path.ends_with("/EditMessageText"))
            })
            .filter_map(|exchange| exchange.fields.get("text").cloned())
            .collect()
    }
    pub fn last_text(&self) -> String {
        self.successful_texts().last().cloned().unwrap()
    }
    pub fn successful_sends(&self) -> Vec<Exchange> {
        self.api
            .exchanges()
            .into_iter()
            .filter(|exchange| {
                exchange.response["ok"] == true
                    && (exchange.path.ends_with("/SendMessage")
                        || exchange.path.ends_with("/SendPhoto"))
            })
            .collect()
    }
    pub async fn recommend(&mut self, query: &str, seed: i32) {
        assert!(self.say("/recommend").await);
        assert!(self.say(query).await);
        assert!(self.last_text().contains("Подтвердите аниме:"));
        let choice = self.button(&format!("MAL ID {seed}"));
        assert!(self.press(&choice).await);
    }
    pub async fn rate(&mut self, query: &str, seed: i32) {
        assert!(self.say("/rate").await);
        assert!(self.say(query).await);
        let choice = self.button(&format!("MAL ID {seed}"));
        assert!(self.press(&choice).await);
    }
}
