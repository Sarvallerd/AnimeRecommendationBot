// The v1 handlers use the versioned API; compatibility adapters preserve legacy history.
#![allow(dead_code)]
use std::{error::Error, fmt};
use tokio::sync::{Mutex, MutexGuard};
use tokio_postgres::{Client, Config, NoTls};

pub type RequestId = i64;
pub type PositionId = i64;
pub type AnimeRatingEventId = i64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteOutcome<T> {
    Created(T),
    AlreadyRecorded(T),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbError {
    InvalidInput(&'static str),
    NotFound,
    Conflict,
    DatabaseFailure,
}
impl fmt::Display for DbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(f, "Invalid database input: {reason}"),
            Self::NotFound => f.write_str("Database record was not found"),
            Self::Conflict => f.write_str("Database record conflicts with an earlier action"),
            Self::DatabaseFailure => {
                f.write_str("Database operation failed; retry the action explicitly")
            }
        }
    }
}
impl Error for DbError {}
fn database_error(_: tokio_postgres::Error) -> DbError {
    DbError::DatabaseFailure
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserProfile {
    pub tg_id: i64,
    pub first_name: String,
    pub language_code: Option<String>,
    pub last_name: Option<String>,
    pub username: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryInput {
    pub rank: i16,
    pub mal_id: i32,
    pub chat_id: i64,
    pub message_id: i32,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveredPosition {
    pub id: PositionId,
    pub request_id: RequestId,
    pub rank: i16,
    pub mal_id: i32,
    pub chat_id: i64,
    pub message_id: i32,
}

pub struct Db {
    config: Config,
    client: Mutex<Client>,
}
impl Db {
    pub async fn new(config: &Config) -> Result<Self, DbError> {
        let client = Self::connect(config).await?;
        Ok(Self {
            config: config.clone(),
            client: Mutex::new(client),
        })
    }
    async fn connect(config: &Config) -> Result<Client, DbError> {
        let (client, connection) = config.connect(NoTls).await.map_err(database_error)?;
        tokio::spawn(async move {
            if connection.await.is_err() {
                eprintln!("Database connection was interrupted.");
            }
        });
        Ok(client)
    }
    async fn client(&self) -> Result<MutexGuard<'_, Client>, DbError> {
        let mut client = self.client.lock().await;
        if client.is_closed() {
            *client = Self::connect(&self.config).await?;
        }
        Ok(client)
    }
    pub async fn migrate(&self) -> Result<(), DbError> {
        let mut client = self.client().await?;
        let transaction = client.transaction().await.map_err(database_error)?;
        transaction
            .query_one("SELECT pg_advisory_xact_lock($1)", &[&0x4152_4230_3035_i64])
            .await
            .map_err(database_error)?;
        transaction.batch_execute(
            "CREATE TABLE IF NOT EXISTS arb_schema_migrations (version BIGINT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        ).await.map_err(database_error)?;
        let versions = transaction
            .query("SELECT version FROM arb_schema_migrations", &[])
            .await
            .map_err(database_error)?;
        if versions.iter().any(|row| row.get::<_, i64>(0) != 1) {
            return Err(DbError::Conflict);
        }
        if versions.is_empty() {
            transaction
                .batch_execute(include_str!("../migrations/0001_feedback_v1.sql"))
                .await
                .map_err(database_error)?;
            transaction
                .execute("INSERT INTO arb_schema_migrations(version) VALUES (1)", &[])
                .await
                .map_err(database_error)?;
        }
        transaction.commit().await.map_err(database_error)
    }
    pub async fn create(&self) -> Result<(), DbError> {
        self.migrate().await
    }

    pub async fn upsert_user(&self, user: &UserProfile) -> Result<(), DbError> {
        positive(user.tg_id)?;
        let client = self.client().await?;
        client
            .execute(
                "INSERT INTO arb_users (tg_id, first_name, language_code, last_name, username)
             VALUES ($1,$2,$3,$4,$5) ON CONFLICT (tg_id) DO UPDATE SET
             first_name=EXCLUDED.first_name, language_code=EXCLUDED.language_code,
             last_name=EXCLUDED.last_name, username=EXCLUDED.username, updated_at=now()",
                &[
                    &user.tg_id,
                    &user.first_name,
                    &user.language_code,
                    &user.last_name,
                    &user.username,
                ],
            )
            .await
            .map_err(database_error)?;
        Ok(())
    }
    pub async fn record_query(
        &self,
        tg_id: i64,
        action_key: &str,
        raw_query: &str,
    ) -> Result<WriteOutcome<RequestId>, DbError> {
        positive(tg_id)?;
        action(action_key)?;
        if raw_query.is_empty() {
            return Err(DbError::InvalidInput("empty query"));
        }
        let client = self.client().await?;
        let row = client
            .query_opt(
                "INSERT INTO arb_requests (tg_id, action_key, raw_query) VALUES ($1,$2,$3)
             ON CONFLICT (tg_id, action_key) DO NOTHING RETURNING id",
                &[&tg_id, &action_key, &raw_query],
            )
            .await
            .map_err(database_error)?;
        if let Some(row) = row {
            return Ok(WriteOutcome::Created(row.get(0)));
        }
        let row = client
            .query_one(
                "SELECT id, raw_query FROM arb_requests WHERE tg_id=$1 AND action_key=$2",
                &[&tg_id, &action_key],
            )
            .await
            .map_err(database_error)?;
        if row.get::<_, String>(1) != raw_query {
            return Err(DbError::Conflict);
        }
        Ok(WriteOutcome::AlreadyRecorded(row.get(0)))
    }
    pub async fn resolve_request(
        &self,
        tg_id: i64,
        request_id: RequestId,
        seed_mal_id: i32,
        bundle_id: &str,
    ) -> Result<WriteOutcome<()>, DbError> {
        positive(tg_id)?;
        positive(request_id)?;
        if seed_mal_id <= 0 {
            return Err(DbError::InvalidInput("invalid MAL ID"));
        }
        bundle(bundle_id)?;
        let client = self.client().await?;
        let updated = client
            .execute(
                "UPDATE arb_requests SET seed_mal_id=$3, bundle_id=$4, resolved_at=now()
             WHERE id=$1 AND tg_id=$2 AND resolved_at IS NULL",
                &[&request_id, &tg_id, &seed_mal_id, &bundle_id],
            )
            .await
            .map_err(database_error)?;
        if updated == 1 {
            return Ok(WriteOutcome::Created(()));
        }
        let row = client
            .query_opt(
                "SELECT seed_mal_id, bundle_id FROM arb_requests WHERE id=$1 AND tg_id=$2",
                &[&request_id, &tg_id],
            )
            .await
            .map_err(database_error)?
            .ok_or(DbError::NotFound)?;
        if row.get::<_, Option<i32>>(0) == Some(seed_mal_id)
            && row.get::<_, Option<String>>(1).as_deref() == Some(bundle_id)
        {
            Ok(WriteOutcome::AlreadyRecorded(()))
        } else {
            Err(DbError::Conflict)
        }
    }
    pub async fn record_delivery(
        &self,
        tg_id: i64,
        request_id: RequestId,
        input: &DeliveryInput,
    ) -> Result<WriteOutcome<PositionId>, DbError> {
        positive(tg_id)?;
        positive(request_id)?;
        if !(1..=5).contains(&input.rank) {
            return Err(DbError::InvalidInput("invalid rank"));
        }
        if input.mal_id <= 0 {
            return Err(DbError::InvalidInput("invalid MAL ID"));
        }
        if input.message_id <= 0 {
            return Err(DbError::InvalidInput("invalid message ID"));
        }
        let client = self.client().await?;
        let row = client
            .query_opt(
                "SELECT seed_mal_id FROM arb_requests WHERE id=$1 AND tg_id=$2",
                &[&request_id, &tg_id],
            )
            .await
            .map_err(database_error)?
            .ok_or(DbError::NotFound)?;
        let seed: i32 = row.get::<_, Option<i32>>(0).ok_or(DbError::Conflict)?;
        if seed == input.mal_id {
            return Err(DbError::Conflict);
        }
        let row = client
            .query_opt(
                "INSERT INTO arb_delivered_positions
             (request_id,tg_id,rank,mal_id,chat_id,message_id)
             VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING RETURNING id",
                &[
                    &request_id,
                    &tg_id,
                    &input.rank,
                    &input.mal_id,
                    &input.chat_id,
                    &input.message_id,
                ],
            )
            .await
            .map_err(database_error)?;
        if let Some(row) = row {
            return Ok(WriteOutcome::Created(row.get(0)));
        }
        let row = client
            .query_opt(
                "SELECT id,mal_id,chat_id,message_id FROM arb_delivered_positions
             WHERE request_id=$1 AND rank=$2",
                &[&request_id, &input.rank],
            )
            .await
            .map_err(database_error)?
            .ok_or(DbError::Conflict)?;
        if row.get::<_, i32>(1) == input.mal_id
            && row.get::<_, i64>(2) == input.chat_id
            && row.get::<_, i32>(3) == input.message_id
        {
            Ok(WriteOutcome::AlreadyRecorded(row.get(0)))
        } else {
            Err(DbError::Conflict)
        }
    }
    pub async fn list_delivered_positions(
        &self,
        tg_id: i64,
        request_id: RequestId,
    ) -> Result<Vec<DeliveredPosition>, DbError> {
        positive(tg_id)?;
        positive(request_id)?;
        let client = self.client().await?;
        if client
            .query_opt(
                "SELECT 1 FROM arb_requests WHERE id=$1 AND tg_id=$2",
                &[&request_id, &tg_id],
            )
            .await
            .map_err(database_error)?
            .is_none()
        {
            return Err(DbError::NotFound);
        }
        let rows = client
            .query(
                "SELECT id,request_id,rank,mal_id,chat_id,message_id FROM arb_delivered_positions
             WHERE request_id=$1 AND tg_id=$2 ORDER BY rank",
                &[&request_id, &tg_id],
            )
            .await
            .map_err(database_error)?;
        Ok(rows
            .into_iter()
            .map(|row| DeliveredPosition {
                id: row.get(0),
                request_id: row.get(1),
                rank: row.get(2),
                mal_id: row.get(3),
                chat_id: row.get(4),
                message_id: row.get(5),
            })
            .collect())
    }
    pub async fn rate_anime(
        &self,
        tg_id: i64,
        request_id: RequestId,
        score: i16,
    ) -> Result<WriteOutcome<AnimeRatingEventId>, DbError> {
        positive(tg_id)?;
        positive(request_id)?;
        if !(1..=10).contains(&score) {
            return Err(DbError::InvalidInput("invalid anime score"));
        }
        let mut client = self.client().await?;
        let transaction = client.transaction().await.map_err(database_error)?;
        let row = transaction
            .query_opt(
                "SELECT seed_mal_id FROM arb_requests WHERE id=$1 AND tg_id=$2 FOR UPDATE",
                &[&request_id, &tg_id],
            )
            .await
            .map_err(database_error)?
            .ok_or(DbError::NotFound)?;
        let mal_id: i32 = row.get::<_, Option<i32>>(0).ok_or(DbError::Conflict)?;
        if let Some(row) = transaction
            .query_opt(
                "SELECT id,score FROM arb_anime_rating_events WHERE request_id=$1",
                &[&request_id],
            )
            .await
            .map_err(database_error)?
        {
            if row.get::<_, i16>(1) != score {
                return Err(DbError::Conflict);
            }
            return Ok(WriteOutcome::AlreadyRecorded(row.get(0)));
        }
        let row = transaction
            .query_one(
                "INSERT INTO arb_anime_rating_events (request_id,tg_id,mal_id,score)
             VALUES ($1,$2,$3,$4) RETURNING id",
                &[&request_id, &tg_id, &mal_id, &score],
            )
            .await
            .map_err(database_error)?;
        let event_id: i64 = row.get(0);
        transaction
            .execute(
                "INSERT INTO arb_anime_ratings (tg_id,mal_id,current_event_id) VALUES ($1,$2,$3)
             ON CONFLICT (tg_id,mal_id) DO UPDATE SET
             current_event_id=EXCLUDED.current_event_id, updated_at=now()
             WHERE arb_anime_ratings.current_event_id < EXCLUDED.current_event_id",
                &[&tg_id, &mal_id, &event_id],
            )
            .await
            .map_err(database_error)?;
        transaction.commit().await.map_err(database_error)?;
        Ok(WriteOutcome::Created(event_id))
    }
    pub async fn rate_recommendation(
        &self,
        tg_id: i64,
        position_id: PositionId,
        score: i16,
    ) -> Result<WriteOutcome<()>, DbError> {
        positive(tg_id)?;
        positive(position_id)?;
        if !(0..=5).contains(&score) {
            return Err(DbError::InvalidInput("invalid recommendation score"));
        }
        let client = self.client().await?;
        let row = client
            .query_opt(
                "INSERT INTO arb_recommendation_ratings (position_id,tg_id,score)
             SELECT id,tg_id,$3 FROM arb_delivered_positions WHERE id=$1 AND tg_id=$2
             ON CONFLICT (position_id) DO NOTHING RETURNING position_id",
                &[&position_id, &tg_id, &score],
            )
            .await
            .map_err(database_error)?;
        if row.is_some() {
            return Ok(WriteOutcome::Created(()));
        }
        let row = client
            .query_opt(
                "SELECT score FROM arb_recommendation_ratings WHERE position_id=$1 AND tg_id=$2",
                &[&position_id, &tg_id],
            )
            .await
            .map_err(database_error)?;
        match row {
            Some(row) if row.get::<_, i16>(0) == score => Ok(WriteOutcome::AlreadyRecorded(())),
            Some(_) => Err(DbError::Conflict),
            None => Err(DbError::NotFound),
        }
    }
    pub async fn save_feedback(
        &self,
        tg_id: i64,
        action_key: &str,
        body: &str,
    ) -> Result<WriteOutcome<i64>, DbError> {
        positive(tg_id)?;
        action(action_key)?;
        let client = self.client().await?;
        let row = client
            .query_opt(
                "INSERT INTO arb_feedback (tg_id,action_key,body) VALUES ($1,$2,$3)
             ON CONFLICT (tg_id,action_key) DO NOTHING RETURNING id",
                &[&tg_id, &action_key, &body],
            )
            .await
            .map_err(database_error)?;
        if let Some(row) = row {
            return Ok(WriteOutcome::Created(row.get(0)));
        }
        let row = client
            .query_one(
                "SELECT id,body FROM arb_feedback WHERE tg_id=$1 AND action_key=$2",
                &[&tg_id, &action_key],
            )
            .await
            .map_err(database_error)?;
        if row.get::<_, String>(1) != body {
            return Err(DbError::Conflict);
        }
        Ok(WriteOutcome::AlreadyRecorded(row.get(0)))
    }
    // Compatibility methods for the current runtime. Legacy rows have no v1 provenance.
    pub async fn insert_user(
        &self,
        tg_id: String,
        language_code: String,
        first_name: String,
        last_name: String,
        username: String,
    ) -> Result<(), DbError> {
        let tg_id = legacy_id(&tg_id)?;
        let client = self.client().await?;
        client
            .execute(
                "INSERT INTO test_users(tg_id,language_code,first_name,last_name,username)
             VALUES ($1,$2,$3,$4,$5) ON CONFLICT (tg_id) DO NOTHING",
                &[&tg_id, &language_code, &first_name, &last_name, &username],
            )
            .await
            .map_err(database_error)?;
        Ok(())
    }
    pub async fn insert_msg(
        &self,
        msg_type: &str,
        tg_id: String,
        msg: String,
    ) -> Result<(), DbError> {
        let tg_id = legacy_id(&tg_id)?;
        let table = match msg_type {
            "feedback" => "test_feedback",
            "request" => "test_request",
            _ => return Err(DbError::InvalidInput("unknown message kind")),
        };
        let client = self.client().await?;
        let sql = format!("INSERT INTO {table} (tg_id,msg) VALUES ($1,$2)");
        client
            .execute(&sql, &[&tg_id, &msg])
            .await
            .map_err(database_error)?;
        Ok(())
    }
}
fn positive(id: i64) -> Result<(), DbError> {
    if id > 0 {
        Ok(())
    } else {
        Err(DbError::InvalidInput("ID must be positive"))
    }
}
fn legacy_id(value: &str) -> Result<i64, DbError> {
    let id = value
        .parse::<i64>()
        .map_err(|_| DbError::InvalidInput("invalid Telegram ID"))?;
    positive(id)?;
    Ok(id)
}
fn action(value: &str) -> Result<(), DbError> {
    if value.is_empty() {
        Err(DbError::InvalidInput("empty action key"))
    } else {
        Ok(())
    }
}
fn bundle(value: &str) -> Result<(), DbError> {
    if value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(DbError::InvalidInput("invalid bundle ID"))
    }
}
#[cfg(test)]
mod tests;
