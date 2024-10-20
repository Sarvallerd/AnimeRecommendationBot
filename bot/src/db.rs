use std::env;
use tokio_postgres::{Client, Error, NoTls};

#[derive(Debug)]
pub struct Db {
    client: Client,
}

impl Db {
    pub async fn new() -> Result<Self, Error> {
        let (client, connection) = tokio_postgres::connect(
            &format!(
                "postgresql://{0}:{1}@{2}:{3}/{4}",
                env::var("db_user").expect("$db_user is not set!"),
                env::var("db_password").expect("$db_password is not set!"),
                "localhost",
                env::var("db_port").expect("$db_port is not set!"),
                env::var("db_name").expect("$db_name is not set!")
            ),
            NoTls,
        )
        .await?;

        // Spawn a task to manage the connection.
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("Connection error: {}", e);
            }
        });

        Ok(Self { client })
    }
    pub async fn insert_user(
        &self,
        tg_id: String,
        language_code: String,
        first_name: String,
        last_name: String,
        username: String,
    ) -> Result<(), Error> {
        let query = format!(
            "INSERT INTO test_users(tg_Id, language_code, first_name, last_name, username)
            SELECT {0}, '{1}', '{2}', '{3}', '{4}'
                WHERE
                NOT EXISTS (
                    SELECT tg_Id FROM test_users WHERE tg_id = {0}
                    );",
            tg_id, language_code, first_name, last_name, username
        );
        self.client.execute(&query, &[]).await?;
        Ok(())
    }
    pub async fn insert_msg(
        &self,
        msg_type: &str,
        tg_id: String,
        msg: String,
    ) -> Result<(), Error> {
        match msg_type {
            "feedback" => {
                let query = format!(
                    "INSERT INTO test_feedback (tg_id, msg)
                    VALUES ({0}, '{1}');",
                    tg_id, msg
                );
                self.client.execute(&query, &[]).await?;
            }
            "request" => {
                let query = format!(
                    "INSERT INTO test_request (tg_id, msg)
                    VALUES ({0}, '{1}');",
                    tg_id, msg
                );
                self.client.execute(&query, &[]).await?;
            }
            _ => {}
        }
        Ok(())
    }
    pub async fn create(&self) -> Result<(), Error> {
        self.client
            .execute(
                "
         CREATE TABLE IF NOT EXISTS test_users
            (
                tg_Id SERIAL PRIMARY KEY,
                language_code VARCHAR,
                first_name VARCHAR,
                last_name VARCHAR,
                username VARCHAR
            )
            ",
                &[],
            )
            .await?;
        self.client
            .execute(
                "
         CREATE TABLE IF NOT EXISTS test_request
            (
                tg_Id INTEGER,
                msg VARCHAR
            )
            ",
                &[],
            )
            .await?;
        self.client
            .execute(
                "
         CREATE TABLE IF NOT EXISTS test_feedback
            (
                tg_Id INTEGER,
                msg VARCHAR
            )
            ",
                &[],
            )
            .await?;
        Ok(())
    }
}
