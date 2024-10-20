use postgres::{Client, NoTls};

pub trait DBActions {
    fn create_client(&mut self);
    fn create_table(&mut self);
    fn insert_row_users(
        &mut self,
        tg_id: i64,
        language_code: &str,
        first_name: &str,
        last_name: &str,
        username: &str,
    );
    fn insert_msg(&mut self, msg_type: &str, tg_id: i64, msg: &str);
}
pub struct DB {
    pub host: String,
    pub user: String,
    pub password: String,
    pub port: String,
    pub db_name: String,
    pub client: Option<Client>,
}

impl DBActions for DB {
    fn create_client(&mut self) {
        let params = format!(
            "postgresql://{0}:{1}@{2}:{3}/{4}",
            self.user, self.password, self.host, self.port, self.db_name
        );
        self.client = Some(Client::connect(params.as_str(), NoTls).unwrap());
    }
    fn create_table(&mut self) {
        let client = self.client.as_mut().unwrap();

        client
            .batch_execute(
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
            )
            .unwrap();

        client
            .batch_execute(
                "
         CREATE TABLE IF NOT EXISTS test_request
            (
                tg_Id INTEGER,
                msg VARCHAR
            )
            ",
            )
            .unwrap();
        client
            .batch_execute(
                "
         CREATE TABLE IF NOT EXISTS test_feedback
            (
                tg_Id INTEGER,
                msg VARCHAR
            )
            ",
            )
            .unwrap();
    }

    fn insert_row_users(
        &mut self,
        tg_id: i64,
        language_code: &str,
        first_name: &str,
        last_name: &str,
        username: &str,
    ) {
        let client = self.client.as_mut().unwrap();
        let query = format!(
            "INSERT INTO test_users(tg_Id, language_code, first_name, last_name, username)
            SELECT {0}, '{1}', '{2}', '{3}', '{4}'
                WHERE
                NOT EXISTS (
                    SELECT tg_Id FROM test_users WHERE tg_id = {0}
                    );",
            tg_id, language_code, first_name, last_name, username
        );

        client.batch_execute(query.as_str()).unwrap();
    }
    fn insert_msg(&mut self, msg_type: &str, tg_id: i64, msg: &str) {
        let client = self.client.as_mut().unwrap();
        match msg_type {
            "feedback" => {
                let query = format!(
                    "INSERT INTO test_feedback (tg_id, msg)
                    VALUES ({0}, '{1}');",
                    tg_id, msg
                );
                client.batch_execute(query.as_str()).unwrap();
            }
            "request" => {
                let query = format!(
                    "INSERT INTO test_request (tg_id, msg)
                    VALUES ({0}, '{1}');",
                    tg_id, msg
                );
                client.batch_execute(query.as_str()).unwrap();
            }
            _ => {}
        }
    }
}
