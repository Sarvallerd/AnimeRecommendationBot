import psycopg2
import pandas as pd


class DB:
    def __init__(self, db_config):
        self.config = db_config

    def send_sql_query(self, query: str):
        conn = psycopg2.connect(**self.config)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            if conn:
                cursor.close()
                conn.close()

    def save_user_info(self, tg_id, language_code, first_name, last_name, username, date, request):
        conn = psycopg2.connect(**self.config)
        try:
            cursor = conn.cursor()
            cursor.execute(f"""INSERT INTO users_requests (tg_id, language_code, first_name, last_name, username, date, request)
             VALUES ({tg_id}, '{language_code}', '{first_name}', '{last_name}', '{username}', '{date}', '{request}');""")
            conn.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            if conn:
                cursor.close()
                conn.close()

    def save_user_feedback(self, tg_id, language_code, first_name, last_name, username, date, feedback):
        conn = psycopg2.connect(**self.config)
        try:
            cursor = conn.cursor()
            cursor.execute(f"""INSERT INTO users_feedbacks (tg_id, language_code, first_name, last_name, username, date, feedbacks)
             VALUES ({tg_id}, '{language_code}', '{first_name}', '{last_name}', '{username}', '{date}', '{feedback}');""")
            conn.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            if conn:
                cursor.close()
                conn.close()