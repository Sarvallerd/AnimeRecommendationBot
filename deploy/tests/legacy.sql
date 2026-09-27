CREATE TABLE test_users (tg_id SERIAL PRIMARY KEY, language_code VARCHAR, first_name VARCHAR, last_name VARCHAR, username VARCHAR);
CREATE TABLE test_request (tg_id INTEGER, msg VARCHAR);
CREATE TABLE test_feedback (tg_id INTEGER, msg VARCHAR);
INSERT INTO test_users(tg_id,language_code,first_name,last_name,username) VALUES (42,'ru','Мария','Тест','maria');
INSERT INTO test_request(tg_id,msg) VALUES (42,E'первая строка\nвторая строка');
INSERT INTO test_feedback(tg_id,msg) VALUES (42,E'отлично\nспасибо');
