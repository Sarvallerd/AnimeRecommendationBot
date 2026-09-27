INSERT INTO arb_users(tg_id,first_name,language_code) VALUES (42,'Мария','ru'),(43,'Алексей','ru');
INSERT INTO arb_requests(tg_id,action_key,raw_query,seed_mal_id,bundle_id,resolved_at)
VALUES (42,'q1',E'космос\nкорабль',1,'sha256:1111111111111111111111111111111111111111111111111111111111111111',now()),
       (43,'q2','море',3,'sha256:1111111111111111111111111111111111111111111111111111111111111111',now());
INSERT INTO arb_delivered_positions(request_id,tg_id,rank,mal_id,chat_id,message_id)
SELECT id,tg_id,1,2,tg_id,100 FROM arb_requests;
INSERT INTO arb_anime_rating_events(request_id,tg_id,mal_id,score)
SELECT id,tg_id,seed_mal_id,9 FROM arb_requests;
INSERT INTO arb_anime_ratings(tg_id,mal_id,current_event_id)
SELECT tg_id,mal_id,id FROM arb_anime_rating_events;
INSERT INTO arb_recommendation_ratings(position_id,tg_id,score)
SELECT id,tg_id,5 FROM arb_delivered_positions;
INSERT INTO arb_feedback(tg_id,action_key,body) VALUES (42,'f1',E'хорошо\nспасибо'),(43,'f2','да');
