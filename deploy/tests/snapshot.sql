SELECT jsonb_build_object(
  'legacy_users',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id) FROM test_users t),
  'legacy_requests',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id,msg) FROM test_request t),
  'legacy_feedback',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id,msg) FROM test_feedback t),
  'users',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id) FROM arb_users t),
  'requests',(SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_requests t),
  'delivered',(SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_delivered_positions t),
  'anime_events',(SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_anime_rating_events t),
  'anime_current',(SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id,mal_id) FROM arb_anime_ratings t),
  'recommendation_ratings',(SELECT jsonb_agg(to_jsonb(t) ORDER BY position_id) FROM arb_recommendation_ratings t),
  'feedback',(SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_feedback t),
  'migrations',(SELECT jsonb_agg(version ORDER BY version) FROM arb_schema_migrations)
)::text;
