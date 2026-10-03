BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL TIME ZONE 'UTC';
SELECT jsonb_build_object(
  'migrations', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY version) FROM arb_schema_migrations t), '[]'::jsonb),
  'users', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id) FROM arb_users t), '[]'::jsonb),
  'requests', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_requests t), '[]'::jsonb),
  'delivered', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_delivered_positions t), '[]'::jsonb),
  'anime_events', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_anime_rating_events t), '[]'::jsonb),
  'anime_current', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id, mal_id) FROM arb_anime_ratings t), '[]'::jsonb),
  'recommendation_ratings', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY position_id) FROM arb_recommendation_ratings t), '[]'::jsonb),
  'feedback', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY id) FROM arb_feedback t), '[]'::jsonb),
  'legacy_users', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id, language_code, first_name, last_name, username) FROM test_users t), '[]'::jsonb),
  'legacy_requests', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id, msg) FROM test_request t), '[]'::jsonb),
  'legacy_feedback', COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY tg_id, msg) FROM test_feedback t), '[]'::jsonb)
)::text;
COMMIT;
