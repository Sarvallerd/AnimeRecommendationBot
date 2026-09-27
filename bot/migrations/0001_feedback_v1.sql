-- The three test_* tables predate versioned migrations. Preserve their data.
CREATE TABLE IF NOT EXISTS test_users (
    tg_id BIGSERIAL PRIMARY KEY,
    language_code VARCHAR,
    first_name VARCHAR,
    last_name VARCHAR,
    username VARCHAR
);
CREATE TABLE IF NOT EXISTS test_request (tg_id BIGINT, msg VARCHAR);
CREATE TABLE IF NOT EXISTS test_feedback (tg_id BIGINT, msg VARCHAR);
ALTER TABLE test_users ALTER COLUMN tg_id TYPE BIGINT;
ALTER TABLE test_request ALTER COLUMN tg_id TYPE BIGINT;
ALTER TABLE test_feedback ALTER COLUMN tg_id TYPE BIGINT;

CREATE TABLE arb_users (
    tg_id BIGINT PRIMARY KEY CHECK (tg_id > 0),
    language_code TEXT,
    first_name TEXT NOT NULL,
    last_name TEXT,
    username TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE arb_requests (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tg_id BIGINT NOT NULL REFERENCES arb_users(tg_id),
    action_key TEXT NOT NULL,
    raw_query TEXT NOT NULL,
    seed_mal_id INTEGER,
    bundle_id TEXT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tg_id, action_key),
    UNIQUE (id, tg_id),
    UNIQUE (id, tg_id, seed_mal_id),
    CHECK ((seed_mal_id IS NULL AND bundle_id IS NULL AND resolved_at IS NULL)
        OR (seed_mal_id IS NOT NULL AND seed_mal_id > 0
            AND bundle_id IS NOT NULL AND bundle_id ~ '^sha256:[0-9a-f]{64}$'
            AND resolved_at IS NOT NULL))
);
CREATE TABLE arb_delivered_positions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id BIGINT NOT NULL,
    tg_id BIGINT NOT NULL,
    rank SMALLINT NOT NULL CHECK (rank BETWEEN 1 AND 5),
    mal_id INTEGER NOT NULL CHECK (mal_id > 0),
    chat_id BIGINT NOT NULL,
    message_id INTEGER NOT NULL CHECK (message_id > 0),
    delivered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (request_id, tg_id) REFERENCES arb_requests(id, tg_id),
    UNIQUE (request_id, rank),
    UNIQUE (request_id, mal_id),
    UNIQUE (id, tg_id)
);
CREATE TABLE arb_anime_rating_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id BIGINT NOT NULL UNIQUE,
    tg_id BIGINT NOT NULL,
    mal_id INTEGER NOT NULL,
    score SMALLINT NOT NULL CHECK (score BETWEEN 1 AND 10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (request_id, tg_id, mal_id)
        REFERENCES arb_requests(id, tg_id, seed_mal_id),
    UNIQUE (id, tg_id, mal_id)
);
CREATE TABLE arb_anime_ratings (
    tg_id BIGINT NOT NULL REFERENCES arb_users(tg_id),
    mal_id INTEGER NOT NULL CHECK (mal_id > 0),
    current_event_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tg_id, mal_id),
    FOREIGN KEY (current_event_id, tg_id, mal_id)
        REFERENCES arb_anime_rating_events(id, tg_id, mal_id)
);
CREATE TABLE arb_recommendation_ratings (
    position_id BIGINT PRIMARY KEY,
    tg_id BIGINT NOT NULL,
    score SMALLINT NOT NULL CHECK (score BETWEEN 0 AND 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (position_id, tg_id)
        REFERENCES arb_delivered_positions(id, tg_id)
);
CREATE TABLE arb_feedback (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tg_id BIGINT NOT NULL REFERENCES arb_users(tg_id),
    action_key TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tg_id, action_key)
);
