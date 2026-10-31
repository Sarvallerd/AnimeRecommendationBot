-- ARB-069: several titles per request, a fixed plan for every delivery and «Уже смотрел» marks.
-- Additive only. Positions delivered before plans existed become their own plan.
ALTER TABLE arb_delivered_positions
    ADD CONSTRAINT arb_delivered_positions_id_tg_id_mal_id_key UNIQUE (id, tg_id, mal_id);

-- Every listed title is a confirmed search of the same user. Position 1 is the request that
-- receives the deliveries; the other positions point at their own searches.
CREATE TABLE arb_request_seeds (
    request_id BIGINT NOT NULL,
    tg_id BIGINT NOT NULL,
    position SMALLINT NOT NULL CHECK (position BETWEEN 1 AND 5),
    seed_request_id BIGINT NOT NULL UNIQUE,
    mal_id INTEGER NOT NULL,
    PRIMARY KEY (request_id, position),
    UNIQUE (request_id, mal_id),
    CHECK ((position = 1) = (seed_request_id = request_id)),
    FOREIGN KEY (request_id, tg_id) REFERENCES arb_requests(id, tg_id),
    FOREIGN KEY (seed_request_id, tg_id, mal_id)
        REFERENCES arb_requests(id, tg_id, seed_mal_id)
);

-- Recommendations fixed before the first card is sent, so a retry shows the same titles even
-- after new ratings or marks. Every delivered position must belong to its request's plan.
CREATE TABLE arb_request_plans (
    request_id BIGINT NOT NULL,
    tg_id BIGINT NOT NULL,
    rank SMALLINT NOT NULL CHECK (rank BETWEEN 1 AND 5),
    mal_id INTEGER NOT NULL CHECK (mal_id > 0),
    PRIMARY KEY (request_id, rank),
    UNIQUE (request_id, mal_id),
    UNIQUE (request_id, rank, mal_id),
    FOREIGN KEY (request_id, tg_id) REFERENCES arb_requests(id, tg_id)
);
INSERT INTO arb_request_plans (request_id, tg_id, rank, mal_id)
SELECT request_id, tg_id, rank, mal_id FROM arb_delivered_positions;
ALTER TABLE arb_delivered_positions
    ADD CONSTRAINT arb_delivered_positions_plan_fkey FOREIGN KEY (request_id, rank, mal_id)
        REFERENCES arb_request_plans(request_id, rank, mal_id);

-- A title is marked once per user, from the card where «Уже смотрел» was pressed.
CREATE TABLE arb_watched_marks (
    tg_id BIGINT NOT NULL REFERENCES arb_users(tg_id),
    mal_id INTEGER NOT NULL,
    position_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tg_id, mal_id),
    FOREIGN KEY (position_id, tg_id, mal_id)
        REFERENCES arb_delivered_positions(id, tg_id, mal_id)
);
