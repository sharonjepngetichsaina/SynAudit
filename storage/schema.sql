-- Starter schema; each area adds its own tables as needed.

CREATE TABLE IF NOT EXISTS runs (
    run_id      SERIAL PRIMARY KEY,
    dataset     TEXT NOT NULL,
    generator   TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per metric per window, e.g. metric_group = 'drift' | 'quality' | 'utility' | 'privacy'
CREATE TABLE IF NOT EXISTS metrics (
    id            BIGSERIAL PRIMARY KEY,
    run_id        INTEGER REFERENCES runs(run_id),
    metric_group  TEXT NOT NULL,
    metric_name   TEXT NOT NULL,
    value         DOUBLE PRECISION,
    window_start  TIMESTAMPTZ,
    window_end    TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
