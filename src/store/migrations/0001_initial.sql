-- Signal Zero schema, Phase 1.
--
-- The features/outcomes separation is baked in here on purpose. Anything that could
-- only be known after publication lives in an outcome table, and nothing under
-- src/features/ or src/score/ is allowed to read those tables. See
-- src/store/db.py:open_feature_scoped and tests/test_no_leakage.py.

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,   -- base id, no version suffix
    version           INTEGER,
    title             TEXT NOT NULL,
    abstract          TEXT NOT NULL,
    comments          TEXT,
    journal_ref       TEXT,
    doi               TEXT,
    primary_category  TEXT,
    categories        TEXT NOT NULL DEFAULT '[]',   -- json array
    authors           TEXT NOT NULL DEFAULT '[]',   -- json array of display names
    author_count      INTEGER NOT NULL DEFAULT 0,
    affiliations      TEXT NOT NULL DEFAULT '[]',   -- json array, often empty in the API
    submitted_at      TEXT,               -- ISO 8601, arXiv "published"
    updated_at        TEXT,               -- ISO 8601, arXiv "updated"
    announced_date    TEXT NOT NULL,      -- YYYY-MM-DD, the day we treat as t=0
    abs_url           TEXT,
    pdf_url           TEXT,
    links             TEXT NOT NULL DEFAULT '[]',   -- json array of related links
    source            TEXT NOT NULL,      -- arxiv_api | arxiv_rss
    ingested_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_announced ON papers (announced_date);
CREATE INDEX IF NOT EXISTS idx_papers_primary_cat ON papers (primary_category);

-- Day-zero features only. Every column here must be computable from information that
-- existed at publication time.
CREATE TABLE IF NOT EXISTS paper_features (
    arxiv_id              TEXT PRIMARY KEY REFERENCES papers (arxiv_id),

    -- structural, deterministic
    has_code_link         INTEGER NOT NULL DEFAULT 0,
    code_url              TEXT,
    code_link_source      TEXT,           -- abstract | comments | links
    claims_numbers        INTEGER NOT NULL DEFAULT 0,
    benchmark_claims      TEXT NOT NULL DEFAULT '[]',   -- json array of parsed claims
    max_point_gain        REAL,           -- absolute points over stated prior art
    max_relative_gain     REAL,           -- multiple, e.g. 3.0 for "3x faster"
    benchmark_note        TEXT,
    cross_listed          INTEGER NOT NULL DEFAULT 0,
    category_count        INTEGER NOT NULL DEFAULT 1,
    claims_system         INTEGER NOT NULL DEFAULT 0,   -- a system, not only a method
    application_domain    TEXT,
    novel_terms           TEXT NOT NULL DEFAULT '[]',   -- json array, unseen in corpus
    abstract_length       INTEGER NOT NULL DEFAULT 0,

    -- author prior, from OpenAlex with a publication-date cutoff
    author_prior          REAL,           -- 0..1, NULL when unknown
    author_prior_coverage INTEGER NOT NULL DEFAULT 0,   -- 1 when the prior is trustworthy
    author_prior_detail   TEXT NOT NULL DEFAULT '{}',   -- json

    -- LLM capability delta, four axes 0..3
    cost_curve            INTEGER,
    cost_curve_why        TEXT,
    constraint_removal    INTEGER,
    constraint_removal_why TEXT,
    usability_threshold   INTEGER,
    usability_threshold_why TEXT,
    modality_opening      INTEGER,
    modality_opening_why  TEXT,
    capability_total      INTEGER,
    compute_band          TEXT,           -- consumer | single_node | small_cluster | frontier | unknown
    compute_band_why      TEXT,
    capability_source     TEXT,           -- provider:model, or heuristic
    prompt_version        TEXT,

    features_version      TEXT NOT NULL,
    extracted_at          TEXT NOT NULL
);

-- The prediction log. Every paper scored gets a row, including the low scores.
CREATE TABLE IF NOT EXISTS paper_scores (
    arxiv_id        TEXT NOT NULL REFERENCES papers (arxiv_id),
    run_date        TEXT NOT NULL,        -- YYYY-MM-DD of the scoring run
    scorer_version  TEXT NOT NULL,
    score           REAL NOT NULL,        -- 0..1 combined, prior weighted in
    claim_score     REAL NOT NULL,        -- 0..1 claim size, no prior
    bucket          TEXT,                 -- confidence | high_variance | NULL
    rank            INTEGER,              -- rank within the bucket, 1 based
    components      TEXT NOT NULL DEFAULT '{}',   -- json, per-component contributions
    shortlisted     INTEGER NOT NULL DEFAULT 0,
    summary         TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, run_date, scorer_version)
);
CREATE INDEX IF NOT EXISTS idx_scores_run ON paper_scores (run_date, bucket, rank);

-- Author prior cache. cutoff_date is '' for a live run and a date for a backtest run,
-- so historical priors never contaminate live ones and the other way round.
CREATE TABLE IF NOT EXISTS author_priors (
    author_key      TEXT NOT NULL,        -- normalised author name
    cutoff_date     TEXT NOT NULL DEFAULT '',
    openalex_id     TEXT,
    display_name    TEXT,
    works_count     INTEGER,
    citations_total INTEGER,
    citations_per_year REAL,
    first_year      INTEGER,
    prior           REAL,                 -- 0..1, NULL when unknown
    coverage        INTEGER NOT NULL DEFAULT 0,
    detail          TEXT NOT NULL DEFAULT '{}',
    computed_at     TEXT NOT NULL,
    PRIMARY KEY (author_key, cutoff_date)
);

-- Corpus vocabulary, used for the first-appearance-of-a-term signal.
CREATE TABLE IF NOT EXISTS corpus_terms (
    term            TEXT PRIMARY KEY,
    first_seen_date TEXT NOT NULL,
    paper_count     INTEGER NOT NULL DEFAULT 1
);

-- Lab blog posts. Low volume, occasionally ahead of arXiv.
CREATE TABLE IF NOT EXISTS lab_posts (
    url         TEXT PRIMARY KEY,
    lab         TEXT NOT NULL,
    title       TEXT NOT NULL,
    summary     TEXT,
    published   TEXT,
    ingested_at TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Outcome side. Written by src/outcomes/, never read by features or scoring.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS paper_outcomes (
    arxiv_id          TEXT NOT NULL REFERENCES papers (arxiv_id),
    horizon_days      INTEGER NOT NULL,   -- 7 | 30 | 90
    observed_at       TEXT NOT NULL,      -- YYYY-MM-DD the observation was taken
    hf_listed         INTEGER,
    hf_upvotes        INTEGER,
    github_repo       TEXT,
    github_stars      INTEGER,
    github_forks      INTEGER,
    citations_openalex INTEGER,
    citations_s2      INTEGER,
    hn_hits           INTEGER,
    hn_points         INTEGER,
    hn_comments       INTEGER,
    hn_url            TEXT,
    sources_ok        TEXT NOT NULL DEFAULT '{}',   -- json, which sources answered
    raw               TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (arxiv_id, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_observed ON paper_outcomes (observed_at);

CREATE TABLE IF NOT EXISTS outcome_labels (
    arxiv_id     TEXT PRIMARY KEY REFERENCES papers (arxiv_id),
    horizon_days INTEGER NOT NULL,
    composite    REAL NOT NULL,
    hit          INTEGER NOT NULL,
    parts        TEXT NOT NULL DEFAULT '{}',
    computed_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Operational bookkeeping.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,            -- daily | outcomes | weekly | backfill
    run_date    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,            -- running | ok | partial | failed
    window_from TEXT,
    window_to   TEXT,
    stats       TEXT NOT NULL DEFAULT '{}',
    notes       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs (kind, run_date);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER,
    ts            TEXT NOT NULL,
    purpose       TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    ok            INTEGER NOT NULL DEFAULT 1,
    error         TEXT
);
