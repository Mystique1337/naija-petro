-- ============================================================================
-- Naija-Petro RAG - vector store schema (Postgres + pgvector)
--
-- Works on a self-hosted Supabase OR a plain Postgres. Run once, either by
-- pasting this file into the Supabase Studio SQL editor or with:
--     psql "$SUPABASE_DB_URL" -f supabase/schema.sql
--
-- SCHEMA: everything is created in the app's own schema so one Supabase
-- instance can host several apps. It must match SUPABASE_DB_SCHEMA in .env
-- (default naija_petro). For a single-tenant database, change the two lines
-- below to `SET search_path = public, extensions;` and drop the CREATE SCHEMA.
--
-- On Supabase, also add this schema to the exposed schemas in
-- Project Settings > API (PGRST_DB_SCHEMAS), otherwise PostgREST cannot see it.
--
-- Embedding dimension is 768 (nomic-embed-text-v1.5). If you change EMBED_MODEL
-- to a different dimension, update every `vector(768)` below and re-run.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS naija_petro;
SET search_path = naija_petro, public, extensions;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- fuzzy text helpers (optional)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------------------------------------------------------------------------
-- documents : one row per ingested source (web page, PDF, seed doc)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title         TEXT,
    source_url    TEXT,
    domain        TEXT,
    source_tier   SMALLINT DEFAULT 3,        -- 1=official/regulatory, 2=reference, 3=news/other
    published_date DATE,
    content       TEXT NOT NULL,
    content_hash  TEXT UNIQUE NOT NULL,       -- SHA-256 of normalised content (dedup)
    retrieved_at  TIMESTAMPTZ DEFAULT now(),
    created_at    TIMESTAMPTZ DEFAULT now(),
    metadata      JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_documents_domain      ON documents (domain);
CREATE INDEX IF NOT EXISTS idx_documents_tier        ON documents (source_tier);
CREATE INDEX IF NOT EXISTS idx_documents_retrieved   ON documents (retrieved_at);

-- ---------------------------------------------------------------------------
-- document_chunks : retrievable units, one embedding each
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    embedding    VECTOR(768),
    token_count  INT,
    metadata     JSONB DEFAULT '{}'::jsonb,    -- denormalised citation fields for fast reads
    created_at   TIMESTAMPTZ DEFAULT now(),
    fts          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, chunk_index)
);

-- Approximate-NN index for cosine similarity (HNSW: best default for < ~1M rows).
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
-- Full-text and metadata indexes for hybrid search + filtering.
CREATE INDEX IF NOT EXISTS idx_chunks_fts       ON document_chunks USING GIN (fts);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata  ON document_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_chunks_document  ON document_chunks (document_id);

-- ---------------------------------------------------------------------------
-- match_documents : pure semantic (cosine) search
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding      VECTOR(768),
    match_count          INT DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id    BIGINT,
    document_id UUID,
    content     TEXT,
    similarity  FLOAT,
    source_url  TEXT,
    title       TEXT,
    domain      TEXT,
    source_tier SMALLINT,
    metadata    JSONB
)
LANGUAGE sql STABLE AS $$
    SELECT c.id, c.document_id, c.content,
           1 - (c.embedding <=> query_embedding) AS similarity,
           d.source_url, d.title, d.domain, d.source_tier, c.metadata
    FROM document_chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.embedding IS NOT NULL
      AND 1 - (c.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- ---------------------------------------------------------------------------
-- hybrid_search : semantic + full-text fused with Reciprocal Rank Fusion.
-- websearch_to_tsquery handles arbitrary user input safely.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION hybrid_search (
    query_text       TEXT,
    query_embedding  VECTOR(768),
    match_count      INT DEFAULT 20,
    rrf_k            INT DEFAULT 60,
    full_text_weight FLOAT DEFAULT 1.0,
    semantic_weight  FLOAT DEFAULT 1.0
)
RETURNS TABLE (
    chunk_id    BIGINT,
    document_id UUID,
    content     TEXT,
    score       FLOAT,
    similarity  FLOAT,
    source_url  TEXT,
    title       TEXT,
    domain      TEXT,
    source_tier SMALLINT,
    metadata    JSONB
)
LANGUAGE sql STABLE AS $$
WITH semantic AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rank,
           1 - (c.embedding <=> query_embedding) AS similarity
    FROM document_chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count * 2
),
keyword AS (
    SELECT c.id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(c.fts, websearch_to_tsquery('english', query_text)) DESC
           ) AS rank
    FROM document_chunks c
    WHERE c.fts @@ websearch_to_tsquery('english', query_text)
    LIMIT match_count * 2
),
fused AS (
    SELECT COALESCE(s.id, k.id) AS id,
           COALESCE(semantic_weight  / (rrf_k + s.rank), 0.0) +
           COALESCE(full_text_weight / (rrf_k + k.rank), 0.0) AS score,
           COALESCE(s.similarity, 0.0) AS similarity
    FROM semantic s
    FULL OUTER JOIN keyword k ON s.id = k.id
)
SELECT c.id, c.document_id, c.content, f.score, f.similarity,
       d.source_url, d.title, d.domain, d.source_tier, c.metadata
FROM fused f
JOIN document_chunks c ON c.id = f.id
JOIN documents d       ON d.id = c.document_id
ORDER BY f.score DESC
LIMIT match_count;
$$;

-- ---------------------------------------------------------------------------
-- kb_stats : lightweight counts for the /kb/stats endpoint
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_stats ()
RETURNS TABLE (documents BIGINT, chunks BIGINT, last_ingest TIMESTAMPTZ)
LANGUAGE sql STABLE AS $$
    SELECT (SELECT count(*) FROM documents),
           (SELECT count(*) FROM document_chunks),
           (SELECT max(retrieved_at) FROM documents);
$$;

-- ===========================================================================
-- Usage analytics + feedback  (query usage_summary / usage_daily in Supabase)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS usage_events (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT,
    user_id       TEXT,
    ip_hash       TEXT,         -- sha256(ip + salt), not the raw IP
    country       TEXT,
    query         TEXT,
    answer_chars  INT,
    n_sources     INT,
    coverage      REAL,
    enriched      BOOLEAN,
    kb_added      INT,
    reasoning     BOOLEAN,
    latency_ms    INT,
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_events (created_at);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_events (session_id);
CREATE INDEX IF NOT EXISTS idx_usage_ip      ON usage_events (ip_hash);

CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT,
    user_id     TEXT,
    query       TEXT,
    rating      SMALLINT,        -- 1 = thumbs up, -1 = thumbs down
    trace_id    TEXT,            -- Langfuse trace id, if available
    comment     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Headline numbers (how many people use the app, etc.)
CREATE OR REPLACE VIEW usage_summary AS
    SELECT count(*)                       AS total_queries,
           count(DISTINCT session_id)     AS total_sessions,
           count(DISTINCT user_id)        AS unique_users,
           count(DISTINCT ip_hash)        AS distinct_ips,
           max(created_at)                AS last_query
    FROM usage_events;

CREATE OR REPLACE VIEW usage_daily AS
    SELECT date_trunc('day', created_at)::date AS day,
           count(*)                            AS queries,
           count(DISTINCT session_id)          AS sessions,
           count(DISTINCT user_id)             AS users,
           round(avg(latency_ms))              AS avg_latency_ms,
           round(avg(coverage)::numeric, 3)    AS avg_coverage,
           sum(kb_added)                       AS docs_added
    FROM usage_events
    GROUP BY 1 ORDER BY 1 DESC;

-- Rate-limit helper: how many requests this IP or session made in a window.
CREATE OR REPLACE FUNCTION recent_request_count(p_ip TEXT, p_session TEXT, p_window_seconds INT)
RETURNS INT LANGUAGE sql STABLE AS $$
    SELECT count(*)::int FROM usage_events
    WHERE created_at > now() - make_interval(secs => p_window_seconds)
      AND (ip_hash = p_ip OR session_id = p_session);
$$;

-- ===========================================================================
-- Email subscribers (optional capture) + training-ready feedback columns
-- ===========================================================================
CREATE TABLE IF NOT EXISTS subscribers (
    id            BIGSERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    wants_updates BOOLEAN DEFAULT FALSE,
    source        TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Store the full exchange with each rating so feedback is usable as training /
-- preference data later.
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS answer  TEXT;
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS sources JSONB;

-- Feature-request board
CREATE TABLE IF NOT EXISTS feature_requests (
    id          BIGSERIAL PRIMARY KEY,
    text        TEXT NOT NULL,
    email       TEXT,
    session_id  TEXT,
    votes       INT DEFAULT 1,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Saved chat history (anonymous, keyed by the browser's persistent user_id)
CREATE TABLE IF NOT EXISTS conversations (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT,
    session_id  TEXT,
    role        TEXT,             -- 'user' or 'assistant'
    content     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conv_user    ON conversations (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations (session_id, created_at);

-- Access tokens: holders bypass the free daily query limit. 3 primary + 7 secondary.
CREATE TABLE IF NOT EXISTS access_tokens (
    id          BIGSERIAL PRIMARY KEY,
    token       TEXT UNIQUE NOT NULL,
    label       TEXT,
    kind        TEXT DEFAULT 'secondary',   -- 'primary' or 'secondary'
    active      BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ===========================================================================
-- Make the functions callable from PostgREST.
--
-- A function in a non-public schema does NOT inherit the caller's search_path,
-- so its unqualified table names (documents, document_chunks) fail to resolve
-- with `relation "documents" does not exist`. Pinning search_path per function
-- fixes that for every caller; `extensions` is included because pgvector and
-- its `<=>` operator live there on Supabase.
-- ===========================================================================
DO $$
DECLARE s TEXT := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION match_documents(vector, int, float) SET search_path = %I, public, extensions', s);
    EXECUTE format(
        'ALTER FUNCTION hybrid_search(text, vector, int, int, float, float) SET search_path = %I, public, extensions', s);
    EXECUTE format(
        'ALTER FUNCTION kb_stats() SET search_path = %I, public, extensions', s);
    EXECUTE format(
        'ALTER FUNCTION recent_request_count(text, text, int) SET search_path = %I, public, extensions', s);
END $$;

-- Supabase roles: let PostgREST read the schema and the service key write it.
-- Skipped automatically on a plain Postgres, where these roles do not exist.
DO $$
DECLARE s TEXT := current_schema();
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        EXECUTE format('GRANT USAGE ON SCHEMA %I TO anon, authenticated, service_role', s);
        EXECUTE format('GRANT ALL ON ALL TABLES IN SCHEMA %I TO service_role', s);
        EXECUTE format('GRANT ALL ON ALL SEQUENCES IN SCHEMA %I TO service_role', s);
        EXECUTE format('GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA %I TO anon, authenticated, service_role', s);
    END IF;
END $$;
