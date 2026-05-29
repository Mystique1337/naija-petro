-- ============================================================================
-- Naija-Petro RAG — vector store schema (Postgres + pgvector)
--
-- Works on a self-hosted Supabase OR a plain Railway Postgres. Run once:
--     psql "$SUPABASE_DB_URL" -f supabase/schema.sql
--
-- Embedding dimension is 768 (nomic-embed-text-v1.5). If you change EMBED_MODEL
-- to a different dimension, update every `vector(768)` below and re-run.
-- ============================================================================

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
