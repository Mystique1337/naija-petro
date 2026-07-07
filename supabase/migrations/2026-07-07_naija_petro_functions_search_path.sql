-- ============================================================================
-- Fix: RAG functions can't resolve their own tables on the shared self-hosted
-- Supabase.
--
-- After migrating the RAG store into the `naija_petro` schema, the functions
-- match_documents / hybrid_search / kb_stats live in `naija_petro` but their
-- bodies reference unqualified table names (`documents`, `document_chunks`).
-- When PostgREST (or any caller) executes them, the search_path does not
-- include `naija_petro`, so they fail with `relation "documents" does not exist`.
--
-- Recreating each with a fixed `SET search_path = naija_petro, public, extensions`
-- pins name resolution to this app's schema regardless of the caller's
-- search_path, so the RPCs work over the REST API (and over a direct
-- connection). `extensions` is included because pgvector (and its `<=>`
-- operator) is installed there on Supabase.
--
-- HOW TO RUN: paste this whole file into the Supabase Studio SQL editor
-- (https://supabase.shinzii.me -> SQL) and run it once. No app downtime.
-- ============================================================================

-- pure semantic (cosine) search --------------------------------------------
CREATE OR REPLACE FUNCTION naija_petro.match_documents (
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
LANGUAGE sql STABLE
SET search_path = naija_petro, public, extensions
AS $$
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

-- hybrid semantic + full-text (Reciprocal Rank Fusion) ----------------------
CREATE OR REPLACE FUNCTION naija_petro.hybrid_search (
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
LANGUAGE sql STABLE
SET search_path = naija_petro, public, extensions
AS $$
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

-- lightweight knowledge-base counts ----------------------------------------
CREATE OR REPLACE FUNCTION naija_petro.kb_stats ()
RETURNS TABLE (documents BIGINT, chunks BIGINT, last_ingest TIMESTAMPTZ)
LANGUAGE sql STABLE
SET search_path = naija_petro, public, extensions
AS $$
    SELECT (SELECT count(*) FROM documents),
           (SELECT count(*) FROM document_chunks),
           (SELECT max(retrieved_at) FROM documents);
$$;

-- The REST roles already reach these functions; ensure EXECUTE is explicit.
GRANT EXECUTE ON FUNCTION naija_petro.match_documents(VECTOR, INT, FLOAT) TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION naija_petro.hybrid_search(TEXT, VECTOR, INT, INT, FLOAT, FLOAT) TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION naija_petro.kb_stats() TO anon, authenticated, service_role;
