-- Placeholder migration establishing pgvector extension for the mastery demo.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS demo_chunks (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    section TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(128)
);
