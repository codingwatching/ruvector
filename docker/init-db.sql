-- RuVector Database Initialization
-- This script runs when the PostgreSQL container first starts

-- Enable pgvector extension (if available)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a sample schema for RuVector features
CREATE SCHEMA IF NOT EXISTS ruvector;

-- Sample table for vector storage
CREATE TABLE IF NOT EXISTS ruvector.embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx ON ruvector.embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Sample agent registry table for Tiny Dancer routing
CREATE TABLE IF NOT EXISTS ruvector.agents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    capabilities vector(384),
    metadata JSONB DEFAULT '{}',
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for agent capability matching
CREATE INDEX IF NOT EXISTS agents_capabilities_idx ON ruvector.agents
USING hnsw (capabilities vector_cosine_ops);

-- Sample table for ReasoningBank trajectories
CREATE TABLE IF NOT EXISTS ruvector.trajectories (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    state JSONB,
    action TEXT,
    outcome JSONB,
    success BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sample graph tables for GNN
CREATE TABLE IF NOT EXISTS ruvector.nodes (
    id SERIAL PRIMARY KEY,
    name TEXT,
    features vector(128),
    node_type TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ruvector.edges (
    id SERIAL PRIMARY KEY,
    source_id INT REFERENCES ruvector.nodes(id),
    target_id INT REFERENCES ruvector.nodes(id),
    edge_type TEXT,
    weight FLOAT DEFAULT 1.0
);

-- Grant permissions
GRANT ALL ON SCHEMA ruvector TO ruvector;
GRANT ALL ON ALL TABLES IN SCHEMA ruvector TO ruvector;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ruvector TO ruvector;

-- Insert sample data
INSERT INTO ruvector.agents (name, description) VALUES
    ('code-agent', 'Expert at writing TypeScript and Python code'),
    ('docs-agent', 'Specializes in documentation and technical writing'),
    ('test-agent', 'Focused on testing and quality assurance')
ON CONFLICT (name) DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'RuVector database initialized successfully!';
END $$;
