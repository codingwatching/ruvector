interface Props {
  stats: {
    vectors: number
    triples: number
    initialized: boolean
  }
}

export default function Overview({ stats }: Props) {
  return (
    <div>
      <div className="page-header">
        <h2>Dashboard Overview</h2>
        <p>RvLite vector database with SQL, SPARQL, and Cypher support</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.vectors}</div>
          <div className="stat-label">Vectors Stored</div>
        </div>

        <div className="stat-card">
          <div className="stat-value">{stats.triples}</div>
          <div className="stat-label">RDF Triples</div>
        </div>

        <div className="stat-card">
          <div className="stat-value">384</div>
          <div className="stat-label">Dimensions</div>
        </div>

        <div className="stat-card">
          <div className="stat-value">
            {stats.initialized ? (
              <span className="badge badge-success">Active</span>
            ) : (
              <span className="badge badge-warning">Loading</span>
            )}
          </div>
          <div className="stat-label">Database Status</div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">Quick Start</div>
        <div style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
          <p style={{ marginBottom: 12 }}>
            <strong>SQL Queries:</strong> Use familiar SQL syntax for vector search
          </p>
          <pre className="results-area" style={{ marginBottom: 16 }}>
{`-- Insert a vector
INSERT INTO vectors (id, embedding) VALUES ('doc1', [0.1, 0.2, ...])

-- Search similar vectors
SELECT * FROM vectors ORDER BY distance(embedding, ?) LIMIT 10`}
          </pre>

          <p style={{ marginBottom: 12 }}>
            <strong>SPARQL Queries:</strong> Query RDF knowledge graphs
          </p>
          <pre className="results-area" style={{ marginBottom: 16 }}>
{`SELECT ?subject ?predicate ?object
WHERE { ?subject ?predicate ?object }
LIMIT 10`}
          </pre>

          <p style={{ marginBottom: 12 }}>
            <strong>Cypher Queries:</strong> Property graph queries
          </p>
          <pre className="results-area">
{`MATCH (n:Person)-[:KNOWS]->(m)
RETURN n.name, m.name`}
          </pre>
        </div>
      </div>

      <div className="card">
        <div className="card-header">Features</div>
        <div className="grid-2">
          <div>
            <h4 style={{ marginBottom: 8, fontSize: 14 }}>Vector Database</h4>
            <ul style={{ color: 'var(--text-secondary)', fontSize: 13, paddingLeft: 20 }}>
              <li>HNSW similarity search</li>
              <li>Cosine, L2, Dot product metrics</li>
              <li>IndexedDB persistence</li>
              <li>Multi-tenant support</li>
            </ul>
          </div>
          <div>
            <h4 style={{ marginBottom: 8, fontSize: 14 }}>Query Languages</h4>
            <ul style={{ color: 'var(--text-secondary)', fontSize: 13, paddingLeft: 20 }}>
              <li>SQL for vectors</li>
              <li>SPARQL for RDF triples</li>
              <li>Cypher for property graphs</li>
              <li>Native Rust + WASM</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
