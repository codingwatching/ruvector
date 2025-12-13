import { useState } from 'react'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RvLiteDb = any

interface Props {
  db: RvLiteDb | null
  onUpdate: () => void
}

const EXAMPLE_QUERIES = {
  sql: `-- Show database stats
SELECT 'vectors' as type, COUNT(*) as count FROM vectors
UNION ALL
SELECT 'triples', COUNT(*) FROM triples`,
  sparql: `SELECT ?s ?p ?o
WHERE {
  ?s ?p ?o
}
LIMIT 10`,
  cypher: `MATCH (n)
RETURN n
LIMIT 10`
}

export default function Vectors({ db, onUpdate }: Props) {
  const [query, setQuery] = useState(EXAMPLE_QUERIES.sql)
  const [queryType, setQueryType] = useState<'sql' | 'sparql' | 'cypher'>('sql')
  const [result, setResult] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [executing, setExecuting] = useState(false)

  function executeQuery() {
    if (!db || !query.trim()) return

    setExecuting(true)
    setError('')

    try {
      let queryResult: unknown

      const upperQuery = query.trim().toUpperCase()

      if (queryType === 'sparql' || upperQuery.startsWith('SELECT ?')) {
        queryResult = db.sparql(query)
      } else if (queryType === 'cypher' || upperQuery.startsWith('MATCH ')) {
        queryResult = db.cypher(query)
      } else {
        queryResult = db.sql(query)
      }

      setResult(JSON.stringify(queryResult, null, 2))
      onUpdate()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setResult('')
    } finally {
      setExecuting(false)
    }
  }

  function loadExample(type: 'sql' | 'sparql' | 'cypher') {
    setQueryType(type)
    setQuery(EXAMPLE_QUERIES[type])
    setResult('')
    setError('')
  }

  async function addSampleData() {
    if (!db) return

    setExecuting(true)
    try {
      // Add sample vectors
      const sampleVectors = [
        { id: 'doc1', text: 'Machine learning basics' },
        { id: 'doc2', text: 'Neural network architectures' },
        { id: 'doc3', text: 'Natural language processing' },
      ]

      for (const item of sampleVectors) {
        // Generate random 384-dim embedding for demo
        const embedding = Array.from({ length: 384 }, () => Math.random() * 2 - 1)
        db.insert(item.id, embedding, { text: item.text })
      }

      // Add sample triples
      db.add_triple('doc1', 'hasType', 'Tutorial')
      db.add_triple('doc2', 'hasType', 'Reference')
      db.add_triple('doc3', 'hasType', 'Guide')
      db.add_triple('doc1', 'relatedTo', 'doc2')

      onUpdate()
      setResult('Added 3 sample vectors and 4 triples')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setExecuting(false)
    }
  }

  return (
    <div>
      <div className="page-header">
        <h2>Vectors & Query</h2>
        <p>Execute SQL, SPARQL, or Cypher queries</p>
      </div>

      <div className="card">
        <div className="card-header flex justify-between items-center">
          <span>Query Editor</span>
          <div className="flex gap-2">
            <button
              className={`btn ${queryType === 'sql' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => loadExample('sql')}
            >
              SQL
            </button>
            <button
              className={`btn ${queryType === 'sparql' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => loadExample('sparql')}
            >
              SPARQL
            </button>
            <button
              className={`btn ${queryType === 'cypher' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => loadExample('cypher')}
            >
              Cypher
            </button>
          </div>
        </div>

        <textarea
          className="query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query..."
          rows={6}
        />

        <div className="flex gap-2 mt-4">
          <button
            className="btn btn-primary"
            onClick={executeQuery}
            disabled={executing || !query.trim()}
          >
            {executing ? 'Executing...' : 'Execute Query'}
          </button>
          <button className="btn btn-secondary" onClick={addSampleData} disabled={executing}>
            Add Sample Data
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: 'var(--error)' }}>
          <div className="card-header" style={{ color: 'var(--error)' }}>
            Error
          </div>
          <pre className="results-area" style={{ color: 'var(--error)' }}>
            {error}
          </pre>
        </div>
      )}

      {result && (
        <div className="card">
          <div className="card-header">Results</div>
          <pre className="results-area">{result}</pre>
        </div>
      )}
    </div>
  )
}
