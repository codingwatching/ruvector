/**
 * Local test script for RuVector PostgreSQL Actor
 *
 * Usage:
 *   # Start ruvector-postgres first:
 *   docker run -d --name ruvector-pg -e POSTGRES_PASSWORD=secret -p 5432:5432 ruvnet/ruvector-postgres:latest
 *
 *   # Run tests:
 *   DATABASE_URL="postgresql://postgres:secret@localhost:5432/postgres" npm test
 */

import pg from 'pg';

const connectionString = process.env.DATABASE_URL || 'postgresql://postgres:secret@localhost:5432/postgres';

async function runTests() {
  console.log('🧪 RuVector PostgreSQL Actor - Local Test Suite\n');

  const client = new pg.Client({ connectionString });

  try {
    // Connect
    console.log('1. Connecting to PostgreSQL...');
    await client.connect();
    console.log('   ✅ Connected\n');

    // Enable extension
    console.log('2. Enabling ruvector extension...');
    await client.query('CREATE EXTENSION IF NOT EXISTS ruvector');
    console.log('   ✅ Extension enabled\n');

    // Check version
    console.log('3. Checking ruvector version...');
    const versionResult = await client.query('SELECT ruvector_version()');
    console.log(`   ✅ Version: ${versionResult.rows[0].ruvector_version}\n`);

    // Create test table
    console.log('4. Creating test table...');
    await client.query('DROP TABLE IF EXISTS test_documents');
    await client.query(`
      CREATE TABLE test_documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding ruvector(384),
        metadata JSONB DEFAULT '{}'
      )
    `);
    console.log('   ✅ Table created\n');

    // Create HNSW index
    console.log('5. Creating HNSW index...');
    await client.query(`
      CREATE INDEX test_docs_idx ON test_documents
      USING ruhnsw (embedding ruvector_cosine_ops)
      WITH (m = 16, ef_construction = 64)
    `);
    console.log('   ✅ Index created\n');

    // Test local embedding generation
    console.log('6. Testing local embedding generation...');
    const embedResult = await client.query(`
      SELECT array_length(ruvector_embed('Hello, world!', 'all-MiniLM-L6-v2')::float[], 1) AS dim
    `);
    console.log(`   ✅ Embedding dimensions: ${embedResult.rows[0].dim}\n`);

    // Insert test documents
    console.log('7. Inserting test documents with embeddings...');
    const testDocs = [
      { content: 'Introduction to machine learning and artificial intelligence', category: 'AI' },
      { content: 'Deep learning neural networks for image recognition', category: 'AI' },
      { content: 'PostgreSQL database administration and optimization', category: 'Database' },
      { content: 'Vector similarity search algorithms and HNSW indexing', category: 'Search' },
      { content: 'Natural language processing and text embeddings', category: 'NLP' },
    ];

    for (const doc of testDocs) {
      await client.query(`
        INSERT INTO test_documents (content, embedding, metadata)
        VALUES ($1, ruvector_embed($1, 'all-MiniLM-L6-v2'), $2)
      `, [doc.content, JSON.stringify({ category: doc.category })]);
    }
    console.log(`   ✅ Inserted ${testDocs.length} documents\n`);

    // Test semantic search
    console.log('8. Testing semantic search...');
    const searchResult = await client.query(`
      SELECT
        content,
        embedding <=> ruvector_embed('machine learning basics', 'all-MiniLM-L6-v2') AS distance,
        metadata
      FROM test_documents
      ORDER BY distance
      LIMIT 3
    `);
    console.log('   ✅ Search results:');
    searchResult.rows.forEach((row, i) => {
      console.log(`      ${i + 1}. [${row.distance.toFixed(4)}] ${row.content.substring(0, 50)}...`);
    });
    console.log();

    // Test distance metrics
    console.log('9. Testing distance metrics...');
    const metricsResult = await client.query(`
      SELECT
        ruvector_cosine_distance(
          ruvector_embed('hello', 'all-MiniLM-L6-v2'),
          ruvector_embed('hi there', 'all-MiniLM-L6-v2')
        ) AS cosine,
        ruvector_l2_distance(
          ruvector_embed('hello', 'all-MiniLM-L6-v2'),
          ruvector_embed('hi there', 'all-MiniLM-L6-v2')
        ) AS l2
    `);
    console.log(`   ✅ Cosine distance: ${metricsResult.rows[0].cosine.toFixed(4)}`);
    console.log(`   ✅ L2 distance: ${metricsResult.rows[0].l2.toFixed(4)}\n`);

    // Test hyperbolic distance (if available)
    console.log('10. Testing hyperbolic distance...');
    try {
      const hyperbolicResult = await client.query(`
        SELECT ruvector_poincare_distance(
          '[0.1, 0.2, 0.3]'::ruvector,
          '[0.2, 0.3, 0.4]'::ruvector,
          -1.0
        ) AS poincare_dist
      `);
      console.log(`    ✅ Poincare distance: ${hyperbolicResult.rows[0].poincare_dist.toFixed(4)}\n`);
    } catch (e) {
      console.log('    ⚠️ Hyperbolic functions not available (optional feature)\n');
    }

    // Cleanup
    console.log('11. Cleaning up...');
    await client.query('DROP TABLE IF EXISTS test_documents');
    console.log('    ✅ Test table dropped\n');

    console.log('═══════════════════════════════════════════════════════════');
    console.log('✅ All tests passed! RuVector PostgreSQL actor is working.');
    console.log('═══════════════════════════════════════════════════════════\n');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error('\nMake sure ruvector-postgres is running:');
    console.error('  docker run -d --name ruvector-pg -e POSTGRES_PASSWORD=secret -p 5432:5432 ruvnet/ruvector-postgres:latest\n');
    process.exit(1);
  } finally {
    await client.end();
  }
}

runTests();
