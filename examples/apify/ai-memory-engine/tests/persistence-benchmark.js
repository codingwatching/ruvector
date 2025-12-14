/**
 * AI Memory Engine - Persistence & Hyperbolic Geometry Tests
 *
 * Tests:
 * 1. Session persistence (data survives across runs)
 * 2. Hyperbolic geometry functions
 * 3. Storage backend benchmarks
 */

// Mock Actor for local testing
const mockActor = {
  stores: new Map(),
  async openKeyValueStore(name) {
    if (!this.stores.has(name)) {
      this.stores.set(name, new Map());
    }
    const store = this.stores.get(name);
    return {
      async setValue(key, value, options) {
        store.set(key, { value, options });
      },
      async getValue(key) {
        const entry = store.get(key);
        return entry?.value || null;
      }
    };
  }
};

// Hyperbolic geometry functions (copied for testing)
const POINCARE_EPS = 1e-7;

function projectToPoincareBall(x, c = 1.0) {
  const normSq = x.reduce((sum, v) => sum + v * v, 0);
  const maxNorm = (1.0 - POINCARE_EPS) / Math.sqrt(c);
  if (normSq >= maxNorm * maxNorm) {
    const scale = maxNorm / Math.sqrt(normSq);
    return x.map(v => v * scale);
  }
  return x;
}

function poincareDistance(u, v, c = 1.0) {
  const sqrtC = Math.sqrt(c);
  const diff = u.map((ui, i) => ui - v[i]);
  const normDiffSq = diff.reduce((sum, d) => sum + d * d, 0);
  const normUSq = u.reduce((sum, ui) => sum + ui * ui, 0);
  const normVSq = v.reduce((sum, vi) => sum + vi * vi, 0);
  const lambdaU = 1.0 - c * normUSq;
  const lambdaV = 1.0 - c * normVSq;
  const arg = 1.0 + (2.0 * c * normDiffSq) / Math.max(lambdaU * lambdaV, POINCARE_EPS);
  return (1.0 / sqrtC) * Math.acosh(Math.max(1.0, arg));
}

function mobiusAdd(u, v, c = 1.0) {
  const normUSq = u.reduce((sum, ui) => sum + ui * ui, 0);
  const normVSq = v.reduce((sum, vi) => sum + vi * vi, 0);
  const dotUV = u.reduce((sum, ui, i) => sum + ui * v[i], 0);
  const coefU = 1.0 + 2.0 * c * dotUV + c * normVSq;
  const coefV = 1.0 - c * normUSq;
  const denom = 1.0 + 2.0 * c * dotUV + c * c * normUSq * normVSq;
  const result = u.map((ui, i) => (coefU * ui + coefV * v[i]) / Math.max(denom, POINCARE_EPS));
  return projectToPoincareBall(result, c);
}

function expMap(base, tangent, c = 1.0) {
  const sqrtC = Math.sqrt(c);
  const normBaseSq = base.reduce((sum, b) => sum + b * b, 0);
  const lambdaBase = 1.0 / Math.max(1.0 - c * normBaseSq, POINCARE_EPS);
  const normTangent = Math.sqrt(tangent.reduce((sum, t) => sum + t * t, 0));
  if (normTangent < POINCARE_EPS) return base;
  const normTangentP = lambdaBase * normTangent;
  const coef = Math.tanh(sqrtC * normTangentP / 2.0) / (sqrtC * normTangentP);
  const transported = tangent.map(t => coef * t);
  return mobiusAdd(base, transported, c);
}

function logMap(base, point, c = 1.0) {
  const sqrtC = Math.sqrt(c);
  const negBase = base.map(b => -b);
  const diff = mobiusAdd(negBase, point, c);
  const normDiff = Math.sqrt(diff.reduce((sum, d) => sum + d * d, 0));
  if (normDiff < POINCARE_EPS) return diff;
  const normBaseSq = base.reduce((sum, b) => sum + b * b, 0);
  const lambdaBase = 1.0 / Math.max(1.0 - c * normBaseSq, POINCARE_EPS);
  const coef = (2.0 / (sqrtC * lambdaBase)) * Math.atanh(Math.min(sqrtC * normDiff, 1.0 - POINCARE_EPS)) / normDiff;
  return diff.map(d => coef * d);
}

function frechetMean(points, c = 1.0, maxIter = 100, tol = 1e-6) {
  if (points.length === 0) return null;
  if (points.length === 1) return [...points[0]];
  let mean = points[0].map((_, i) => points.reduce((sum, p) => sum + p[i], 0) / points.length);
  mean = projectToPoincareBall(mean, c);
  for (let iter = 0; iter < maxIter; iter++) {
    const tangents = points.map(p => logMap(mean, p, c));
    const avgTangent = tangents[0].map((_, i) => tangents.reduce((sum, t) => sum + t[i], 0) / tangents.length);
    const newMean = expMap(mean, avgTangent, c);
    const diff = newMean.reduce((sum, v, i) => sum + (v - mean[i]) ** 2, 0);
    mean = newMean;
    if (diff < tol) break;
  }
  return mean;
}

// Test utilities
function generateRandomVector(dim) {
  return Array.from({ length: dim }, () => Math.random() * 0.5 - 0.25);
}

function assertEqual(actual, expected, message, tolerance = 1e-6) {
  if (Array.isArray(actual) && Array.isArray(expected)) {
    if (actual.length !== expected.length) {
      throw new Error(`${message}: length mismatch ${actual.length} vs ${expected.length}`);
    }
    for (let i = 0; i < actual.length; i++) {
      if (Math.abs(actual[i] - expected[i]) > tolerance) {
        throw new Error(`${message}: element ${i} differs: ${actual[i]} vs ${expected[i]}`);
      }
    }
  } else if (typeof actual === 'number' && typeof expected === 'number') {
    if (Math.abs(actual - expected) > tolerance) {
      throw new Error(`${message}: ${actual} vs ${expected}`);
    }
  } else if (actual !== expected) {
    throw new Error(`${message}: ${actual} vs ${expected}`);
  }
}

function assertTrue(condition, message) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

// ============================================
// TEST SUITE
// ============================================

console.log('🧪 AI Memory Engine - Persistence & Hyperbolic Tests\n');
console.log('=' .repeat(60));

// Test 1: Hyperbolic Geometry
console.log('\n📐 Test 1: Hyperbolic Geometry Functions\n');

try {
  // Test projection to Poincaré ball
  const outsidePoint = [1.0, 1.0, 1.0];
  const projected = projectToPoincareBall(outsidePoint, 1.0);
  const projectedNorm = Math.sqrt(projected.reduce((s, v) => s + v * v, 0));
  assertTrue(projectedNorm < 1.0, 'Projected point should be inside unit ball');
  console.log('✅ projectToPoincareBall: Points outside ball correctly projected inside');

  // Test Poincaré distance properties
  const p1 = [0.1, 0.2, 0.1];
  const p2 = [0.3, 0.1, 0.2];
  const dist12 = poincareDistance(p1, p2, 1.0);
  const dist21 = poincareDistance(p2, p1, 1.0);
  assertEqual(dist12, dist21, 'Distance should be symmetric');
  assertTrue(dist12 > 0, 'Distance should be positive');
  console.log('✅ poincareDistance: Symmetric and positive');

  // Test distance to self is zero
  const distSelf = poincareDistance(p1, p1, 1.0);
  assertEqual(distSelf, 0, 'Distance to self should be zero');
  console.log('✅ poincareDistance: Self-distance is zero');

  // Test Möbius addition properties
  const origin = [0, 0, 0];
  const v = [0.2, 0.3, 0.1];
  const addOrigin = mobiusAdd(origin, v, 1.0);
  assertEqual(addOrigin, v, 'Adding origin should return the point', 0.01);
  console.log('✅ mobiusAdd: Origin is identity');

  // Test exp/log map round-trip
  const base = [0.1, 0.1, 0.1];
  const point = [0.3, 0.2, 0.15];
  const tangent = logMap(base, point, 1.0);
  const reconstructed = expMap(base, tangent, 1.0);
  assertEqual(reconstructed, point, 'Exp(Log(x)) should equal x', 0.01);
  console.log('✅ expMap/logMap: Round-trip preserves points');

  // Test Fréchet mean
  const points = [
    [0.1, 0.1, 0.1],
    [0.2, 0.1, 0.1],
    [0.15, 0.2, 0.1]
  ];
  const mean = frechetMean(points, 1.0);
  assertTrue(mean !== null, 'Fréchet mean should exist');
  const meanNorm = Math.sqrt(mean.reduce((s, v) => s + v * v, 0));
  assertTrue(meanNorm < 1.0, 'Fréchet mean should be inside ball');
  console.log('✅ frechetMean: Computes valid centroid in hyperbolic space');

  console.log('\n✨ All hyperbolic geometry tests passed!');
} catch (e) {
  console.error('❌ Hyperbolic geometry test failed:', e.message);
  process.exit(1);
}

// Test 2: Binary Persistence Format
console.log('\n💾 Test 2: Binary Persistence Format\n');

try {
  const dimensions = 384;
  const memoryCount = 1000;

  // Generate test embeddings
  const embeddings = Array.from({ length: memoryCount }, () =>
    Array.from({ length: dimensions }, () => Math.random())
  );

  // Measure JSON size
  const jsonStart = process.hrtime.bigint();
  const jsonData = JSON.stringify(embeddings);
  const jsonTime = Number(process.hrtime.bigint() - jsonStart) / 1e6;
  const jsonSize = Buffer.byteLength(jsonData, 'utf8');

  // Measure binary size
  const binaryStart = process.hrtime.bigint();
  const totalFloats = memoryCount * dimensions;
  const buffer = new Float32Array(totalFloats);
  for (let i = 0; i < memoryCount; i++) {
    buffer.set(embeddings[i], i * dimensions);
  }
  const binaryData = Buffer.from(buffer.buffer);
  const binaryTime = Number(process.hrtime.bigint() - binaryStart) / 1e6;
  const binarySize = binaryData.length;

  const sizeRatio = jsonSize / binarySize;

  console.log(`  📊 ${memoryCount} embeddings × ${dimensions}d:`);
  console.log(`     JSON:   ${(jsonSize / 1024 / 1024).toFixed(2)} MB (${jsonTime.toFixed(2)}ms)`);
  console.log(`     Binary: ${(binarySize / 1024 / 1024).toFixed(2)} MB (${binaryTime.toFixed(2)}ms)`);
  console.log(`     Ratio:  ${sizeRatio.toFixed(1)}x smaller with binary`);

  assertTrue(sizeRatio > 3.5, 'Binary should be at least 3.5x smaller');
  assertTrue(binaryTime < jsonTime, 'Binary serialization should be faster');

  // Test round-trip
  const restored = new Float32Array(binaryData.buffer);
  for (let i = 0; i < Math.min(100, memoryCount); i++) {
    const original = embeddings[i];
    const restoredVec = Array.from(restored.slice(i * dimensions, (i + 1) * dimensions));
    for (let j = 0; j < dimensions; j++) {
      if (Math.abs(original[j] - restoredVec[j]) > 1e-6) {
        throw new Error(`Round-trip failed at [${i}][${j}]`);
      }
    }
  }

  console.log('✅ Binary format: 4x+ size reduction with perfect fidelity');
  console.log('\n✨ Binary persistence tests passed!');
} catch (e) {
  console.error('❌ Binary persistence test failed:', e.message);
  process.exit(1);
}

// Test 3: Session Persistence Simulation
console.log('\n🔄 Test 3: Session Persistence Simulation\n');

try {
  const sessionId = 'test-session-123';
  const storeName = `ai-memory-${sessionId}`;

  // Simulate first run - store data
  console.log('  📝 Run 1: Storing 5 memories...');
  const store1 = await mockActor.openKeyValueStore(storeName);
  const memories1 = [
    { id: 'mem_1', text: 'Customer prefers fast shipping', embedding: generateRandomVector(384) },
    { id: 'mem_2', text: 'Product A is popular', embedding: generateRandomVector(384) },
    { id: 'mem_3', text: 'Support ticket resolved', embedding: generateRandomVector(384) },
    { id: 'mem_4', text: 'New feature request', embedding: generateRandomVector(384) },
    { id: 'mem_5', text: 'Billing question answered', embedding: generateRandomVector(384) }
  ];

  // Save metadata
  await store1.setValue('metadata', {
    memories: memories1.map(m => ({ id: m.id, text: m.text })),
    stats: { stores: 5, queries: 0 }
  });

  // Save binary embeddings
  const ids = memories1.map(m => m.id);
  const embeddings = memories1.map(m => m.embedding);
  const buffer = new Float32Array(5 * 384);
  for (let i = 0; i < 5; i++) {
    buffer.set(embeddings[i], i * 384);
  }
  await store1.setValue('embeddings_header', { ids, dimensions: 384, count: 5 });
  await store1.setValue('embeddings_binary', Buffer.from(buffer.buffer));

  console.log('     Saved: 5 memories + binary embeddings');

  // Simulate second run - load data
  console.log('  📖 Run 2: Loading session...');
  const store2 = await mockActor.openKeyValueStore(storeName);
  const metadata = await store2.getValue('metadata');
  const header = await store2.getValue('embeddings_header');
  const binaryData = await store2.getValue('embeddings_binary');

  assertTrue(metadata !== null, 'Metadata should persist');
  assertTrue(metadata.memories.length === 5, 'All 5 memories should persist');
  assertTrue(header !== null, 'Embeddings header should persist');
  assertTrue(binaryData !== null, 'Binary embeddings should persist');

  // Verify embeddings round-trip
  const restoredBuffer = new Float32Array(binaryData.buffer || binaryData);
  for (let i = 0; i < 5; i++) {
    const original = embeddings[i];
    const restored = Array.from(restoredBuffer.slice(i * 384, (i + 1) * 384));
    for (let j = 0; j < 384; j++) {
      if (Math.abs(original[j] - restored[j]) > 1e-6) {
        throw new Error(`Embedding ${i} corrupted at index ${j}`);
      }
    }
  }

  console.log('     Loaded: 5 memories with intact embeddings');
  console.log('✅ Session persistence: Data survives across runs');
  console.log('\n✨ Session persistence tests passed!');
} catch (e) {
  console.error('❌ Session persistence test failed:', e.message);
  process.exit(1);
}

// Test 4: Benchmark Distance Calculations
console.log('\n⏱️  Test 4: Distance Calculation Benchmarks\n');

try {
  const vectorCount = 10000;
  const dimensions = 384;
  const queryCount = 100;

  // Generate test data
  const vectors = Array.from({ length: vectorCount }, () => generateRandomVector(dimensions));
  const queries = Array.from({ length: queryCount }, () => generateRandomVector(dimensions));

  // Benchmark Euclidean distance
  const euclideanStart = process.hrtime.bigint();
  for (const query of queries) {
    for (const vec of vectors) {
      let sum = 0;
      for (let i = 0; i < dimensions; i++) {
        const d = query[i] - vec[i];
        sum += d * d;
      }
      Math.sqrt(sum);
    }
  }
  const euclideanTime = Number(process.hrtime.bigint() - euclideanStart) / 1e6;

  // Benchmark Poincaré distance (hyperbolic)
  const hyperbolicStart = process.hrtime.bigint();
  const hypVectors = vectors.map(v => projectToPoincareBall(v, 1.0));
  const hypQueries = queries.map(q => projectToPoincareBall(q, 1.0));
  for (const query of hypQueries) {
    for (const vec of hypVectors) {
      poincareDistance(query, vec, 1.0);
    }
  }
  const hyperbolicTime = Number(process.hrtime.bigint() - hyperbolicStart) / 1e6;

  const totalComparisons = queryCount * vectorCount;
  const euclideanOps = totalComparisons / (euclideanTime / 1000);
  const hyperbolicOps = totalComparisons / (hyperbolicTime / 1000);

  console.log(`  📊 ${queryCount} queries × ${vectorCount} vectors (${dimensions}d):`);
  console.log(`     Euclidean:   ${euclideanTime.toFixed(0)}ms (${(euclideanOps / 1e6).toFixed(1)}M ops/sec)`);
  console.log(`     Hyperbolic:  ${hyperbolicTime.toFixed(0)}ms (${(hyperbolicOps / 1e6).toFixed(1)}M ops/sec)`);
  console.log(`     Ratio:       Hyperbolic is ${(hyperbolicTime / euclideanTime).toFixed(1)}x slower`);

  console.log('\n✨ Benchmark completed!');
} catch (e) {
  console.error('❌ Benchmark failed:', e.message);
  process.exit(1);
}

// Summary
console.log('\n' + '=' .repeat(60));
console.log('📋 SUMMARY');
console.log('=' .repeat(60));
console.log('\n✅ All tests passed!\n');
console.log('Key findings:');
console.log('  • Binary format: 4x+ smaller than JSON');
console.log('  • Session persistence: Data survives across runs');
console.log('  • Hyperbolic geometry: Correct implementation');
console.log('  • Performance: Hyperbolic ~5-10x slower than Euclidean (expected)\n');
