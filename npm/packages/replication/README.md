# @ruvector/replication

[![npm version](https://img.shields.io/npm/v/@ruvector/replication.svg)](https://www.npmjs.com/package/@ruvector/replication)
[![npm downloads](https://img.shields.io/npm/dm/@ruvector/replication.svg)](https://www.npmjs.com/package/@ruvector/replication)
[![License](https://img.shields.io/npm/l/@ruvector/replication.svg)](https://github.com/ruvnet/ruvector/blob/main/LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

**Data replication and synchronization for distributed TypeScript applications.**

Multi-node replica management with vector clocks, conflict resolution, and automatic failover.

## Features

- 🔄 **Multi-Node Replication** - Primary/secondary replica sets with automatic promotion
- ⏱️ **Vector Clocks** - Causality tracking for conflict detection
- 🤝 **Conflict Resolution** - Last-write-wins, custom merge functions, or manual resolution
- 📊 **Sync Modes** - Synchronous, asynchronous, or semi-synchronous replication
- 🔀 **Automatic Failover** - Configurable failover policies with health monitoring
- 📡 **Change Data Capture** - Stream changes for real-time updates

## Installation

```bash
npm install @ruvector/replication
```

## Quick Start

```typescript
import {
  ReplicaSet,
  ReplicaRole,
  SyncManager,
  ReplicationLog,
  SyncMode,
  ChangeOperation,
} from '@ruvector/replication';

// Create a replica set
const replicaSet = new ReplicaSet('my-cluster', {
  minQuorum: 2,
  heartbeatInterval: 1000,
  healthCheckTimeout: 5000,
});

// Add replicas
replicaSet.addReplica('node-1', '192.168.1.10:9001', ReplicaRole.Primary);
replicaSet.addReplica('node-2', '192.168.1.11:9001', ReplicaRole.Secondary);
replicaSet.addReplica('node-3', '192.168.1.12:9001', ReplicaRole.Secondary);

// Create replication log and sync manager
const log = new ReplicationLog('node-1');
const syncManager = new SyncManager(replicaSet, log, {
  mode: SyncMode.SemiSync,
  minReplicas: 1,
  batchSize: 100,
});

// Listen for events
syncManager.on('changeReceived', (change) => {
  console.log(`Change: ${change.operation} on ${change.key}`);
});

syncManager.on('conflictDetected', ({ local, remote }) => {
  console.log('Conflict detected!', { local, remote });
});

// Record a change
await syncManager.recordChange(
  'user:123',
  ChangeOperation.Update,
  { name: 'Alice', email: 'alice@example.com' }
);
```

## Tutorials

### Setting Up a Replica Set

```typescript
import { ReplicaSet, ReplicaRole, ReplicationEvent, FailoverPolicy } from '@ruvector/replication';

// Create replica set with automatic failover
const replicaSet = new ReplicaSet('production-cluster', {
  minQuorum: 2,
  heartbeatInterval: 1000,
  healthCheckTimeout: 5000,
  failoverPolicy: FailoverPolicy.Automatic,
});

// Add replicas
replicaSet.addReplica('primary', 'db1.example.com:5432', ReplicaRole.Primary);
replicaSet.addReplica('secondary-1', 'db2.example.com:5432', ReplicaRole.Secondary);
replicaSet.addReplica('secondary-2', 'db3.example.com:5432', ReplicaRole.Secondary);

// Monitor replica events
replicaSet.on(ReplicationEvent.ReplicaStatusChanged, ({ replica, newStatus }) => {
  console.log(`${replica.id} status: ${newStatus}`);
});

replicaSet.on(ReplicationEvent.PrimaryChanged, ({ previousPrimary, newPrimary }) => {
  console.log(`Primary changed: ${previousPrimary} → ${newPrimary}`);
});

replicaSet.on(ReplicationEvent.FailoverStarted, () => {
  console.log('Failover initiated...');
});

// Start health monitoring
replicaSet.startHeartbeat();

// Check cluster status
console.log('Stats:', replicaSet.getStats());
// { total: 3, active: 3, syncing: 0, offline: 0, failed: 0, hasQuorum: true }
```

### Using Vector Clocks for Conflict Detection

```typescript
import { VectorClock, VectorClockComparison } from '@ruvector/replication';

// Create clocks for two replicas
const clockA = new VectorClock();
const clockB = new VectorClock();

// Replica A performs operations
clockA.increment('replica-a');
clockA.increment('replica-a');
console.log(clockA.toJSON()); // { 'replica-a': 2 }

// Replica B performs operations
clockB.increment('replica-b');
console.log(clockB.toJSON()); // { 'replica-b': 1 }

// Check causal relationship
const comparison = clockA.compare(clockB);
console.log(comparison); // 'concurrent' - no causal relationship

// Merge clocks after sync
clockA.merge(clockB);
console.log(clockA.toJSON()); // { 'replica-a': 2, 'replica-b': 1 }

// Now clockA happens after the original clockB
console.log(clockA.happensAfter(clockB)); // true
```

### Custom Conflict Resolution

```typescript
import { SyncManager, MergeFunction, VectorClock } from '@ruvector/replication';

// Define your data type
interface UserProfile {
  name: string;
  email: string;
  updatedAt: number;
  version: number;
}

// Custom merge function - merge fields intelligently
const userMerger = new MergeFunction<UserProfile>((local, remote) => {
  // Take the most recent update for each field
  return {
    name: local.updatedAt > remote.updatedAt ? local.name : remote.name,
    email: local.updatedAt > remote.updatedAt ? local.email : remote.email,
    updatedAt: Math.max(local.updatedAt, remote.updatedAt),
    version: Math.max(local.version, remote.version) + 1,
  };
});

// Apply to sync manager
syncManager.setConflictResolver(userMerger);

// Or handle conflicts manually
syncManager.on('conflictDetected', ({ local, remote }) => {
  // Custom logic
  const resolved = customResolve(local, remote);
  // Apply resolved value...
});
```

### Implementing Change Data Capture (CDC)

```typescript
import { SyncManager, ReplicationEvent, ChangeOperation } from '@ruvector/replication';

// Set up CDC listeners
syncManager.on(ReplicationEvent.ChangeReceived, (change) => {
  // Stream to message queue
  kafka.produce('changes', {
    id: change.id,
    operation: change.operation,
    key: change.key,
    value: change.value,
    timestamp: change.timestamp,
    source: change.sourceReplica,
  });
});

// Track changes in your application
async function updateUser(userId: string, data: UserData) {
  const previous = await db.get(`user:${userId}`);
  await db.set(`user:${userId}`, data);

  // Record for replication
  await syncManager.recordChange(
    `user:${userId}`,
    previous ? ChangeOperation.Update : ChangeOperation.Insert,
    data,
    previous
  );
}

async function deleteUser(userId: string) {
  const previous = await db.get(`user:${userId}`);
  await db.delete(`user:${userId}`);

  await syncManager.recordChange(
    `user:${userId}`,
    ChangeOperation.Delete,
    undefined,
    previous
  );
}
```

### Configuring Sync Modes

```typescript
import { SyncManager, SyncMode } from '@ruvector/replication';

// Synchronous - wait for all replicas
syncManager.setSyncMode(SyncMode.Synchronous);
// Highest consistency, highest latency

// Semi-synchronous - wait for N replicas
syncManager.setSyncMode(SyncMode.SemiSync, 1);
// Good balance of consistency and performance

// Asynchronous - immediate return, replicate in background
syncManager.setSyncMode(SyncMode.Asynchronous);
syncManager.startBackgroundSync(1000); // Batch every 1 second
// Lowest latency, eventual consistency

// Check sync stats
console.log(syncManager.getStats());
// { pendingChanges: 5, lastSequence: 1234, syncMode: 'semi-sync' }
```

## API Reference

### ReplicaSet

| Method | Description |
|--------|-------------|
| `addReplica(id, address, role)` | Add a replica to the set |
| `removeReplica(id)` | Remove a replica |
| `getReplica(id)` | Get replica by ID |
| `promote(id)` | Promote secondary to primary |
| `updateStatus(id, status)` | Update replica status |
| `startHeartbeat()` | Start health monitoring |
| `stopHeartbeat()` | Stop health monitoring |
| `getStats()` | Get cluster statistics |

### VectorClock

| Method | Description |
|--------|-------------|
| `increment(replicaId)` | Increment clock for replica |
| `merge(other)` | Merge with another clock |
| `compare(other)` | Compare two clocks |
| `happensBefore(other)` | Check causal ordering |
| `isConcurrent(other)` | Check for concurrent updates |
| `toJSON()` | Serialize to JSON |

### SyncManager

| Method | Description |
|--------|-------------|
| `recordChange(key, op, value)` | Record a change for replication |
| `setSyncMode(mode, minReplicas?)` | Configure sync mode |
| `setConflictResolver(resolver)` | Set conflict resolution strategy |
| `startBackgroundSync(interval)` | Start async sync |
| `stopBackgroundSync()` | Stop async sync |
| `resolveConflict(local, remote, ...)` | Manually resolve conflict |

## Related Packages

- [@ruvector/raft](https://www.npmjs.com/package/@ruvector/raft) - Raft consensus for leader election
- [@ruvector/scipix](https://www.npmjs.com/package/@ruvector/scipix) - OCR for scientific documents
- [ruvector](https://www.npmjs.com/package/ruvector) - High-performance vector database

## License

MIT OR Apache-2.0
