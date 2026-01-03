/**
 * Workers Service Tests (TDD)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { workersService } from '../services/panel-services/workersService';

// Mock the stores and services
vi.mock('../stores/networkStore', () => ({
  useNetworkStore: {
    getState: vi.fn(() => ({
      isWASMReady: true,
      nodeId: 'test-node-12345678',
      contributionSettings: {
        enabled: true,
        cpuLimit: 50,
        consentGiven: true,
      },
      stats: {
        tasksCompleted: 10,
        totalCompute: 1.5,
        uptime: 3600,
      },
      credits: { earned: 100 },
      pendingTasks: [],
      firebasePeers: [
        {
          id: 'peer-1',
          room: 'default',
          online: true,
          lastSeen: Date.now(),
          capabilities: ['compute'],
          isVerified: false,
          uptimeMs: 3600000,
        },
      ],
      startContributing: vi.fn(),
      stopContributing: vi.fn(),
    })),
  },
}));

vi.mock('../services/edgeNet', () => ({
  edgeNetService: {
    getStats: vi.fn(() => ({
      tasks_completed: BigInt(10),
      ruv_earned: BigInt(100e9),
    })),
  },
}));

describe('WorkersService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getLocalWorker', () => {
    it('should return local worker when WASM is ready', () => {
      const worker = workersService.getLocalWorker();

      expect(worker).not.toBeNull();
      expect(worker?.isLocal).toBe(true);
      expect(worker?.type).toBe('wasm');
      expect(worker?.status).toBe('active');
    });

    it('should include truncated node ID in worker name', () => {
      const worker = workersService.getLocalWorker();

      // Node ID is sliced to first 8 chars
      expect(worker?.name).toContain('test-nod');
      expect(worker?.name).toContain('Local Node');
    });
  });

  describe('getPeerWorkers', () => {
    it('should return connected peers from Firebase', () => {
      const peers = workersService.getPeerWorkers();

      expect(peers.length).toBeGreaterThan(0);
      expect(peers[0].isLocal).toBe(false);
    });

    it('should set peer status based on isActive flag', () => {
      const peers = workersService.getPeerWorkers();

      expect(peers[0].status).toBe('active');
    });
  });

  describe('getAllWorkers', () => {
    it('should combine local and peer workers', () => {
      const workers = workersService.getAllWorkers();

      expect(workers.length).toBeGreaterThanOrEqual(2); // local + at least 1 peer
      expect(workers.some(w => w.isLocal)).toBe(true);
      expect(workers.some(w => !w.isLocal)).toBe(true);
    });
  });

  describe('getWorkerStats', () => {
    it('should aggregate worker statistics', () => {
      const stats = workersService.getWorkerStats();

      expect(stats.totalWorkers).toBeGreaterThan(0);
      expect(stats.activeWorkers).toBeGreaterThan(0);
      expect(stats.totalTasksCompleted).toBeGreaterThan(0);
    });
  });

  describe('formatUptime', () => {
    it('should format seconds correctly', () => {
      expect(workersService.formatUptime(30)).toBe('30s');
    });

    it('should format minutes correctly', () => {
      expect(workersService.formatUptime(120)).toBe('2m');
    });

    it('should format hours correctly', () => {
      expect(workersService.formatUptime(3660)).toBe('1h 1m');
    });

    it('should format days correctly', () => {
      expect(workersService.formatUptime(90000)).toBe('1d 1h');
    });
  });
});
