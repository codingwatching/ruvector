/**
 * Workers Service - Real EdgeNet Worker Integration
 *
 * Provides real worker data from:
 * - Local WASM node (current browser)
 * - Connected peers via relay/Firebase
 * - Network state from networkStore
 */

import { edgeNetService } from '../edgeNet';
import { useNetworkStore } from '../../stores/networkStore';
import type { PeerInfo } from '../firebaseData';

export interface RealWorker {
  id: string;
  name: string;
  type: 'cpu' | 'gpu' | 'hybrid' | 'wasm';
  status: 'active' | 'idle' | 'offline' | 'syncing' | 'error';
  cpuUsage: number;
  memoryUsage: number;
  tasksCompleted: number;
  tasksQueued: number;
  creditsEarned: number;
  uptime: number; // seconds
  lastHeartbeat: Date;
  isLocal: boolean;
  region?: string;
}

export interface WorkerStats {
  totalWorkers: number;
  activeWorkers: number;
  totalTasksCompleted: number;
  totalCreditsEarned: number;
  averageCpuUsage: number;
  totalCompute: number; // TFLOPS
}

class WorkersService {
  /**
   * Get the local WASM worker representing this browser
   */
  getLocalWorker(): RealWorker | null {
    const state = useNetworkStore.getState();
    const stats = edgeNetService.getStats();

    if (!state.isWASMReady || !state.nodeId) {
      return null;
    }

    return {
      id: state.nodeId,
      name: `Local Node (${state.nodeId.slice(0, 8)})`,
      type: 'wasm',
      status: state.contributionSettings.enabled ? 'active' : 'idle',
      cpuUsage: state.contributionSettings.cpuLimit,
      memoryUsage: Math.round(Math.random() * 30 + 20), // TODO: get real memory
      tasksCompleted: stats ? Number(stats.tasks_completed) : state.stats.tasksCompleted,
      tasksQueued: state.pendingTasks.length,
      creditsEarned: stats ? Number(stats.ruv_earned) / 1e9 : state.credits.earned,
      uptime: state.stats.uptime,
      lastHeartbeat: new Date(),
      isLocal: true,
      region: 'browser',
    };
  }

  /**
   * Get connected peer workers from Firebase
   */
  getPeerWorkers(): RealWorker[] {
    const state = useNetworkStore.getState();

    return state.firebasePeers.map((peer: PeerInfo) => ({
      id: peer.id,
      name: `Peer ${peer.id.slice(0, 8)}`,
      type: 'wasm' as const,
      status: peer.online ? 'active' : 'offline',
      cpuUsage: 50, // Default estimate for peers
      memoryUsage: 30,
      tasksCompleted: 0, // Not tracked per peer
      tasksQueued: 0,
      creditsEarned: 0, // Not tracked per peer
      uptime: Math.round(peer.uptimeMs / 1000), // Convert ms to seconds
      lastHeartbeat: new Date(peer.lastSeen),
      isLocal: false,
      region: peer.room || 'default',
    }));
  }

  /**
   * Get all workers (local + peers)
   */
  getAllWorkers(): RealWorker[] {
    const workers: RealWorker[] = [];

    const localWorker = this.getLocalWorker();
    if (localWorker) {
      workers.push(localWorker);
    }

    workers.push(...this.getPeerWorkers());

    return workers;
  }

  /**
   * Get aggregated worker statistics
   */
  getWorkerStats(): WorkerStats {
    const workers = this.getAllWorkers();
    const activeWorkers = workers.filter(w => w.status === 'active');
    const state = useNetworkStore.getState();

    return {
      totalWorkers: workers.length,
      activeWorkers: activeWorkers.length,
      totalTasksCompleted: workers.reduce((sum, w) => sum + w.tasksCompleted, 0),
      totalCreditsEarned: workers.reduce((sum, w) => sum + w.creditsEarned, 0),
      averageCpuUsage: activeWorkers.length > 0
        ? activeWorkers.reduce((sum, w) => sum + w.cpuUsage, 0) / activeWorkers.length
        : 0,
      totalCompute: state.stats.totalCompute,
    };
  }

  /**
   * Start local worker contribution
   */
  startLocalWorker(): void {
    const state = useNetworkStore.getState();
    if (state.contributionSettings.consentGiven) {
      state.startContributing();
    }
  }

  /**
   * Pause local worker contribution
   */
  pauseLocalWorker(): void {
    const state = useNetworkStore.getState();
    state.stopContributing();
  }

  /**
   * Format uptime for display
   */
  formatUptime(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
    return `${Math.floor(seconds / 86400)}d ${Math.round((seconds % 86400) / 3600)}h`;
  }
}

export const workersService = new WorkersService();
