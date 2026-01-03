/**
 * useWorkers Hook - Real-time worker data from EdgeNet
 */

import { useState, useEffect, useCallback } from 'react';
import { workersService, type RealWorker, type WorkerStats } from '../services/panel-services/workersService';
import { useNetworkStore } from '../stores/networkStore';

export function useWorkers() {
  const [workers, setWorkers] = useState<RealWorker[]>([]);
  const [stats, setStats] = useState<WorkerStats>({
    totalWorkers: 0,
    activeWorkers: 0,
    totalTasksCompleted: 0,
    totalCreditsEarned: 0,
    averageCpuUsage: 0,
    totalCompute: 0,
  });
  const [isLoading, setIsLoading] = useState(true);

  // Subscribe to network store changes
  const isWASMReady = useNetworkStore(state => state.isWASMReady);
  const contributionEnabled = useNetworkStore(state => state.contributionSettings.enabled);
  const firebasePeers = useNetworkStore(state => state.firebasePeers);
  const networkStats = useNetworkStore(state => state.stats);

  // Refresh worker data
  const refresh = useCallback(() => {
    const allWorkers = workersService.getAllWorkers();
    const workerStats = workersService.getWorkerStats();
    setWorkers(allWorkers);
    setStats(workerStats);
    setIsLoading(false);
  }, []);

  // Initial load and periodic refresh
  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 2000);
    return () => clearInterval(interval);
  }, [refresh, isWASMReady, contributionEnabled, firebasePeers, networkStats]);

  // Actions
  const startWorker = useCallback((workerId: string) => {
    const worker = workers.find(w => w.id === workerId);
    if (worker?.isLocal) {
      workersService.startLocalWorker();
      refresh();
    }
  }, [workers, refresh]);

  const pauseWorker = useCallback((workerId: string) => {
    const worker = workers.find(w => w.id === workerId);
    if (worker?.isLocal) {
      workersService.pauseLocalWorker();
      refresh();
    }
  }, [workers, refresh]);

  return {
    workers,
    stats,
    isLoading,
    refresh,
    startWorker,
    pauseWorker,
    formatUptime: workersService.formatUptime,
  };
}
