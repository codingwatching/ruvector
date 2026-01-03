/**
 * useGenesis Hook - Real-time genesis/network evolution data
 */

import { useState, useEffect, useCallback } from 'react';
import { genesisService, type GenesisNode, type GenesisStats } from '../services/panel-services/genesisService';
import { useNetworkStore } from '../stores/networkStore';

export function useGenesis() {
  const [nodes, setNodes] = useState<GenesisNode[]>([]);
  const [rootNode, setRootNode] = useState<GenesisNode | null>(null);
  const [stats, setStats] = useState<GenesisStats>({
    totalNodes: 0,
    activeNodes: 0,
    verifiedNodes: 0,
    totalGenerations: 1,
    avgCompute: 0,
    avgNetwork: 0,
    avgReliability: 0,
  });
  const [isLoading, setIsLoading] = useState(true);

  // Subscribe to network store changes
  const isWASMReady = useNetworkStore(state => state.isWASMReady);
  const firebasePeers = useNetworkStore(state => state.firebasePeers);

  // Refresh genesis data
  const refresh = useCallback(() => {
    const allNodes = genesisService.getAllNodes();
    const root = genesisService.getRootNode();
    const genesisStats = genesisService.getGenesisStats();
    setNodes(allNodes);
    setRootNode(root);
    setStats(genesisStats);
    setIsLoading(false);
  }, []);

  // Initial load and periodic refresh
  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [refresh, isWASMReady, firebasePeers]);

  return {
    nodes,
    rootNode,
    stats,
    isLoading,
    refresh,
  };
}
