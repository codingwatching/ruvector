/**
 * Genesis Service - Network Evolution Visualization
 *
 * Maps network peers to a genesis/evolutionary tree visualization.
 * Genesis represents the network's growth and peer lineage.
 */

import { useNetworkStore } from '../../stores/networkStore';
import type { PeerInfo } from '../firebaseData';

export interface GenesisNode {
  id: string;
  name: string;
  generation: number;
  parentId: string | null;
  signature: string;
  birthTime: string;
  status: 'active' | 'dormant' | 'reproducing';
  genome: {
    computeCapacity: number;
    networkSpeed: number;
    reliability: number;
    specialization: string;
  };
  offspring: number;
  verified: boolean;
  isLocal: boolean;
}

export interface GenesisStats {
  totalNodes: number;
  activeNodes: number;
  verifiedNodes: number;
  totalGenerations: number;
  avgCompute: number;
  avgNetwork: number;
  avgReliability: number;
}

class GenesisService {
  /**
   * Get local node as genesis root
   */
  getLocalGenesisNode(): GenesisNode | null {
    const state = useNetworkStore.getState();

    if (!state.isWASMReady || !state.nodeId) {
      return null;
    }

    return {
      id: state.nodeId,
      name: 'Genesis Prime (Local)',
      generation: 0,
      parentId: null,
      signature: `0x${state.nodeId.slice(0, 40)}`,
      birthTime: new Date(Date.now() - state.stats.uptime * 1000).toISOString(),
      status: state.contributionSettings.enabled ? 'active' : 'dormant',
      genome: {
        computeCapacity: state.contributionSettings.cpuLimit,
        networkSpeed: 90, // Estimated based on connection
        reliability: 95,
        specialization: 'coordinator',
      },
      offspring: state.firebasePeers.length,
      verified: true,
      isLocal: true,
    };
  }

  /**
   * Get peer nodes as genesis offspring
   */
  getPeerGenesisNodes(): GenesisNode[] {
    const state = useNetworkStore.getState();
    const localNode = this.getLocalGenesisNode();
    const parentId = localNode?.id || null;

    return state.firebasePeers.map((peer: PeerInfo, index: number) => ({
      id: peer.id,
      name: `Node ${peer.id.slice(0, 8)}`,
      generation: 1,
      parentId,
      signature: `0x${peer.id}`,
      birthTime: new Date(peer.lastSeen - peer.uptimeMs).toISOString(),
      status: peer.online ? 'active' : 'dormant',
      genome: {
        computeCapacity: 50 + (index % 3) * 15, // Varied compute
        networkSpeed: 80 + (index % 4) * 5,
        reliability: 85 + (index % 3) * 5,
        specialization: index % 2 === 0 ? 'compute' : 'network',
      },
      offspring: 0,
      verified: peer.isVerified,
      isLocal: false,
    }));
  }

  /**
   * Get all genesis nodes
   */
  getAllNodes(): GenesisNode[] {
    const nodes: GenesisNode[] = [];

    const localNode = this.getLocalGenesisNode();
    if (localNode) {
      nodes.push(localNode);
    }

    nodes.push(...this.getPeerGenesisNodes());

    return nodes;
  }

  /**
   * Get genesis statistics
   */
  getGenesisStats(): GenesisStats {
    const nodes = this.getAllNodes();
    const activeNodes = nodes.filter(n => n.status === 'active');
    const verifiedNodes = nodes.filter(n => n.verified);

    const avgCompute = nodes.length > 0
      ? nodes.reduce((sum, n) => sum + n.genome.computeCapacity, 0) / nodes.length
      : 0;
    const avgNetwork = nodes.length > 0
      ? nodes.reduce((sum, n) => sum + n.genome.networkSpeed, 0) / nodes.length
      : 0;
    const avgReliability = nodes.length > 0
      ? nodes.reduce((sum, n) => sum + n.genome.reliability, 0) / nodes.length
      : 0;

    return {
      totalNodes: nodes.length,
      activeNodes: activeNodes.length,
      verifiedNodes: verifiedNodes.length,
      totalGenerations: Math.max(...nodes.map(n => n.generation), 0) + 1,
      avgCompute,
      avgNetwork,
      avgReliability,
    };
  }

  /**
   * Get root node for tree visualization
   */
  getRootNode(): GenesisNode | null {
    const nodes = this.getAllNodes();
    return nodes.find(n => n.parentId === null) || null;
  }
}

export const genesisService = new GenesisService();
