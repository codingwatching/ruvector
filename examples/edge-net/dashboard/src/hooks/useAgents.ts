/**
 * useAgents Hook - Real-time agent data
 */

import { useState, useEffect, useCallback } from 'react';
import { agentsService, type RealAgent, type AgentStats } from '../services/panel-services/agentsService';
import { useMCPStore } from '../stores/mcpStore';

export function useAgents() {
  const [agents, setAgents] = useState<RealAgent[]>([]);
  const [stats, setStats] = useState<AgentStats>({
    totalAgents: 0,
    workingAgents: 0,
    learningAgents: 0,
    totalTasks: 0,
    avgLearning: 0,
  });
  const [isLoading, setIsLoading] = useState(true);

  // Subscribe to MCP store changes
  const mcpTools = useMCPStore(state => state.tools);

  // Refresh agent data
  const refresh = useCallback(() => {
    const allAgents = agentsService.getAllAgents();
    const agentStats = agentsService.getAgentStats();
    setAgents(allAgents);
    setStats(agentStats);
    setIsLoading(false);
  }, []);

  // Initial load and periodic refresh
  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [refresh, mcpTools]);

  return {
    agents,
    stats,
    isLoading,
    refresh,
  };
}
