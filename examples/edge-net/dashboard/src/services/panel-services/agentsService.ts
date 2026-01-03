/**
 * Agents Service - MCP Tools as AI Agents
 *
 * Maps MCP tools to AI agent representation for visualization.
 * In Edge-Net, "agents" are MCP tools that can execute tasks.
 */

import { useMCPStore } from '../../stores/mcpStore';

export interface RealAgent {
  id: string;
  name: string;
  type: 'researcher' | 'coder' | 'tester' | 'analyst' | 'optimizer' | 'coordinator';
  status: 'idle' | 'working' | 'learning' | 'coordinating';
  neuralPattern: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'adaptive';
  patternStrength: number;
  tasksCompleted: number;
  currentTask?: string;
  learningProgress: number;
  connections: string[];
  lastActive: string;
  isReal: boolean; // True if derived from real MCP tool
}

export interface AgentStats {
  totalAgents: number;
  workingAgents: number;
  learningAgents: number;
  totalTasks: number;
  avgLearning: number;
}

// Map MCP tool categories to agent types
const toolToAgentType: Record<string, RealAgent['type']> = {
  compute: 'optimizer',
  network: 'coordinator',
  storage: 'analyst',
  task: 'coder',
  vectorSearch: 'researcher',
  genesis: 'coordinator',
};

// Map tool types to neural patterns
const typeToPattern: Record<string, RealAgent['neuralPattern']> = {
  researcher: 'divergent',
  coder: 'convergent',
  tester: 'systems',
  analyst: 'lateral',
  optimizer: 'adaptive',
  coordinator: 'systems',
};

class AgentsService {
  /**
   * Get agents from MCP tools
   */
  getMCPAgents(): RealAgent[] {
    const state = useMCPStore.getState();

    return state.tools.map((tool) => {
      const agentType = toolToAgentType[tool.category] || 'coder';
      const pattern = typeToPattern[agentType];

      return {
        id: tool.id,
        name: tool.name,
        type: agentType,
        status: tool.status === 'running' ? 'working' :
                tool.status === 'ready' ? 'idle' : 'learning',
        neuralPattern: pattern,
        patternStrength: tool.status === 'ready' ? 85 : 50,
        tasksCompleted: 0, // MCP tools don't track this
        currentTask: tool.status === 'running' ? tool.description : undefined,
        learningProgress: tool.status === 'ready' ? 100 : 50,
        connections: [], // Could be derived from tool dependencies
        lastActive: tool.status === 'running' ? 'now' : 'idle',
        isReal: true,
      };
    });
  }

  /**
   * Get all agents (MCP + demo for visualization)
   */
  getAllAgents(): RealAgent[] {
    const mcpAgents = this.getMCPAgents();

    // If we have real MCP agents, show only those
    if (mcpAgents.length > 0) {
      return mcpAgents;
    }

    // Otherwise, return demo agents for visualization
    return this.getDemoAgents();
  }

  /**
   * Demo agents for when MCP tools aren't loaded
   */
  private getDemoAgents(): RealAgent[] {
    return [
      {
        id: 'local-coordinator',
        name: 'Local Coordinator',
        type: 'coordinator',
        status: 'coordinating',
        neuralPattern: 'systems',
        patternStrength: 95,
        tasksCompleted: 0,
        currentTask: 'Orchestrating distributed compute',
        learningProgress: 100,
        connections: [],
        lastActive: 'now',
        isReal: false,
      },
      {
        id: 'compute-optimizer',
        name: 'Compute Optimizer',
        type: 'optimizer',
        status: 'working',
        neuralPattern: 'adaptive',
        patternStrength: 89,
        tasksCompleted: 0,
        currentTask: 'Optimizing task distribution',
        learningProgress: 75,
        connections: ['local-coordinator'],
        lastActive: 'now',
        isReal: false,
      },
    ];
  }

  /**
   * Get aggregated agent statistics
   */
  getAgentStats(): AgentStats {
    const agents = this.getAllAgents();
    const workingAgents = agents.filter(a => a.status === 'working' || a.status === 'coordinating');
    const learningAgents = agents.filter(a => a.status === 'learning');

    return {
      totalAgents: agents.length,
      workingAgents: workingAgents.length,
      learningAgents: learningAgents.length,
      totalTasks: agents.reduce((sum, a) => sum + a.tasksCompleted, 0),
      avgLearning: agents.length > 0
        ? agents.reduce((sum, a) => sum + a.learningProgress, 0) / agents.length
        : 0,
    };
  }
}

export const agentsService = new AgentsService();
