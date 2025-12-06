import { useState, useEffect, useRef } from 'react'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'
import {
  Navigation,
  Users,
  Target,
  Zap,
  Play,
  Copy,
  Check,
  Activity,
  BarChart3,
  ArrowRight,
  Bot,
  MessageSquare,
  GitBranch,
  TrendingUp,
  Settings,
  Plus,
} from 'lucide-react'

// Animated routing visualization
const RoutingVisualization = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setFrame(f => (f + 1) % 120)
    }, 50)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = canvas.offsetWidth * dpr
    canvas.height = canvas.offsetHeight * dpr
    ctx.scale(dpr, dpr)

    const width = canvas.offsetWidth
    const height = canvas.offsetHeight

    ctx.clearRect(0, 0, width, height)

    // Query node (left)
    const queryX = 30
    const queryY = height / 2
    ctx.beginPath()
    ctx.arc(queryX, queryY, 12, 0, Math.PI * 2)
    ctx.fillStyle = '#3b82f6'
    ctx.fill()

    // Agent nodes (right)
    const agents = [
      { y: height * 0.2, color: '#22c55e', active: frame % 120 < 40 },
      { y: height * 0.5, color: '#8b5cf6', active: frame % 120 >= 40 && frame % 120 < 80 },
      { y: height * 0.8, color: '#f59e0b', active: frame % 120 >= 80 },
    ]

    agents.forEach((agent, i) => {
      const agentX = width - 30

      // Draw connection line
      ctx.beginPath()
      ctx.moveTo(queryX + 12, queryY)
      ctx.lineTo(agentX - 12, agent.y)
      ctx.strokeStyle = agent.active ? agent.color : 'rgba(255,255,255,0.1)'
      ctx.lineWidth = agent.active ? 2 : 1
      ctx.stroke()

      // Draw agent node
      ctx.beginPath()
      ctx.arc(agentX, agent.y, 10, 0, Math.PI * 2)
      ctx.fillStyle = agent.active ? agent.color : 'rgba(255,255,255,0.3)'
      ctx.fill()

      // Animated packet
      if (agent.active) {
        const progress = (frame % 40) / 40
        const packetX = queryX + 12 + (agentX - 12 - queryX - 12) * progress
        const packetY = queryY + (agent.y - queryY) * progress
        ctx.beginPath()
        ctx.arc(packetX, packetY, 4, 0, Math.PI * 2)
        ctx.fillStyle = '#fff'
        ctx.fill()
      }
    })
  }, [frame])

  return <canvas ref={canvasRef} className="w-32 h-24" />
}

// Simulated agent data
const agents = [
  { id: 1, name: 'Support Agent', capabilities: 'Customer support, FAQ, troubleshooting', queries: 1250, successRate: 94 },
  { id: 2, name: 'Sales Agent', capabilities: 'Product info, pricing, recommendations', queries: 890, successRate: 91 },
  { id: 3, name: 'Technical Agent', capabilities: 'Code help, documentation, debugging', queries: 567, successRate: 88 },
  { id: 4, name: 'General Agent', capabilities: 'General queries, routing fallback', queries: 234, successRate: 85 },
]

const AgentRoutingPage: NextPageWithLayout = () => {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'agents' | 'test'>('overview')
  const [testQuery, setTestQuery] = useState('')
  const [testResult, setTestResult] = useState<{ agent: string; confidence: number } | null>(null)

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const runTestQuery = () => {
    if (!testQuery.trim()) return
    // Simulate routing
    const randomAgent = agents[Math.floor(Math.random() * agents.length)]
    setTestResult({
      agent: randomAgent.name,
      confidence: 0.75 + Math.random() * 0.2,
    })
  }

  const routingFunctions = [
    {
      name: 'register_agent',
      signature: '(name text, description text, capabilities_embedding vector)',
      description: 'Add a new agent to the routing registry with capability embedding',
    },
    {
      name: 'route_query',
      signature: '(query_embedding vector, top_k int DEFAULT 3)',
      description: 'Find best matching agents for a query using similarity search',
    },
    {
      name: 'update_agent_performance',
      signature: '(agent_id int, success boolean, latency_ms float)',
      description: 'Update agent statistics based on routing outcomes',
    },
  ]

  return (
    <div className="w-full h-full overflow-y-auto">
      <div className="px-6 py-8">
        <div className="mx-auto max-w-7xl space-y-8">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500">
                  <Navigation className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-3xl font-bold text-foreground">Agent Routing</h1>
                <span className="px-2 py-1 rounded-full bg-cyan-500/10 text-cyan-500 text-xs font-medium">
                  Tiny Dancer
                </span>
              </div>
              <p className="text-foreground-light">
                Intelligent semantic routing for multi-agent systems
              </p>
            </div>
            <RoutingVisualization />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { label: 'Active Agents', value: agents.length, icon: Bot, color: 'text-cyan-500' },
              { label: 'Queries Routed', value: '2.9K', icon: MessageSquare, color: 'text-blue-500' },
              { label: 'Avg Latency', value: '12ms', icon: Zap, color: 'text-green-500' },
              { label: 'Success Rate', value: '92%', icon: Target, color: 'text-purple-500' },
            ].map((stat, i) => (
              <div key={i} className="rounded-xl border border-default bg-surface-100 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-foreground-light">{stat.label}</span>
                  <stat.icon className={`w-4 h-4 ${stat.color}`} />
                </div>
                <div className="text-2xl font-bold text-foreground">{stat.value}</div>
              </div>
            ))}
          </div>

          {/* Tabs */}
          <div className="flex gap-2 border-b border-default">
            {[
              { id: 'overview', label: 'How It Works', icon: GitBranch },
              { id: 'agents', label: 'Agent Registry', icon: Users },
              { id: 'test', label: 'Test Routing', icon: Play },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-brand-500 text-brand-500'
                    : 'border-transparent text-foreground-light hover:text-foreground'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* How It Works */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-6">Routing Pipeline</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    {
                      step: 1,
                      title: 'Register Agents',
                      description: 'Define agent capabilities as embeddings',
                      icon: Plus,
                      color: 'from-cyan-500 to-blue-500',
                    },
                    {
                      step: 2,
                      title: 'Route Queries',
                      description: 'Match requests via similarity search',
                      icon: Navigation,
                      color: 'from-blue-500 to-purple-500',
                    },
                    {
                      step: 3,
                      title: 'Learn & Adapt',
                      description: 'Improve routing from outcomes',
                      icon: TrendingUp,
                      color: 'from-purple-500 to-pink-500',
                    },
                  ].map((item, i) => (
                    <div key={item.step} className="relative">
                      <div className="text-center">
                        <div className={`mx-auto w-16 h-16 rounded-2xl bg-gradient-to-br ${item.color} flex items-center justify-center mb-4`}>
                          <item.icon className="w-8 h-8 text-white" />
                        </div>
                        <div className="text-xs text-foreground-light mb-1">Step {item.step}</div>
                        <h4 className="font-semibold text-foreground mb-2">{item.title}</h4>
                        <p className="text-sm text-foreground-light">{item.description}</p>
                      </div>
                      {i < 2 && (
                        <div className="hidden md:block absolute top-8 -right-3">
                          <ArrowRight className="w-6 h-6 text-foreground-light" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Routing Functions */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Core Functions</h3>
                <div className="space-y-4">
                  {routingFunctions.map((func) => (
                    <div key={func.name} className="p-4 rounded-lg bg-surface-200">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <code className="text-sm font-mono text-brand-500">{func.name}</code>
                          <div className="text-xs font-mono text-foreground-light mt-1">{func.signature}</div>
                          <p className="text-sm text-foreground-light mt-2">{func.description}</p>
                        </div>
                        <button
                          onClick={() => copyToClipboard(`SELECT ${func.name}${func.signature};`, func.name)}
                          className="p-2 rounded bg-surface-300 hover:bg-surface-400"
                        >
                          {copiedCode === func.name ? (
                            <Check className="w-4 h-4 text-green-500" />
                          ) : (
                            <Copy className="w-4 h-4 text-foreground-light" />
                          )}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Schema */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Agent Registry Schema</h3>
                <pre className="bg-surface-200 rounded-lg p-4 text-xs font-mono text-foreground-light overflow-x-auto">
{`CREATE TABLE agents (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  capabilities vector(384),  -- Embedding of agent capabilities
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index for fast similarity search
CREATE INDEX ON agents
USING hnsw (capabilities vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Performance tracking
CREATE TABLE agent_metrics (
  agent_id INT REFERENCES agents(id),
  queries_handled INT DEFAULT 0,
  success_count INT DEFAULT 0,
  avg_latency_ms FLOAT DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);`}
                </pre>
              </div>

              {/* Use Cases */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  { title: 'Customer Support', desc: 'Route tickets to specialized support agents', icon: MessageSquare },
                  { title: 'Task Distribution', desc: 'Assign work items to appropriate AI agents', icon: GitBranch },
                  { title: 'Multi-Agent RAG', desc: 'Select domain experts for retrieval tasks', icon: Bot },
                  { title: 'Intent Classification', desc: 'Classify and route user intents semantically', icon: Target },
                ].map((uc) => (
                  <div key={uc.title} className="rounded-xl border border-default bg-surface-100 p-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-lg bg-cyan-500/10">
                        <uc.icon className="w-5 h-5 text-cyan-500" />
                      </div>
                      <div>
                        <h4 className="font-medium text-foreground">{uc.title}</h4>
                        <p className="text-sm text-foreground-light">{uc.desc}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'agents' && (
            <div className="space-y-6">
              {/* Add Agent Button */}
              <div className="flex justify-end">
                <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium hover:opacity-90">
                  <Plus className="w-4 h-4" />
                  Register Agent
                </button>
              </div>

              {/* Agent List */}
              <div className="rounded-xl border border-default bg-surface-100 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-surface-200">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Agent</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Capabilities</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Queries</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Success Rate</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Status</th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-foreground-light uppercase">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-default">
                      {agents.map((agent) => (
                        <tr key={agent.id} className="hover:bg-surface-200 transition-colors">
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
                                <Bot className="w-4 h-4 text-white" />
                              </div>
                              <span className="font-medium text-foreground">{agent.name}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 text-sm text-foreground-light max-w-xs truncate">
                            {agent.capabilities}
                          </td>
                          <td className="px-6 py-4 text-sm text-foreground">
                            {agent.queries.toLocaleString()}
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-2 bg-surface-300 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-green-500 rounded-full"
                                  style={{ width: `${agent.successRate}%` }}
                                />
                              </div>
                              <span className="text-sm text-foreground">{agent.successRate}%</span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <span className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                              <span className="text-sm text-green-500">Active</span>
                            </span>
                          </td>
                          <td className="px-6 py-4 text-right">
                            <button className="text-foreground-light hover:text-foreground">
                              <Settings className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Agent Performance Chart Placeholder */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Query Distribution</h3>
                <div className="grid grid-cols-4 gap-4">
                  {agents.map((agent) => (
                    <div key={agent.id} className="text-center">
                      <div className="text-2xl font-bold text-foreground">{Math.round(agent.queries / 29)}%</div>
                      <div className="text-xs text-foreground-light">{agent.name.split(' ')[0]}</div>
                      <div className="mt-2 h-2 bg-surface-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                          style={{ width: `${(agent.queries / 1250) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'test' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Test Input */}
              <div className="rounded-xl border border-default bg-surface-100 p-6 space-y-4">
                <h3 className="text-lg font-semibold text-foreground">Test Query Routing</h3>
                <p className="text-sm text-foreground-light">
                  Enter a query to see which agent would handle it
                </p>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">Query</label>
                  <textarea
                    value={testQuery}
                    onChange={(e) => setTestQuery(e.target.value)}
                    placeholder="e.g., How do I reset my password?"
                    className="w-full px-4 py-3 rounded-lg bg-surface-200 border border-default text-foreground resize-none h-32"
                  />
                </div>

                <button
                  onClick={runTestQuery}
                  disabled={!testQuery.trim()}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium hover:opacity-90 disabled:opacity-50"
                >
                  <Navigation className="w-4 h-4" />
                  Route Query
                </button>
              </div>

              {/* Test Result */}
              <div className="rounded-xl border border-default bg-surface-100 p-6 space-y-4">
                <h3 className="text-lg font-semibold text-foreground">Routing Result</h3>

                {testResult ? (
                  <div className="space-y-4">
                    <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
                          <Bot className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <div className="font-semibold text-foreground">{testResult.agent}</div>
                          <div className="text-sm text-foreground-light">Selected Agent</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-foreground-light">Confidence:</span>
                        <div className="flex-1 h-2 bg-surface-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-green-500 rounded-full"
                            style={{ width: `${testResult.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-foreground">
                          {(testResult.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-surface-200">
                      <h4 className="text-sm font-medium text-foreground mb-2">SQL Used</h4>
                      <pre className="text-xs font-mono text-foreground-light">
{`SELECT route_query(
  embed('${testQuery.slice(0, 30)}...'),
  top_k => 1
);`}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <Navigation className="w-12 h-12 text-foreground-light opacity-30 mb-4" />
                    <p className="text-sm text-foreground-light">
                      Enter a query and click "Route Query" to see results
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

AgentRoutingPage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default AgentRoutingPage
