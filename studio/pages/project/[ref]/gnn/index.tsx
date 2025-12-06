import { useState, useEffect, useRef } from 'react'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'
import {
  Network,
  GitBranch,
  Layers,
  Play,
  Copy,
  Check,
  Settings,
  Activity,
  Database,
  ArrowRight,
  Zap,
  Target,
  BarChart3,
} from 'lucide-react'

// Animated graph visualization
const GraphVisualization = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setFrame(f => (f + 1) % 360)
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
    const centerX = width / 2
    const centerY = height / 2

    ctx.clearRect(0, 0, width, height)

    // Define nodes in a circle
    const nodes = Array.from({ length: 6 }, (_, i) => {
      const angle = (i / 6) * Math.PI * 2 + frame * 0.01
      const radius = 40
      return {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      }
    })

    // Draw edges with animated opacity
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)'
    ctx.lineWidth = 1
    nodes.forEach((node, i) => {
      nodes.forEach((other, j) => {
        if (i < j && Math.random() > 0.3) {
          ctx.beginPath()
          ctx.moveTo(node.x, node.y)
          ctx.lineTo(other.x, other.y)
          ctx.stroke()
        }
      })
    })

    // Draw nodes
    nodes.forEach((node, i) => {
      const pulse = Math.sin(frame * 0.1 + i) * 0.3 + 0.7
      ctx.beginPath()
      ctx.arc(node.x, node.y, 6, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(59, 130, 246, ${pulse})`
      ctx.fill()
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 2
      ctx.stroke()
    })

    // Draw center node (aggregated)
    ctx.beginPath()
    ctx.arc(centerX, centerY, 10, 0, Math.PI * 2)
    const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 10)
    gradient.addColorStop(0, '#a855f7')
    gradient.addColorStop(1, '#6366f1')
    ctx.fillStyle = gradient
    ctx.fill()
  }, [frame])

  return <canvas ref={canvasRef} className="w-32 h-32" />
}

const gnnModels = [
  {
    name: 'GCN',
    title: 'Graph Convolutional Network',
    description: 'Aggregate features from neighboring nodes using normalized adjacency matrix',
    func: 'gcn_forward',
    icon: Network,
    color: 'from-blue-500 to-cyan-500',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/20',
    complexity: 'O(|E|)',
    layers: '1-3 typical',
    bestFor: 'Node classification, Semi-supervised learning',
  },
  {
    name: 'GraphSAGE',
    title: 'Graph Sample and Aggregate',
    description: 'Sample and aggregate features from node neighborhoods with learnable aggregators',
    func: 'graphsage_forward',
    icon: GitBranch,
    color: 'from-purple-500 to-pink-500',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/20',
    complexity: 'O(|V|·k²)',
    layers: '2-3 typical',
    bestFor: 'Inductive learning, Large graphs',
  },
  {
    name: 'GAT',
    title: 'Graph Attention Network',
    description: 'Apply attention mechanisms to weight neighbor contributions dynamically',
    func: 'gat_forward',
    icon: Target,
    color: 'from-orange-500 to-amber-500',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/20',
    complexity: 'O(|V|·F + |E|)',
    layers: '2-4 typical',
    bestFor: 'Heterogeneous graphs, Variable importance',
  },
]

const GraphNeuralNetworksPage: NextPageWithLayout = () => {
  const [selectedModel, setSelectedModel] = useState<typeof gnnModels[0] | null>(null)
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'playground' | 'monitor'>('overview')

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  return (
    <div className="w-full h-full overflow-y-auto">
      <div className="px-6 py-8">
        <div className="mx-auto max-w-7xl space-y-8">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500">
                  <Network className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-3xl font-bold text-foreground">Graph Neural Networks</h1>
              </div>
              <p className="text-foreground-light">
                Native PostgreSQL implementations of GCN, GraphSAGE, and GAT for relational data modeling
              </p>
            </div>
            <GraphVisualization />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { label: 'Architectures', value: '3', icon: Network, color: 'text-indigo-500' },
              { label: 'Graph Operations', value: '12', icon: GitBranch, color: 'text-purple-500' },
              { label: 'Max Nodes', value: '1M+', icon: Database, color: 'text-blue-500' },
              { label: 'Avg Inference', value: '<5ms', icon: Zap, color: 'text-green-500' },
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
              { id: 'overview', label: 'Architectures', icon: Layers },
              { id: 'playground', label: 'Playground', icon: Play },
              { id: 'monitor', label: 'Operations', icon: Activity },
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
            <div className="grid grid-cols-1 gap-6">
              {gnnModels.map((model) => (
                <div
                  key={model.name}
                  className={`rounded-xl border ${model.borderColor} ${model.bgColor} p-6`}
                >
                  <div className="flex items-start gap-6">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${model.color} shrink-0`}>
                      <model.icon className="w-8 h-8 text-white" />
                    </div>

                    <div className="flex-1 space-y-4">
                      <div>
                        <div className="flex items-center gap-3 mb-1">
                          <h3 className="text-xl font-bold text-foreground">{model.name}</h3>
                          <span className="text-sm text-foreground-light">({model.title})</span>
                        </div>
                        <p className="text-foreground-light">{model.description}</p>
                      </div>

                      <div className="grid grid-cols-3 gap-4">
                        <div className="rounded-lg bg-surface-200/50 p-3">
                          <div className="text-xs text-foreground-light mb-1">Complexity</div>
                          <div className="font-mono text-sm text-foreground">{model.complexity}</div>
                        </div>
                        <div className="rounded-lg bg-surface-200/50 p-3">
                          <div className="text-xs text-foreground-light mb-1">Layers</div>
                          <div className="font-mono text-sm text-foreground">{model.layers}</div>
                        </div>
                        <div className="rounded-lg bg-surface-200/50 p-3">
                          <div className="text-xs text-foreground-light mb-1">Best For</div>
                          <div className="text-sm text-foreground">{model.bestFor}</div>
                        </div>
                      </div>

                      <div className="relative">
                        <pre className="bg-surface-200 rounded-lg p-4 text-xs font-mono text-foreground-light overflow-x-auto">
{`SELECT ${model.func}(
  node_features,    -- vector[] of node embeddings
  adjacency_matrix, -- edge connections
  weights           -- learned parameters
) FROM graph_data;`}
                        </pre>
                        <button
                          onClick={() => copyToClipboard(`SELECT ${model.func}(node_features, adjacency_matrix, weights) FROM graph_data;`, model.name)}
                          className="absolute top-2 right-2 p-1.5 rounded bg-surface-300 hover:bg-surface-400 transition-colors"
                        >
                          {copiedCode === model.name ? (
                            <Check className="w-3.5 h-3.5 text-green-500" />
                          ) : (
                            <Copy className="w-3.5 h-3.5 text-foreground-light" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'playground' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="rounded-xl border border-default bg-surface-100 p-6 space-y-4">
                <h3 className="text-lg font-semibold text-foreground">Configure GNN</h3>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">Architecture</label>
                  <div className="grid grid-cols-3 gap-2">
                    {gnnModels.map((m) => (
                      <button
                        key={m.name}
                        onClick={() => setSelectedModel(m)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                          selectedModel?.name === m.name
                            ? 'bg-brand-500 text-white'
                            : 'bg-surface-200 text-foreground hover:bg-surface-300'
                        }`}
                      >
                        {m.name}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">Node Features Table</label>
                  <select className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground">
                    <option>nodes (embedding vector(128))</option>
                    <option>users (features vector(64))</option>
                    <option>items (vec vector(256))</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">Edges Table</label>
                  <select className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground">
                    <option>edges (source_id, target_id)</option>
                    <option>relationships (from_node, to_node)</option>
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">Hidden Dim</label>
                    <input type="number" defaultValue={64} className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">Num Layers</label>
                    <input type="number" defaultValue={2} className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground" />
                  </div>
                </div>

                <button className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-gradient-to-r from-indigo-500 to-purple-500 text-white font-medium hover:opacity-90">
                  <Play className="w-4 h-4" />
                  Run Forward Pass
                </button>
              </div>

              <div className="rounded-xl border border-default bg-surface-100 p-6 space-y-4">
                <h3 className="text-lg font-semibold text-foreground">Generated SQL</h3>
                <pre className="bg-surface-200 rounded-lg p-4 text-xs font-mono text-foreground-light overflow-x-auto h-64">
{`-- GNN Forward Pass
WITH adjacency AS (
  SELECT
    source_id,
    target_id,
    1.0 as weight
  FROM edges
),
node_features AS (
  SELECT id, embedding
  FROM nodes
),
aggregated AS (
  SELECT
    n.id,
    ${selectedModel?.func || 'gcn_forward'}(
      n.embedding,
      array_agg(m.embedding),
      model_weights
    ) as output
  FROM node_features n
  JOIN adjacency a ON a.source_id = n.id
  JOIN node_features m ON m.id = a.target_id
  GROUP BY n.id, n.embedding
)
SELECT * FROM aggregated;`}
                </pre>
              </div>
            </div>
          )}

          {activeTab === 'monitor' && (
            <div className="space-y-6">
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Graph Operations</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[
                    { name: 'aggregate_neighbors', desc: 'Sum/mean/max neighbor features', calls: 1250 },
                    { name: 'message_passing', desc: 'Send messages along edges', calls: 890 },
                    { name: 'normalize_adjacency', desc: 'Degree normalization', calls: 445 },
                    { name: 'sample_neighbors', desc: 'K-hop sampling', calls: 320 },
                    { name: 'attention_weights', desc: 'Compute edge attention', calls: 180 },
                    { name: 'readout_graph', desc: 'Graph-level pooling', calls: 95 },
                  ].map((op) => (
                    <div key={op.name} className="rounded-lg bg-surface-200 p-4">
                      <div className="flex items-center justify-between mb-2">
                        <code className="text-sm font-mono text-brand-500">{op.name}</code>
                        <span className="text-xs text-foreground-light">{op.calls} calls</span>
                      </div>
                      <p className="text-xs text-foreground-light">{op.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Use Cases</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    { title: 'Social Network Analysis', desc: 'Community detection, influence propagation' },
                    { title: 'Recommendation Systems', desc: 'User-item graphs, collaborative filtering' },
                    { title: 'Knowledge Graphs', desc: 'Entity relationships, link prediction' },
                    { title: 'Fraud Detection', desc: 'Transaction networks, anomaly detection' },
                  ].map((uc) => (
                    <div key={uc.title} className="flex items-start gap-3 p-4 rounded-lg bg-surface-200">
                      <Network className="w-5 h-5 text-brand-500 shrink-0 mt-0.5" />
                      <div>
                        <h4 className="font-medium text-foreground text-sm">{uc.title}</h4>
                        <p className="text-xs text-foreground-light">{uc.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

GraphNeuralNetworksPage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default GraphNeuralNetworksPage
