import { useState, useEffect, useCallback, useRef } from 'react'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'
import {
  Database,
  Zap,
  Target,
  Settings,
  Play,
  Copy,
  Check,
  TrendingUp,
  Clock,
  Layers,
  BarChart3,
  RefreshCw,
  Plus,
  Search,
  ArrowRight,
} from 'lucide-react'

interface IndexStats {
  name: string
  type: 'hnsw' | 'ivfflat'
  table: string
  column: string
  size: string
  rows: number
  avgQueryTime: number
  status: 'active' | 'building' | 'invalid'
}

// Simulated index stats - in production would fetch from pg-meta
const useIndexStats = () => {
  const [stats, setStats] = useState<IndexStats[]>([
    { name: 'items_embedding_hnsw_idx', type: 'hnsw', table: 'items', column: 'embedding', size: '24 MB', rows: 125000, avgQueryTime: 2.3, status: 'active' },
    { name: 'documents_vec_ivfflat_idx', type: 'ivfflat', table: 'documents', column: 'vec', size: '18 MB', rows: 89000, avgQueryTime: 4.1, status: 'active' },
  ])
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(() => {
    setLoading(true)
    setTimeout(() => {
      setStats(prev => prev.map(s => ({
        ...s,
        avgQueryTime: Math.max(0.5, s.avgQueryTime + (Math.random() - 0.5) * 0.5)
      })))
      setLoading(false)
    }, 500)
  }, [])

  return { stats, loading, refresh }
}

// Performance graph component
const PerformanceGraph = ({ data }: { data: number[] }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

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

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, height)
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)')
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)')

    const maxVal = Math.max(...data) * 1.2
    const stepX = width / (data.length - 1)

    ctx.beginPath()
    ctx.moveTo(0, height)
    data.forEach((val, i) => {
      const x = i * stepX
      const y = height - (val / maxVal) * height * 0.9
      if (i === 0) ctx.lineTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.lineTo(width, height)
    ctx.closePath()
    ctx.fillStyle = gradient
    ctx.fill()

    // Draw line
    ctx.beginPath()
    data.forEach((val, i) => {
      const x = i * stepX
      const y = height - (val / maxVal) * height * 0.9
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.strokeStyle = '#3b82f6'
    ctx.lineWidth = 2
    ctx.stroke()
  }, [data])

  return <canvas ref={canvasRef} className="w-full h-20" />
}

const VectorIndexesPage: NextPageWithLayout = () => {
  const { stats, loading, refresh } = useIndexStats()
  const [copiedIndex, setCopiedIndex] = useState<string | null>(null)
  const [selectedTab, setSelectedTab] = useState<'overview' | 'create' | 'monitor'>('overview')
  const [queryTimes, setQueryTimes] = useState<number[]>([2.1, 2.3, 2.0, 2.5, 2.2, 2.4, 2.1, 2.3, 2.6, 2.2])

  useEffect(() => {
    const interval = setInterval(() => {
      setQueryTimes(prev => [...prev.slice(1), Math.max(1, prev[prev.length - 1] + (Math.random() - 0.5) * 0.8)])
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedIndex(id)
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  const indexTypes = [
    {
      type: 'HNSW',
      icon: Zap,
      color: 'from-blue-500 to-cyan-500',
      bgColor: 'bg-blue-500/10',
      borderColor: 'border-blue-500/20',
      description: 'Hierarchical Navigable Small World graphs for fast approximate nearest neighbor search',
      metrics: { speed: '~2ms', recall: '99%+', memory: 'High' },
      code: `CREATE INDEX ON items
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);`,
    },
    {
      type: 'IVFFlat',
      icon: Layers,
      color: 'from-purple-500 to-pink-500',
      bgColor: 'bg-purple-500/10',
      borderColor: 'border-purple-500/20',
      description: 'Inverted file with flat compression for memory-efficient vector search',
      metrics: { speed: '~5ms', recall: '95%+', memory: 'Medium' },
      code: `CREATE INDEX ON items
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);`,
    },
  ]

  const distanceOps = [
    { name: 'vector_cosine_ops', symbol: '<=>',  desc: 'Cosine similarity', use: 'Text embeddings, normalized vectors' },
    { name: 'vector_l2_ops', symbol: '<->',  desc: 'Euclidean (L2) distance', use: 'Image features, spatial data' },
    { name: 'vector_ip_ops', symbol: '<#>',  desc: 'Inner product', use: 'Maximum inner product search' },
  ]

  return (
    <div className="w-full h-full overflow-y-auto">
      <div className="px-6 py-8">
        <div className="mx-auto max-w-7xl space-y-8">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500">
                  <Database className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-3xl font-bold text-foreground">Vector Indexes</h1>
              </div>
              <p className="text-foreground-light">
                High-performance HNSW and IVFFlat indexes for similarity search
              </p>
            </div>
            <button
              onClick={refresh}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-surface-200 hover:bg-surface-300 transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              <span className="text-sm">Refresh</span>
            </button>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Indexes', value: stats.length, icon: Database, color: 'text-blue-500' },
              { label: 'Total Vectors', value: stats.reduce((a, s) => a + s.rows, 0).toLocaleString(), icon: Layers, color: 'text-purple-500' },
              { label: 'Avg Query Time', value: `${(stats.reduce((a, s) => a + s.avgQueryTime, 0) / stats.length).toFixed(1)}ms`, icon: Clock, color: 'text-green-500' },
              { label: 'Index Size', value: stats.reduce((a, s) => a + parseInt(s.size), 0) + ' MB', icon: BarChart3, color: 'text-orange-500' },
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
              { id: 'overview', label: 'Index Types', icon: Layers },
              { id: 'create', label: 'Create Index', icon: Plus },
              { id: 'monitor', label: 'Monitor', icon: BarChart3 },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSelectedTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  selectedTab === tab.id
                    ? 'border-brand-500 text-brand-500'
                    : 'border-transparent text-foreground-light hover:text-foreground'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {selectedTab === 'overview' && (
            <div className="space-y-6">
              {/* Index Type Cards */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {indexTypes.map((idx) => (
                  <div key={idx.type} className={`rounded-xl border ${idx.borderColor} ${idx.bgColor} p-6 space-y-4`}>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg bg-gradient-to-br ${idx.color}`}>
                          <idx.icon className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-foreground">{idx.type}</h3>
                          <p className="text-sm text-foreground-light">{idx.description}</p>
                        </div>
                      </div>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-3 gap-4">
                      {Object.entries(idx.metrics).map(([key, value]) => (
                        <div key={key} className="text-center">
                          <div className="text-lg font-semibold text-foreground">{value}</div>
                          <div className="text-xs text-foreground-light capitalize">{key}</div>
                        </div>
                      ))}
                    </div>

                    {/* Code Block */}
                    <div className="relative">
                      <pre className="bg-surface-200 rounded-lg p-4 text-xs font-mono text-foreground-light overflow-x-auto">
                        {idx.code}
                      </pre>
                      <button
                        onClick={() => copyToClipboard(idx.code, idx.type)}
                        className="absolute top-2 right-2 p-1.5 rounded bg-surface-300 hover:bg-surface-400 transition-colors"
                      >
                        {copiedIndex === idx.type ? (
                          <Check className="w-3.5 h-3.5 text-green-500" />
                        ) : (
                          <Copy className="w-3.5 h-3.5 text-foreground-light" />
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Distance Operators */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-brand-500" />
                  Distance Operators
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {distanceOps.map((op) => (
                    <div key={op.name} className="rounded-lg bg-surface-200 p-4 space-y-2">
                      <div className="flex items-center justify-between">
                        <code className="text-sm font-mono text-brand-500">{op.name}</code>
                        <span className="text-lg font-mono text-foreground-light">{op.symbol}</span>
                      </div>
                      <p className="text-sm font-medium text-foreground">{op.desc}</p>
                      <p className="text-xs text-foreground-light">{op.use}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {selectedTab === 'create' && (
            <div className="space-y-6">
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Create New Vector Index</h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">Table</label>
                      <select className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground">
                        <option>items</option>
                        <option>documents</option>
                        <option>embeddings</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">Vector Column</label>
                      <select className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground">
                        <option>embedding</option>
                        <option>vec</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">Index Type</label>
                      <div className="flex gap-2">
                        <button className="flex-1 px-4 py-2 rounded-lg bg-brand-500 text-white text-sm font-medium">
                          HNSW
                        </button>
                        <button className="flex-1 px-4 py-2 rounded-lg bg-surface-200 text-foreground text-sm font-medium hover:bg-surface-300">
                          IVFFlat
                        </button>
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-foreground mb-2">Distance Function</label>
                      <select className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground">
                        <option>vector_cosine_ops (Cosine)</option>
                        <option>vector_l2_ops (Euclidean)</option>
                        <option>vector_ip_ops (Inner Product)</option>
                      </select>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h4 className="text-sm font-medium text-foreground">HNSW Parameters</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs text-foreground-light mb-1">m (connections)</label>
                        <input type="number" defaultValue={16} className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground text-sm" />
                      </div>
                      <div>
                        <label className="block text-xs text-foreground-light mb-1">ef_construction</label>
                        <input type="number" defaultValue={64} className="w-full px-3 py-2 rounded-lg bg-surface-200 border border-default text-foreground text-sm" />
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-surface-200 border border-default">
                      <h5 className="text-xs font-medium text-foreground-light mb-2">Preview SQL</h5>
                      <pre className="text-xs font-mono text-foreground-light">
{`CREATE INDEX items_embedding_idx ON items
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);`}
                      </pre>
                    </div>

                    <button className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-medium hover:opacity-90 transition-opacity">
                      <Play className="w-4 h-4" />
                      Create Index
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedTab === 'monitor' && (
            <div className="space-y-6">
              {/* Active Indexes */}
              <div className="rounded-xl border border-default bg-surface-100 overflow-hidden">
                <div className="px-6 py-4 border-b border-default flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-foreground">Active Indexes</h3>
                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-foreground-light" />
                      <input
                        type="text"
                        placeholder="Search indexes..."
                        className="pl-9 pr-4 py-1.5 rounded-lg bg-surface-200 border border-default text-sm text-foreground"
                      />
                    </div>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-surface-200">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Name</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Type</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Table</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Rows</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Size</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Avg Query</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Status</th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-foreground-light uppercase">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-default">
                      {stats.map((idx) => (
                        <tr key={idx.name} className="hover:bg-surface-200 transition-colors">
                          <td className="px-6 py-4">
                            <code className="text-sm font-mono text-foreground">{idx.name}</code>
                          </td>
                          <td className="px-6 py-4">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              idx.type === 'hnsw' ? 'bg-blue-500/10 text-blue-500' : 'bg-purple-500/10 text-purple-500'
                            }`}>
                              {idx.type.toUpperCase()}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sm text-foreground">{idx.table}</td>
                          <td className="px-6 py-4 text-sm text-foreground">{idx.rows.toLocaleString()}</td>
                          <td className="px-6 py-4 text-sm text-foreground">{idx.size}</td>
                          <td className="px-6 py-4 text-sm text-foreground">{idx.avgQueryTime.toFixed(1)}ms</td>
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

              {/* Performance Graph */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Query Performance (Last 20s)</h3>
                <PerformanceGraph data={queryTimes} />
                <div className="flex justify-between mt-2 text-xs text-foreground-light">
                  <span>20s ago</span>
                  <span>Now</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

VectorIndexesPage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default VectorIndexesPage
