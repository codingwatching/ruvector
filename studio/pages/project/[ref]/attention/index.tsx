import { useState, useEffect, useRef } from 'react'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'
import {
  Brain,
  Zap,
  Cpu,
  Play,
  Copy,
  Check,
  Clock,
  BarChart3,
  Search,
  Filter,
  ChevronRight,
  Sparkles,
  Activity,
  Layers,
} from 'lucide-react'

interface AttentionType {
  name: string
  func: string
  category: 'core' | 'efficient' | 'specialized' | 'causal'
  description: string
  complexity: string
  useCase: string
}

const attentionTypes: AttentionType[] = [
  { name: 'Scaled Dot-Product', func: 'scaled_dot_product_attention', category: 'core', description: 'Foundation attention mechanism', complexity: 'O(n²)', useCase: 'General transformer layers' },
  { name: 'Multi-Head', func: 'multi_head_attention', category: 'core', description: 'Parallel attention with multiple heads', complexity: 'O(n²·h)', useCase: 'BERT, GPT models' },
  { name: 'Self Attention', func: 'self_attention', category: 'core', description: 'Query, key, value from same sequence', complexity: 'O(n²)', useCase: 'Sequence modeling' },
  { name: 'Cross Attention', func: 'cross_attention', category: 'core', description: 'Attention between two sequences', complexity: 'O(n·m)', useCase: 'Encoder-decoder models' },
  { name: 'Flash Attention', func: 'flash_attention', category: 'efficient', description: 'Memory-efficient via tiling', complexity: 'O(n²)', useCase: 'Long sequences, memory constrained' },
  { name: 'Sparse Attention', func: 'sparse_attention', category: 'efficient', description: 'Attend to subset of positions', complexity: 'O(n√n)', useCase: 'Very long sequences' },
  { name: 'Linear Attention', func: 'linear_attention', category: 'efficient', description: 'Kernel-based linear complexity', complexity: 'O(n)', useCase: 'Real-time applications' },
  { name: 'Local Attention', func: 'local_attention', category: 'efficient', description: 'Sliding window attention', complexity: 'O(n·w)', useCase: 'Long documents' },
  { name: 'Causal Attention', func: 'causal_attention', category: 'causal', description: 'Masked for autoregressive', complexity: 'O(n²)', useCase: 'Language generation' },
  { name: 'Global Attention', func: 'global_attention', category: 'specialized', description: 'Full attention on special tokens', complexity: 'O(n·g)', useCase: 'Document classification' },
  { name: 'Additive Attention', func: 'additive_attention', category: 'specialized', description: 'Bahdanau-style attention', complexity: 'O(n²)', useCase: 'Seq2seq models' },
  { name: 'Multiplicative', func: 'multiplicative_attention', category: 'specialized', description: 'Luong-style attention', complexity: 'O(n²)', useCase: 'Machine translation' },
]

const categories = [
  { id: 'all', label: 'All', count: attentionTypes.length },
  { id: 'core', label: 'Core', count: attentionTypes.filter(a => a.category === 'core').length },
  { id: 'efficient', label: 'Efficient', count: attentionTypes.filter(a => a.category === 'efficient').length },
  { id: 'specialized', label: 'Specialized', count: attentionTypes.filter(a => a.category === 'specialized').length },
  { id: 'causal', label: 'Causal', count: attentionTypes.filter(a => a.category === 'causal').length },
]

// Attention visualization component
const AttentionViz = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setFrame(f => (f + 1) % 60)
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
    const size = 8
    const gap = 4
    const gridSize = size + gap

    ctx.clearRect(0, 0, width, height)

    // Draw attention matrix visualization
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        const x = j * gridSize + 10
        const y = i * gridSize + 10

        // Animated attention weights
        const wave = Math.sin((i + j + frame * 0.1) * 0.5) * 0.5 + 0.5
        const alpha = wave * 0.8 + 0.2

        ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`
        ctx.fillRect(x, y, size, size)
      }
    }
  }, [frame])

  return <canvas ref={canvasRef} className="w-24 h-24" />
}

const AttentionMechanismsPage: NextPageWithLayout = () => {
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [copiedFunc, setCopiedFunc] = useState<string | null>(null)
  const [selectedFunc, setSelectedFunc] = useState<AttentionType | null>(null)

  const filteredAttention = attentionTypes.filter(a => {
    const matchesCategory = selectedCategory === 'all' || a.category === selectedCategory
    const matchesSearch = a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          a.func.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesCategory && matchesSearch
  })

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedFunc(text)
    setTimeout(() => setCopiedFunc(null), 2000)
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'core': return 'bg-blue-500/10 text-blue-500 border-blue-500/20'
      case 'efficient': return 'bg-green-500/10 text-green-500 border-green-500/20'
      case 'specialized': return 'bg-purple-500/10 text-purple-500 border-purple-500/20'
      case 'causal': return 'bg-orange-500/10 text-orange-500 border-orange-500/20'
      default: return 'bg-gray-500/10 text-gray-500 border-gray-500/20'
    }
  }

  return (
    <div className="w-full h-full overflow-y-auto">
      <div className="px-6 py-8">
        <div className="mx-auto max-w-7xl space-y-8">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-3xl font-bold text-foreground">Attention Mechanisms</h1>
              </div>
              <p className="text-foreground-light">
                39 attention mechanisms implemented as PostgreSQL functions for in-database transformer computations
              </p>
            </div>
            <AttentionViz />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Functions', value: '39', icon: Brain, color: 'text-purple-500' },
              { label: 'Core Mechanisms', value: '4', icon: Cpu, color: 'text-blue-500' },
              { label: 'Efficient Variants', value: '4', icon: Zap, color: 'text-green-500' },
              { label: 'Avg Execution', value: '<1ms', icon: Clock, color: 'text-orange-500' },
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

          {/* Search and Filters */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-foreground-light" />
              <input
                type="text"
                placeholder="Search attention mechanisms..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-lg bg-surface-100 border border-default text-foreground"
              />
            </div>
            <div className="flex gap-2">
              {categories.map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedCategory === cat.id
                      ? 'bg-brand-500 text-white'
                      : 'bg-surface-100 text-foreground-light hover:bg-surface-200'
                  }`}
                >
                  {cat.label}
                  <span className="ml-1.5 opacity-70">({cat.count})</span>
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Function List */}
            <div className="lg:col-span-2 space-y-3">
              {filteredAttention.map((attention) => (
                <div
                  key={attention.func}
                  onClick={() => setSelectedFunc(attention)}
                  className={`rounded-xl border bg-surface-100 p-4 cursor-pointer transition-all hover:border-brand-500/50 ${
                    selectedFunc?.func === attention.func ? 'border-brand-500 ring-1 ring-brand-500/20' : 'border-default'
                  }`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-foreground">{attention.name}</h3>
                        <span className={`px-2 py-0.5 rounded text-xs font-medium border ${getCategoryColor(attention.category)}`}>
                          {attention.category}
                        </span>
                      </div>
                      <p className="text-sm text-foreground-light mb-2">{attention.description}</p>
                      <div className="flex items-center gap-4 text-xs text-foreground-light">
                        <span className="flex items-center gap-1">
                          <Activity className="w-3 h-3" />
                          {attention.complexity}
                        </span>
                        <span className="flex items-center gap-1">
                          <Layers className="w-3 h-3" />
                          {attention.useCase}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        copyToClipboard(attention.func)
                      }}
                      className="p-2 rounded-lg bg-surface-200 hover:bg-surface-300 transition-colors"
                    >
                      {copiedFunc === attention.func ? (
                        <Check className="w-4 h-4 text-green-500" />
                      ) : (
                        <Copy className="w-4 h-4 text-foreground-light" />
                      )}
                    </button>
                  </div>
                  <div className="mt-3 pt-3 border-t border-default">
                    <code className="text-xs font-mono text-brand-500">{attention.func}(query, key, value)</code>
                  </div>
                </div>
              ))}
            </div>

            {/* Detail Panel */}
            <div className="space-y-4">
              {selectedFunc ? (
                <>
                  <div className="rounded-xl border border-default bg-surface-100 p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <Sparkles className="w-5 h-5 text-brand-500" />
                      <h3 className="text-lg font-semibold text-foreground">{selectedFunc.name}</h3>
                    </div>
                    <p className="text-sm text-foreground-light mb-4">{selectedFunc.description}</p>

                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-foreground-light">Complexity</span>
                        <span className="font-mono text-foreground">{selectedFunc.complexity}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-foreground-light">Category</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-medium border ${getCategoryColor(selectedFunc.category)}`}>
                          {selectedFunc.category}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-foreground-light">Use Case</span>
                        <span className="text-foreground">{selectedFunc.useCase}</span>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-default bg-surface-100 p-6">
                    <h4 className="text-sm font-semibold text-foreground mb-3">Example Usage</h4>
                    <pre className="bg-surface-200 rounded-lg p-4 text-xs font-mono text-foreground-light overflow-x-auto">
{`SELECT ${selectedFunc.func}(
  query_vector,
  key_vector,
  value_vector
) FROM embeddings
WHERE id = 1;`}
                    </pre>
                    <button
                      onClick={() => copyToClipboard(`SELECT ${selectedFunc.func}(query_vector, key_vector, value_vector) FROM embeddings WHERE id = 1;`)}
                      className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-brand-500 text-white text-sm font-medium hover:opacity-90"
                    >
                      <Copy className="w-4 h-4" />
                      Copy SQL
                    </button>
                  </div>
                </>
              ) : (
                <div className="rounded-xl border border-dashed border-default bg-surface-100/50 p-8 text-center">
                  <Brain className="w-12 h-12 text-foreground-light mx-auto mb-3 opacity-50" />
                  <p className="text-sm text-foreground-light">
                    Select an attention mechanism to view details
                  </p>
                </div>
              )}

              {/* Quick Reference */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h4 className="text-sm font-semibold text-foreground mb-3">Quick Reference</h4>
                <div className="space-y-2">
                  {[
                    { label: 'Best for Speed', value: 'linear_attention' },
                    { label: 'Best for Memory', value: 'flash_attention' },
                    { label: 'Best for Quality', value: 'multi_head_attention' },
                    { label: 'Best for Generation', value: 'causal_attention' },
                  ].map((item) => (
                    <div key={item.label} className="flex justify-between items-center text-sm">
                      <span className="text-foreground-light">{item.label}</span>
                      <code className="text-xs font-mono text-brand-500">{item.value}</code>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

AttentionMechanismsPage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default AttentionMechanismsPage
