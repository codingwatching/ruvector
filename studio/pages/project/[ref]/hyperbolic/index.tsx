import { useState, useEffect, useRef } from 'react'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'
import {
  Circle,
  GitBranch,
  Layers,
  Play,
  Copy,
  Check,
  Activity,
  ArrowRight,
  Zap,
  Target,
  TreePine,
  Network,
  BookOpen,
  Building,
} from 'lucide-react'

// Animated Poincaré disk visualization
const PoincareDiskViz = () => {
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
    const radius = Math.min(width, height) / 2 - 10

    ctx.clearRect(0, 0, width, height)

    // Draw boundary circle
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.5)'
    ctx.lineWidth = 2
    ctx.stroke()

    // Draw geodesic lines (hyperbolic lines appear as arcs)
    for (let i = 0; i < 5; i++) {
      const angle = (i / 5) * Math.PI + frame * 0.005
      ctx.beginPath()
      ctx.arc(
        centerX + Math.cos(angle) * radius * 1.5,
        centerY + Math.sin(angle) * radius * 1.5,
        radius * 1.2,
        Math.PI + angle - 0.5,
        Math.PI + angle + 0.5
      )
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.2)'
      ctx.lineWidth = 1
      ctx.stroke()
    }

    // Draw hierarchical points (closer to center = higher in hierarchy)
    const hierarchy = [
      { x: 0, y: 0, level: 0 }, // root
      { x: -0.3, y: -0.3, level: 1 },
      { x: 0.3, y: -0.3, level: 1 },
      { x: -0.5, y: 0.2, level: 2 },
      { x: -0.2, y: 0.4, level: 2 },
      { x: 0.2, y: 0.4, level: 2 },
      { x: 0.5, y: 0.2, level: 2 },
      { x: -0.6, y: 0.5, level: 3 },
      { x: -0.4, y: 0.6, level: 3 },
      { x: 0.4, y: 0.6, level: 3 },
      { x: 0.6, y: 0.5, level: 3 },
    ]

    // Draw edges
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.3)'
    ctx.lineWidth = 1
    const edges = [[0,1],[0,2],[1,3],[1,4],[2,5],[2,6],[3,7],[4,8],[5,9],[6,10]]
    edges.forEach(([from, to]) => {
      const p1 = hierarchy[from]
      const p2 = hierarchy[to]
      ctx.beginPath()
      ctx.moveTo(centerX + p1.x * radius * 0.9, centerY + p1.y * radius * 0.9)
      ctx.lineTo(centerX + p2.x * radius * 0.9, centerY + p2.y * radius * 0.9)
      ctx.stroke()
    })

    // Draw nodes
    hierarchy.forEach((point, i) => {
      const x = centerX + point.x * radius * 0.9
      const y = centerY + point.y * radius * 0.9
      const size = 6 - point.level
      const pulse = Math.sin(frame * 0.05 + i * 0.5) * 0.2 + 0.8

      ctx.beginPath()
      ctx.arc(x, y, size, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(139, 92, 246, ${pulse})`
      ctx.fill()
    })
  }, [frame])

  return <canvas ref={canvasRef} className="w-36 h-36" />
}

const HyperbolicEmbeddingsPage: NextPageWithLayout = () => {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'models' | 'operations' | 'usecases'>('models')

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const models = [
    {
      name: 'Poincaré Ball',
      icon: Circle,
      color: 'from-violet-500 to-purple-500',
      bgColor: 'bg-violet-500/10',
      borderColor: 'border-violet-500/20',
      description: 'Conformal model of hyperbolic space ideal for representing tree-like hierarchies',
      curvature: 'Negative (K = -1)',
      advantages: ['Preserves angles', 'Natural for hierarchies', 'Efficient distance'],
      operations: [
        { name: 'poincare_distance', desc: 'Compute hyperbolic distance between two points' },
        { name: 'poincare_exp_map', desc: 'Map tangent vectors to the Poincaré ball' },
        { name: 'poincare_log_map', desc: 'Map points back to tangent space' },
        { name: 'poincare_add', desc: 'Möbius addition of two points' },
      ],
    },
    {
      name: 'Lorentz Hyperboloid',
      icon: Layers,
      color: 'from-blue-500 to-cyan-500',
      bgColor: 'bg-blue-500/10',
      borderColor: 'border-blue-500/20',
      description: 'Alternative hyperbolic model with numerically stable operations',
      curvature: 'Negative (K = -1)',
      advantages: ['Numerically stable', 'Efficient optimization', 'Natural Minkowski geometry'],
      operations: [
        { name: 'lorentz_distance', desc: 'Compute distance using Minkowski inner product' },
        { name: 'lorentz_exp_map', desc: 'Project tangent vectors to hyperboloid' },
        { name: 'lorentz_log_map', desc: 'Logarithmic map to tangent space' },
        { name: 'lorentz_to_poincare', desc: 'Convert between models' },
      ],
    },
  ]

  const useCases = [
    {
      title: 'Taxonomy Embeddings',
      icon: TreePine,
      description: 'Embed hierarchical taxonomies with fewer dimensions than Euclidean space',
      example: 'WordNet, product categories, biological taxonomies',
    },
    {
      title: 'Knowledge Graphs',
      icon: Network,
      description: 'Capture hierarchical relationships in knowledge bases naturally',
      example: 'Entity hierarchies, type systems, ontologies',
    },
    {
      title: 'Organizational Charts',
      icon: Building,
      description: 'Represent company structures preserving management hierarchies',
      example: 'Employee reporting chains, department structures',
    },
    {
      title: 'Document Hierarchies',
      icon: BookOpen,
      description: 'Model document structures from sections to paragraphs',
      example: 'Legal documents, technical manuals, wikis',
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
                <div className="p-2 rounded-lg bg-gradient-to-br from-violet-500 to-purple-500">
                  <Circle className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-3xl font-bold text-foreground">Hyperbolic Embeddings</h1>
              </div>
              <p className="text-foreground-light">
                Poincaré and Lorentz model operations for hierarchical data representation
              </p>
            </div>
            <PoincareDiskViz />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { label: 'Models', value: '2', icon: Circle, color: 'text-violet-500' },
              { label: 'Operations', value: '8', icon: Activity, color: 'text-blue-500' },
              { label: 'Dimension Savings', value: '10x', icon: Zap, color: 'text-green-500' },
              { label: 'Hierarchy Depth', value: 'Unlimited', icon: GitBranch, color: 'text-orange-500' },
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
              { id: 'models', label: 'Models', icon: Circle },
              { id: 'operations', label: 'Operations', icon: Activity },
              { id: 'usecases', label: 'Use Cases', icon: Target },
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

          {activeTab === 'models' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {models.map((model) => (
                <div key={model.name} className={`rounded-xl border ${model.borderColor} ${model.bgColor} p-6 space-y-4`}>
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg bg-gradient-to-br ${model.color}`}>
                      <model.icon className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-foreground">{model.name}</h3>
                      <p className="text-sm text-foreground-light">{model.description}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-surface-200/50 p-3">
                      <div className="text-xs text-foreground-light mb-1">Curvature</div>
                      <div className="text-sm font-mono text-foreground">{model.curvature}</div>
                    </div>
                    <div className="rounded-lg bg-surface-200/50 p-3">
                      <div className="text-xs text-foreground-light mb-1">Advantages</div>
                      <div className="text-sm text-foreground">{model.advantages[0]}</div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-foreground">Key Operations</h4>
                    {model.operations.map((op) => (
                      <div key={op.name} className="flex items-center justify-between p-2 rounded-lg bg-surface-200/50">
                        <div>
                          <code className="text-xs font-mono text-brand-500">{op.name}()</code>
                          <p className="text-xs text-foreground-light">{op.desc}</p>
                        </div>
                        <button
                          onClick={() => copyToClipboard(`SELECT ${op.name}(v1, v2) FROM embeddings;`, op.name)}
                          className="p-1.5 rounded bg-surface-300 hover:bg-surface-400 transition-colors"
                        >
                          {copiedCode === op.name ? (
                            <Check className="w-3 h-3 text-green-500" />
                          ) : (
                            <Copy className="w-3 h-3 text-foreground-light" />
                          )}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'operations' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Distance Operations */}
                <div className="rounded-xl border border-default bg-surface-100 p-6">
                  <h3 className="text-lg font-semibold text-foreground mb-4">Distance Operations</h3>
                  <div className="space-y-3">
                    {[
                      { func: 'poincare_distance', args: 'v1 vector, v2 vector', returns: 'float8' },
                      { func: 'lorentz_distance', args: 'v1 vector, v2 vector', returns: 'float8' },
                    ].map((op) => (
                      <div key={op.func} className="p-4 rounded-lg bg-surface-200">
                        <code className="text-sm font-mono text-brand-500">{op.func}({op.args})</code>
                        <div className="flex items-center gap-2 mt-2 text-xs text-foreground-light">
                          <ArrowRight className="w-3 h-3" />
                          Returns: {op.returns}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Mapping Operations */}
                <div className="rounded-xl border border-default bg-surface-100 p-6">
                  <h3 className="text-lg font-semibold text-foreground mb-4">Mapping Operations</h3>
                  <div className="space-y-3">
                    {[
                      { func: 'poincare_exp_map', args: 'point vector, tangent vector', returns: 'vector' },
                      { func: 'poincare_log_map', args: 'point vector, target vector', returns: 'vector' },
                      { func: 'lorentz_exp_map', args: 'point vector, tangent vector', returns: 'vector' },
                      { func: 'lorentz_log_map', args: 'point vector, target vector', returns: 'vector' },
                    ].map((op) => (
                      <div key={op.func} className="p-4 rounded-lg bg-surface-200">
                        <code className="text-sm font-mono text-brand-500">{op.func}({op.args})</code>
                        <div className="flex items-center gap-2 mt-2 text-xs text-foreground-light">
                          <ArrowRight className="w-3 h-3" />
                          Returns: {op.returns}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Example Query */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Example: Hierarchical Search</h3>
                <pre className="bg-surface-200 rounded-lg p-4 text-xs font-mono text-foreground-light overflow-x-auto">
{`-- Find all descendants within hyperbolic distance threshold
SELECT
  child.id,
  child.name,
  poincare_distance(parent.embedding, child.embedding) as distance
FROM taxonomy parent
JOIN taxonomy child ON child.parent_id = parent.id
WHERE parent.id = 1
  AND poincare_distance(parent.embedding, child.embedding) < 2.0
ORDER BY distance;`}
                </pre>
              </div>
            </div>
          )}

          {activeTab === 'usecases' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {useCases.map((uc) => (
                <div key={uc.title} className="rounded-xl border border-default bg-surface-100 p-6">
                  <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-violet-500/10">
                      <uc.icon className="w-6 h-6 text-violet-500" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-foreground mb-2">{uc.title}</h3>
                      <p className="text-sm text-foreground-light mb-3">{uc.description}</p>
                      <div className="flex items-center gap-2 text-xs">
                        <span className="text-foreground-light">Examples:</span>
                        <span className="text-foreground">{uc.example}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Why Hyperbolic */}
              <div className="md:col-span-2 rounded-xl border border-violet-500/20 bg-violet-500/5 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Why Hyperbolic Space?</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    { title: '10x Fewer Dimensions', desc: 'Trees with N nodes need only O(log N) dimensions in hyperbolic space' },
                    { title: 'Natural Hierarchy', desc: 'Distance from origin encodes hierarchy level naturally' },
                    { title: 'Better Embeddings', desc: 'Preserves hierarchical structure that Euclidean space distorts' },
                  ].map((benefit) => (
                    <div key={benefit.title} className="text-center p-4">
                      <h4 className="font-medium text-foreground mb-2">{benefit.title}</h4>
                      <p className="text-xs text-foreground-light">{benefit.desc}</p>
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

HyperbolicEmbeddingsPage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default HyperbolicEmbeddingsPage
