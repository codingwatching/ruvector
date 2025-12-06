import { useState, useEffect, useRef } from 'react'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'
import {
  Brain,
  TrendingUp,
  Target,
  Sparkles,
  Play,
  Copy,
  Check,
  Activity,
  BarChart3,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  Layers,
  Zap,
  ArrowRight,
} from 'lucide-react'

// Learning progress visualization
const LearningProgressViz = ({ progress }: { progress: number }) => {
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
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) / 2 - 15

    ctx.clearRect(0, 0, width, height)

    // Background circle
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.1)'
    ctx.lineWidth = 8
    ctx.stroke()

    // Progress arc
    const startAngle = -Math.PI / 2
    const endAngle = startAngle + (progress / 100) * Math.PI * 2

    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, startAngle, endAngle)
    const gradient = ctx.createLinearGradient(0, 0, width, height)
    gradient.addColorStop(0, '#3b82f6')
    gradient.addColorStop(1, '#8b5cf6')
    ctx.strokeStyle = gradient
    ctx.lineWidth = 8
    ctx.lineCap = 'round'
    ctx.stroke()

    // Center text
    ctx.fillStyle = '#fff'
    ctx.font = 'bold 20px system-ui'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`${progress}%`, centerX, centerY - 5)
    ctx.font = '10px system-ui'
    ctx.fillStyle = 'rgba(255,255,255,0.6)'
    ctx.fillText('accuracy', centerX, centerY + 12)
  }, [progress])

  return <canvas ref={canvasRef} className="w-24 h-24" />
}

// Simulated learning stats
const useLearningStats = () => {
  const [stats, setStats] = useState({
    trajectories: 1247,
    successRate: 78,
    patterns: 156,
    avgImprovement: 12.3,
  })

  useEffect(() => {
    const interval = setInterval(() => {
      setStats(prev => ({
        ...prev,
        trajectories: prev.trajectories + Math.floor(Math.random() * 3),
        successRate: Math.min(95, prev.successRate + (Math.random() > 0.7 ? 0.1 : 0)),
      }))
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  return stats
}

const SelfLearningPage: NextPageWithLayout = () => {
  const stats = useLearningStats()
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'pipeline' | 'functions' | 'monitor'>('pipeline')
  const [selectedStage, setSelectedStage] = useState<number | null>(null)

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const pipeline = [
    {
      stage: 1,
      name: 'Trajectory',
      icon: Activity,
      color: 'from-blue-500 to-cyan-500',
      description: 'Record decision-making sequences and their contexts',
      func: 'record_trajectory',
      details: 'Captures state, action, and outcome triplets for later analysis',
    },
    {
      stage: 2,
      name: 'Verdict',
      icon: Target,
      color: 'from-purple-500 to-pink-500',
      description: 'Evaluate outcomes and mark trajectories as successful or not',
      func: 'evaluate_verdict',
      details: 'Binary or scored evaluation of trajectory outcomes',
    },
    {
      stage: 3,
      name: 'Distillation',
      icon: Sparkles,
      color: 'from-orange-500 to-amber-500',
      description: 'Extract reusable patterns from successful trajectories',
      func: 'distill_patterns',
      details: 'Identifies common patterns in successful decision sequences',
    },
    {
      stage: 4,
      name: 'Update',
      icon: RefreshCw,
      color: 'from-green-500 to-emerald-500',
      description: 'Apply learned patterns to improve future decisions',
      func: 'get_recommendations',
      details: 'Returns relevant patterns for new decision contexts',
    },
  ]

  const functions = [
    {
      name: 'record_trajectory',
      signature: '(state jsonb, action text, outcome jsonb)',
      returns: 'trajectory_id bigint',
      description: 'Log a decision-making sequence for later analysis',
    },
    {
      name: 'evaluate_verdict',
      signature: '(trajectory_id bigint, success boolean, score float DEFAULT NULL)',
      returns: 'void',
      description: 'Mark a trajectory as successful or unsuccessful with optional score',
    },
    {
      name: 'distill_patterns',
      signature: '(min_confidence float DEFAULT 0.7, min_occurrences int DEFAULT 3)',
      returns: 'TABLE(pattern_id, pattern jsonb, confidence float)',
      description: 'Extract reusable patterns from successful trajectories',
    },
    {
      name: 'get_recommendations',
      signature: '(current_state jsonb, top_k int DEFAULT 5)',
      returns: 'TABLE(pattern_id, action text, confidence float)',
      description: 'Get action recommendations based on learned patterns',
    },
  ]

  const recentTrajectories = [
    { id: 1234, state: 'user_query', action: 'route_to_agent_a', success: true, confidence: 0.92 },
    { id: 1233, state: 'classification', action: 'category_tech', success: true, confidence: 0.88 },
    { id: 1232, state: 'search_query', action: 'semantic_search', success: false, confidence: 0.65 },
    { id: 1231, state: 'recommendation', action: 'collaborative_filter', success: true, confidence: 0.91 },
    { id: 1230, state: 'intent_detect', action: 'purchase_intent', success: true, confidence: 0.85 },
  ]

  return (
    <div className="w-full h-full overflow-y-auto">
      <div className="px-6 py-8">
        <div className="mx-auto max-w-7xl space-y-8">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-lg bg-gradient-to-br from-green-500 to-emerald-500">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-3xl font-bold text-foreground">Self-Learning</h1>
                <span className="px-2 py-1 rounded-full bg-green-500/10 text-green-500 text-xs font-medium">
                  ReasoningBank
                </span>
              </div>
              <p className="text-foreground-light">
                Adaptive learning system that improves through experience
              </p>
            </div>
            <LearningProgressViz progress={stats.successRate} />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[
              { label: 'Trajectories', value: stats.trajectories.toLocaleString(), icon: Activity, color: 'text-blue-500', live: true },
              { label: 'Success Rate', value: `${stats.successRate.toFixed(1)}%`, icon: Target, color: 'text-green-500', live: true },
              { label: 'Patterns', value: stats.patterns, icon: Sparkles, color: 'text-purple-500', live: false },
              { label: 'Avg Improvement', value: `+${stats.avgImprovement}%`, icon: TrendingUp, color: 'text-orange-500', live: false },
            ].map((stat, i) => (
              <div key={i} className="rounded-xl border border-default bg-surface-100 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-foreground-light">{stat.label}</span>
                  <div className="flex items-center gap-2">
                    {stat.live && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
                    <stat.icon className={`w-4 h-4 ${stat.color}`} />
                  </div>
                </div>
                <div className="text-2xl font-bold text-foreground">{stat.value}</div>
              </div>
            ))}
          </div>

          {/* Tabs */}
          <div className="flex gap-2 border-b border-default">
            {[
              { id: 'pipeline', label: 'Learning Pipeline', icon: Layers },
              { id: 'functions', label: 'Functions', icon: Zap },
              { id: 'monitor', label: 'Monitor', icon: BarChart3 },
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

          {activeTab === 'pipeline' && (
            <div className="space-y-6">
              {/* Pipeline Visualization */}
              <div className="rounded-xl border border-default bg-surface-100 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-6">4-Stage Learning Pipeline</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {pipeline.map((stage, i) => (
                    <div key={stage.stage} className="relative">
                      <div
                        onClick={() => setSelectedStage(selectedStage === stage.stage ? null : stage.stage)}
                        className={`rounded-xl p-4 cursor-pointer transition-all ${
                          selectedStage === stage.stage
                            ? 'ring-2 ring-brand-500 bg-surface-200'
                            : 'bg-surface-200/50 hover:bg-surface-200'
                        }`}
                      >
                        <div className="text-center">
                          <div className={`mx-auto w-14 h-14 rounded-xl bg-gradient-to-br ${stage.color} flex items-center justify-center mb-3`}>
                            <stage.icon className="w-7 h-7 text-white" />
                          </div>
                          <div className="text-xs text-foreground-light mb-1">Stage {stage.stage}</div>
                          <h4 className="font-semibold text-foreground">{stage.name}</h4>
                          <p className="text-xs text-foreground-light mt-2">{stage.description}</p>
                        </div>
                      </div>
                      {i < pipeline.length - 1 && (
                        <div className="hidden md:block absolute top-1/2 -right-2 transform -translate-y-1/2">
                          <ArrowRight className="w-4 h-4 text-foreground-light" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Selected Stage Details */}
              {selectedStage && (
                <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 p-6">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${pipeline[selectedStage - 1].color}`}>
                      {(() => {
                        const Icon = pipeline[selectedStage - 1].icon
                        return <Icon className="w-6 h-6 text-white" />
                      })()}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-foreground mb-2">
                        {pipeline[selectedStage - 1].name}
                      </h3>
                      <p className="text-foreground-light mb-4">{pipeline[selectedStage - 1].details}</p>
                      <div className="relative">
                        <pre className="bg-surface-200 rounded-lg p-4 text-xs font-mono text-foreground-light">
{`SELECT ${pipeline[selectedStage - 1].func}(
  current_state,
  'action_taken',
  outcome_data
) FROM decisions;`}
                        </pre>
                        <button
                          onClick={() => copyToClipboard(`SELECT ${pipeline[selectedStage - 1].func}(...)`, pipeline[selectedStage - 1].func)}
                          className="absolute top-2 right-2 p-1.5 rounded bg-surface-300 hover:bg-surface-400"
                        >
                          {copiedCode === pipeline[selectedStage - 1].func ? (
                            <Check className="w-3.5 h-3.5 text-green-500" />
                          ) : (
                            <Copy className="w-3.5 h-3.5 text-foreground-light" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Benefits */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { icon: RefreshCw, title: 'Continuous Improvement', desc: 'Learns from production data automatically' },
                  { icon: Target, title: 'No Manual Labeling', desc: 'Uses outcome feedback, not manual labels' },
                  { icon: TrendingUp, title: 'Adaptive', desc: 'Adjusts to changing patterns over time' },
                  { icon: CheckCircle, title: 'Dual Learning', desc: 'Learns from both successes and failures' },
                ].map((benefit) => (
                  <div key={benefit.title} className="rounded-xl border border-default bg-surface-100 p-4">
                    <benefit.icon className="w-5 h-5 text-green-500 mb-2" />
                    <h4 className="font-medium text-foreground text-sm mb-1">{benefit.title}</h4>
                    <p className="text-xs text-foreground-light">{benefit.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'functions' && (
            <div className="space-y-4">
              {functions.map((func) => (
                <div key={func.name} className="rounded-xl border border-default bg-surface-100 p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <code className="text-lg font-mono text-brand-500">{func.name}</code>
                      <div className="text-sm font-mono text-foreground-light mt-1">{func.signature}</div>
                      <p className="text-sm text-foreground-light mt-3">{func.description}</p>
                      <div className="flex items-center gap-2 mt-3 text-xs text-foreground-light">
                        <ArrowRight className="w-3 h-3" />
                        Returns: <code className="text-foreground">{func.returns}</code>
                      </div>
                    </div>
                    <button
                      onClick={() => copyToClipboard(`SELECT ${func.name}${func.signature};`, func.name)}
                      className="p-2 rounded-lg bg-surface-200 hover:bg-surface-300"
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
          )}

          {activeTab === 'monitor' && (
            <div className="space-y-6">
              {/* Recent Trajectories */}
              <div className="rounded-xl border border-default bg-surface-100 overflow-hidden">
                <div className="px-6 py-4 border-b border-default">
                  <h3 className="text-lg font-semibold text-foreground">Recent Trajectories</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-surface-200">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">ID</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">State</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Action</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Confidence</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-foreground-light uppercase">Result</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-default">
                      {recentTrajectories.map((t) => (
                        <tr key={t.id} className="hover:bg-surface-200 transition-colors">
                          <td className="px-6 py-4 text-sm font-mono text-foreground">#{t.id}</td>
                          <td className="px-6 py-4 text-sm text-foreground">{t.state}</td>
                          <td className="px-6 py-4">
                            <code className="text-xs bg-surface-200 px-2 py-1 rounded text-foreground">{t.action}</code>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-2 bg-surface-200 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-brand-500 rounded-full"
                                  style={{ width: `${t.confidence * 100}%` }}
                                />
                              </div>
                              <span className="text-xs text-foreground-light">{(t.confidence * 100).toFixed(0)}%</span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            {t.success ? (
                              <span className="flex items-center gap-1 text-green-500 text-sm">
                                <CheckCircle className="w-4 h-4" />
                                Success
                              </span>
                            ) : (
                              <span className="flex items-center gap-1 text-red-500 text-sm">
                                <XCircle className="w-4 h-4" />
                                Failed
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Pattern Stats */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="rounded-xl border border-default bg-surface-100 p-6">
                  <h3 className="text-lg font-semibold text-foreground mb-4">Top Learned Patterns</h3>
                  <div className="space-y-3">
                    {[
                      { pattern: 'user_query → route_to_expert', confidence: 0.94, uses: 342 },
                      { pattern: 'search_miss → fallback_semantic', confidence: 0.89, uses: 156 },
                      { pattern: 'high_intent → priority_queue', confidence: 0.87, uses: 98 },
                    ].map((p, i) => (
                      <div key={i} className="p-3 rounded-lg bg-surface-200">
                        <div className="flex justify-between items-center mb-2">
                          <code className="text-xs font-mono text-brand-500">{p.pattern}</code>
                          <span className="text-xs text-foreground-light">{p.uses} uses</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-surface-300 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-green-500 rounded-full"
                              style={{ width: `${p.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-foreground">{(p.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-xl border border-default bg-surface-100 p-6">
                  <h3 className="text-lg font-semibold text-foreground mb-4">Learning Activity</h3>
                  <div className="space-y-4">
                    {[
                      { label: 'Trajectories Today', value: 156, change: '+12%' },
                      { label: 'New Patterns', value: 3, change: '+2' },
                      { label: 'Success Rate Δ', value: '+2.3%', change: 'improving' },
                      { label: 'Distillation Runs', value: 4, change: 'auto' },
                    ].map((item) => (
                      <div key={item.label} className="flex justify-between items-center">
                        <span className="text-sm text-foreground-light">{item.label}</span>
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-foreground">{item.value}</span>
                          <span className="text-xs text-green-500">{item.change}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

SelfLearningPage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default SelfLearningPage
