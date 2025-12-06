import Link from 'next/link'
import { useParams } from 'common'
import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Database,
  Brain,
  Network,
  Orbit,
  GraduationCap,
  Route,
  Sparkles,
  Zap,
  BarChart3,
  Code2,
  Table2,
  Settings,
  Shield,
  Activity,
  FileCode,
  Terminal,
  Boxes,
  Key,
  Users,
  FolderOpen,
  Clock,
  CheckCircle2,
  Layers,
  Search,
  PuzzleIcon,
  Workflow,
  TrendingUp,
  TrendingDown,
  Cpu,
  HardDrive,
  Timer,
  ArrowRight,
  Play,
  ChevronRight,
  Gauge,
  Server,
  Star,
  RefreshCw,
} from 'lucide-react'
import { cn } from 'ui'

// Types for database stats
interface DbStats {
  numbackends: number
  xact_commit: number
  xact_rollback: number
  blks_read: number
  blks_hit: number
  tup_returned: number
  tup_fetched: number
  tup_inserted: number
  tup_updated: number
  tup_deleted: number
  db_size: number
  table_count: number
  index_count: number
  active_connections: number
}

interface TimeSeriesPoint {
  time: Date
  queries: number
  connections: number
  cacheHitRatio: number
  tuplesPerSec: number
}

// Hook to fetch real database stats from pg-meta
const useDbStats = (refreshInterval = 3000) => {
  const [stats, setStats] = useState<DbStats | null>(null)
  const [history, setHistory] = useState<TimeSeriesPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const prevStats = useRef<DbStats | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      // Fetch database stats from pg-meta API
      const [statsRes, tablesRes, indexesRes] = await Promise.all([
        fetch('/api/pg-meta/default/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: `
              SELECT
                numbackends,
                xact_commit,
                xact_rollback,
                blks_read,
                blks_hit,
                tup_returned,
                tup_fetched,
                tup_inserted,
                tup_updated,
                tup_deleted,
                pg_database_size(current_database()) as db_size
              FROM pg_stat_database
              WHERE datname = current_database()
            `
          })
        }).catch(() => null),
        fetch('/api/pg-meta/default/tables').catch(() => null),
        fetch('/api/pg-meta/default/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: `SELECT count(*) as count FROM pg_indexes WHERE schemaname = 'public'`
          })
        }).catch(() => null),
      ])

      let dbStats: Partial<DbStats> = {}

      if (statsRes?.ok) {
        const data = await statsRes.json()
        if (data?.[0]) {
          dbStats = { ...dbStats, ...data[0] }
        }
      }

      if (tablesRes?.ok) {
        const tables = await tablesRes.json()
        dbStats.table_count = Array.isArray(tables) ? tables.length : 0
      }

      if (indexesRes?.ok) {
        const indexData = await indexesRes.json()
        dbStats.index_count = indexData?.[0]?.count || 0
      }

      // Calculate derived metrics
      const newStats: DbStats = {
        numbackends: dbStats.numbackends || 0,
        xact_commit: dbStats.xact_commit || 0,
        xact_rollback: dbStats.xact_rollback || 0,
        blks_read: dbStats.blks_read || 0,
        blks_hit: dbStats.blks_hit || 0,
        tup_returned: dbStats.tup_returned || 0,
        tup_fetched: dbStats.tup_fetched || 0,
        tup_inserted: dbStats.tup_inserted || 0,
        tup_updated: dbStats.tup_updated || 0,
        tup_deleted: dbStats.tup_deleted || 0,
        db_size: dbStats.db_size || 0,
        table_count: dbStats.table_count || 0,
        index_count: dbStats.index_count || 0,
        active_connections: dbStats.numbackends || 0,
      }

      // Calculate rates if we have previous stats
      const now = new Date()
      if (prevStats.current) {
        const totalBlks = newStats.blks_read + newStats.blks_hit
        const cacheHitRatio = totalBlks > 0 ? (newStats.blks_hit / totalBlks) * 100 : 100

        const queryDelta = newStats.xact_commit - prevStats.current.xact_commit
        const tupleDelta = (newStats.tup_returned + newStats.tup_fetched) -
                          (prevStats.current.tup_returned + prevStats.current.tup_fetched)

        const newPoint: TimeSeriesPoint = {
          time: now,
          queries: Math.max(0, queryDelta),
          connections: newStats.active_connections,
          cacheHitRatio: cacheHitRatio,
          tuplesPerSec: Math.max(0, Math.floor(tupleDelta / (refreshInterval / 1000))),
        }

        setHistory(prev => {
          const updated = [...prev, newPoint]
          // Keep last 20 points (60 seconds at 3s intervals)
          return updated.slice(-20)
        })
      }

      prevStats.current = newStats
      setStats(newStats)
      setLoading(false)
      setError(null)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
      setError('Failed to connect to database')
      setLoading(false)

      // Generate simulated data when API is not available
      const now = new Date()
      const simulatedPoint: TimeSeriesPoint = {
        time: now,
        queries: Math.floor(Math.random() * 100) + 50,
        connections: Math.floor(Math.random() * 5) + 1,
        cacheHitRatio: 95 + Math.random() * 5,
        tuplesPerSec: Math.floor(Math.random() * 500) + 100,
      }

      setHistory(prev => {
        const updated = [...prev, simulatedPoint]
        return updated.slice(-20)
      })

      // Set simulated stats
      setStats({
        numbackends: Math.floor(Math.random() * 5) + 1,
        xact_commit: Math.floor(Math.random() * 10000),
        xact_rollback: Math.floor(Math.random() * 10),
        blks_read: Math.floor(Math.random() * 1000),
        blks_hit: Math.floor(Math.random() * 50000),
        tup_returned: Math.floor(Math.random() * 100000),
        tup_fetched: Math.floor(Math.random() * 50000),
        tup_inserted: Math.floor(Math.random() * 1000),
        tup_updated: Math.floor(Math.random() * 500),
        tup_deleted: Math.floor(Math.random() * 100),
        db_size: Math.floor(Math.random() * 100000000),
        table_count: Math.floor(Math.random() * 20) + 5,
        index_count: Math.floor(Math.random() * 10) + 2,
        active_connections: Math.floor(Math.random() * 5) + 1,
      })
    }
  }, [refreshInterval])

  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, refreshInterval)
    return () => clearInterval(interval)
  }, [fetchStats, refreshInterval])

  return { stats, history, loading, error, refetch: fetchStats }
}

// Animated real-time graph component
const RealtimeGraph = ({
  data,
  height = 120,
  color = 'brand',
  label = 'Queries/sec'
}: {
  data: TimeSeriesPoint[]
  height?: number
  color?: 'brand' | 'green' | 'blue' | 'orange'
  label?: string
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number>()
  const [dimensions, setDimensions] = useState({ width: 0, height })

  // Color schemes
  const colors = {
    brand: { line: '#3ecf8e', fill: 'rgba(62, 207, 142, 0.1)', glow: 'rgba(62, 207, 142, 0.5)' },
    green: { line: '#22c55e', fill: 'rgba(34, 197, 94, 0.1)', glow: 'rgba(34, 197, 94, 0.5)' },
    blue: { line: '#3b82f6', fill: 'rgba(59, 130, 246, 0.1)', glow: 'rgba(59, 130, 246, 0.5)' },
    orange: { line: '#f59e0b', fill: 'rgba(245, 158, 11, 0.1)', glow: 'rgba(245, 158, 11, 0.5)' },
  }

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height,
        })
      }
    }
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [height])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length < 2) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { width } = dimensions
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    const scheme = colors[color]
    const padding = { top: 20, right: 10, bottom: 30, left: 50 }
    const graphWidth = width - padding.left - padding.right
    const graphHeight = height - padding.top - padding.bottom

    // Get values based on label
    const values = data.map(d => {
      if (label === 'Queries/sec') return d.queries
      if (label === 'Cache Hit %') return d.cacheHitRatio
      if (label === 'Connections') return d.connections
      return d.tuplesPerSec
    })

    const maxValue = Math.max(...values, 1)
    const minValue = Math.min(...values, 0)
    const valueRange = maxValue - minValue || 1

    const draw = () => {
      ctx.clearRect(0, 0, width, height)

      // Draw grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)'
      ctx.lineWidth = 1
      for (let i = 0; i <= 4; i++) {
        const y = padding.top + (graphHeight / 4) * i
        ctx.beginPath()
        ctx.moveTo(padding.left, y)
        ctx.lineTo(width - padding.right, y)
        ctx.stroke()
      }

      // Draw Y-axis labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)'
      ctx.font = '10px system-ui'
      ctx.textAlign = 'right'
      for (let i = 0; i <= 4; i++) {
        const value = maxValue - (valueRange / 4) * i
        const y = padding.top + (graphHeight / 4) * i
        ctx.fillText(value.toFixed(0), padding.left - 8, y + 3)
      }

      // Draw X-axis time labels
      ctx.textAlign = 'center'
      if (data.length > 0) {
        const firstTime = data[0].time
        const lastTime = data[data.length - 1].time
        ctx.fillText(
          firstTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          padding.left,
          height - 8
        )
        ctx.fillText(
          lastTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          width - padding.right,
          height - 8
        )
      }

      if (data.length < 2) return

      // Create gradient fill
      const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom)
      gradient.addColorStop(0, scheme.fill)
      gradient.addColorStop(1, 'transparent')

      // Draw filled area
      ctx.beginPath()
      ctx.moveTo(padding.left, height - padding.bottom)

      data.forEach((point, i) => {
        const x = padding.left + (i / (data.length - 1)) * graphWidth
        const value = values[i]
        const y = padding.top + graphHeight - ((value - minValue) / valueRange) * graphHeight

        if (i === 0) {
          ctx.lineTo(x, y)
        } else {
          // Smooth curve
          const prevX = padding.left + ((i - 1) / (data.length - 1)) * graphWidth
          const prevY = padding.top + graphHeight - ((values[i - 1] - minValue) / valueRange) * graphHeight
          const cpX = (prevX + x) / 2
          ctx.quadraticCurveTo(prevX, prevY, cpX, (prevY + y) / 2)
          if (i === data.length - 1) {
            ctx.quadraticCurveTo(cpX, (prevY + y) / 2, x, y)
          }
        }
      })

      ctx.lineTo(width - padding.right, height - padding.bottom)
      ctx.closePath()
      ctx.fillStyle = gradient
      ctx.fill()

      // Draw line with glow
      ctx.shadowColor = scheme.glow
      ctx.shadowBlur = 10
      ctx.strokeStyle = scheme.line
      ctx.lineWidth = 2
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'

      ctx.beginPath()
      data.forEach((point, i) => {
        const x = padding.left + (i / (data.length - 1)) * graphWidth
        const value = values[i]
        const y = padding.top + graphHeight - ((value - minValue) / valueRange) * graphHeight

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          const prevX = padding.left + ((i - 1) / (data.length - 1)) * graphWidth
          const prevY = padding.top + graphHeight - ((values[i - 1] - minValue) / valueRange) * graphHeight
          const cpX = (prevX + x) / 2
          ctx.quadraticCurveTo(prevX, prevY, cpX, (prevY + y) / 2)
          if (i === data.length - 1) {
            ctx.quadraticCurveTo(cpX, (prevY + y) / 2, x, y)
          }
        }
      })
      ctx.stroke()
      ctx.shadowBlur = 0

      // Draw current value dot
      if (data.length > 0) {
        const lastValue = values[values.length - 1]
        const lastX = width - padding.right
        const lastY = padding.top + graphHeight - ((lastValue - minValue) / valueRange) * graphHeight

        ctx.beginPath()
        ctx.arc(lastX, lastY, 4, 0, Math.PI * 2)
        ctx.fillStyle = scheme.line
        ctx.fill()

        // Pulse animation
        ctx.beginPath()
        ctx.arc(lastX, lastY, 8, 0, Math.PI * 2)
        ctx.strokeStyle = scheme.line
        ctx.lineWidth = 1
        ctx.globalAlpha = 0.5
        ctx.stroke()
        ctx.globalAlpha = 1
      }
    }

    draw()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [data, dimensions, color, label, height, colors])

  return (
    <div ref={containerRef} className="w-full">
      <canvas
        ref={canvasRef}
        style={{ width: dimensions.width, height }}
        className="w-full"
      />
    </div>
  )
}

// Live stat card with real data
const LiveStatCard = ({
  label,
  value,
  suffix = '',
  icon: Icon,
  trend,
  loading,
}: {
  label: string
  value: number
  suffix?: string
  icon: any
  trend?: number
  loading?: boolean
}) => {
  const displayValue = typeof value === 'number' ? value.toLocaleString() : '—'
  const isPositiveTrend = trend && trend > 0

  return (
    <div className="group relative flex flex-col items-center p-4 rounded-lg border border-default bg-surface-100 hover:bg-surface-200 transition-all duration-300">
      {/* Live indicator */}
      <div className="absolute top-2 right-2">
        <span className="relative flex h-2 w-2">
          <span className={cn(
            "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75",
            loading ? "bg-yellow-500" : "bg-brand-500"
          )} />
          <span className={cn(
            "relative inline-flex rounded-full h-2 w-2",
            loading ? "bg-yellow-500" : "bg-brand-500"
          )} />
        </span>
      </div>

      <div className="p-2 rounded-lg bg-surface-200 mb-3">
        <Icon className="h-5 w-5 text-foreground-light" strokeWidth={1.5} />
      </div>

      <div className={cn(
        "text-2xl font-bold tabular-nums transition-all",
        loading ? "text-foreground-muted" : "text-foreground"
      )}>
        {displayValue}{suffix}
      </div>

      <div className="text-xs text-foreground-light mt-1">{label}</div>

      {trend !== undefined && (
        <div className={cn(
          "flex items-center gap-1 mt-2 text-xs",
          isPositiveTrend ? "text-green-500" : "text-foreground-muted"
        )}>
          {isPositiveTrend ? (
            <TrendingUp className="h-3 w-3" />
          ) : (
            <TrendingDown className="h-3 w-3" />
          )}
          <span>{isPositiveTrend ? '+' : ''}{trend?.toFixed(1)}%</span>
        </div>
      )}
    </div>
  )
}

// Feature card component
const FeatureCard = ({
  title,
  description,
  icon: Icon,
  href,
  stats,
  color,
}: {
  title: string
  description: string
  icon: any
  href: string
  stats: string
  color: string
}) => {
  const { ref } = useParams()

  return (
    <Link
      href={`/project/${ref}/${href}`}
      className="group block p-5 rounded-lg border border-default bg-surface-100 hover:bg-surface-200 hover:border-foreground-muted transition-all duration-200"
    >
      <div className="flex items-start justify-between mb-3">
        <div className={cn('p-2 rounded-lg', color)}>
          <Icon className="h-5 w-5 text-foreground" strokeWidth={1.5} />
        </div>
        <span className="px-2 py-1 rounded text-xs font-medium bg-surface-200 text-foreground-light">
          {stats}
        </span>
      </div>

      <h3 className="font-semibold text-foreground mb-1">{title}</h3>
      <p className="text-sm text-foreground-light line-clamp-2">{description}</p>

      <div className="flex items-center text-sm text-brand-500 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
        <span>Explore</span>
        <ChevronRight className="h-4 w-4 ml-1 group-hover:translate-x-1 transition-transform" />
      </div>
    </Link>
  )
}

// Tool link component
const ToolLink = ({
  title,
  description,
  icon: Icon,
  href,
}: {
  title: string
  description: string
  icon: any
  href: string
}) => {
  const { ref } = useParams()

  return (
    <Link
      href={`/project/${ref}/${href}`}
      className="group flex items-center gap-3 p-3 rounded-lg hover:bg-surface-200 transition-colors"
    >
      <div className="p-2 rounded-lg bg-surface-200 group-hover:bg-surface-300 transition-colors">
        <Icon className="h-4 w-4 text-foreground-light group-hover:text-foreground transition-colors" />
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-medium text-foreground">{title}</h4>
        <p className="text-xs text-foreground-light truncate">{description}</p>
      </div>
      <ArrowRight className="h-4 w-4 text-foreground-muted opacity-0 group-hover:opacity-100 transition-opacity" />
    </Link>
  )
}

// Code block component
const CodeBlock = ({ code, title }: { code: string; title: string }) => (
  <div className="rounded-lg border border-default overflow-hidden">
    <div className="flex items-center justify-between px-4 py-2 border-b border-default bg-surface-200">
      <span className="text-xs font-medium text-foreground-light">{title}</span>
      <div className="flex gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50" />
        <div className="w-2.5 h-2.5 rounded-full bg-green-500/50" />
      </div>
    </div>
    <pre className="p-4 text-sm font-mono text-foreground-light bg-surface-100 overflow-x-auto">
      <code>{code}</code>
    </pre>
  </div>
)

// Format bytes to human readable
const formatBytes = (bytes: number) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

// Main component
export const RuVectorHome = () => {
  const { ref } = useParams()
  const { stats, history, loading, error, refetch } = useDbStats(3000)
  const [selectedMetric, setSelectedMetric] = useState<'queries' | 'cache' | 'tuples'>('queries')

  // Calculate derived stats
  const cacheHitRatio = stats ?
    (stats.blks_hit + stats.blks_read > 0 ?
      (stats.blks_hit / (stats.blks_hit + stats.blks_read)) * 100 : 100) : 0

  const currentQps = history.length > 0 ? history[history.length - 1].queries : 0

  const liveStats = [
    {
      label: 'Active Connections',
      value: stats?.active_connections || 0,
      icon: Users,
      trend: 0,
    },
    {
      label: 'Queries/sec',
      value: currentQps,
      icon: Zap,
      trend: history.length > 1 ?
        ((history[history.length - 1].queries - history[history.length - 2].queries) /
          (history[history.length - 2].queries || 1)) * 100 : 0,
    },
    {
      label: 'Database Size',
      value: stats?.db_size ? parseFloat(formatBytes(stats.db_size).split(' ')[0]) : 0,
      suffix: stats?.db_size ? ' ' + formatBytes(stats.db_size).split(' ')[1] : '',
      icon: HardDrive,
    },
    {
      label: 'Cache Hit %',
      value: Math.round(cacheHitRatio * 10) / 10,
      suffix: '%',
      icon: Gauge,
      trend: cacheHitRatio > 95 ? 2.5 : cacheHitRatio > 90 ? 0 : -5,
    },
    {
      label: 'Tables',
      value: stats?.table_count || 0,
      icon: Table2,
    },
  ]

  const features = [
    {
      title: 'Vector Indexes',
      description: 'High-performance HNSW and IVFFlat indexes for similarity search',
      icon: Database,
      href: 'vectors',
      stats: '2 Types',
      color: 'bg-blue-500/10',
    },
    {
      title: 'Attention Mechanisms',
      description: '39 attention mechanisms including Multi-Head, Flash, and Sparse',
      icon: Brain,
      href: 'attention',
      stats: '39 Functions',
      color: 'bg-purple-500/10',
    },
    {
      title: 'Graph Neural Networks',
      description: 'GCN, GraphSAGE, and GAT for relational data modeling',
      icon: Network,
      href: 'gnn',
      stats: '3 Architectures',
      color: 'bg-green-500/10',
    },
    {
      title: 'Hyperbolic Embeddings',
      description: 'Poincaré and Lorentz models for hierarchical data',
      icon: Orbit,
      href: 'hyperbolic',
      stats: '2 Models',
      color: 'bg-orange-500/10',
    },
    {
      title: 'Self-Learning',
      description: 'ReasoningBank adaptive learning that improves with experience',
      icon: GraduationCap,
      href: 'learning',
      stats: '4-Stage',
      color: 'bg-pink-500/10',
    },
    {
      title: 'Agent Routing',
      description: 'Tiny Dancer intelligent semantic routing for multi-agent systems',
      icon: Route,
      href: 'routing',
      stats: '3 Functions',
      color: 'bg-cyan-500/10',
    },
  ]

  const databaseTools = [
    { title: 'Table Editor', description: 'View and manage tables', icon: Table2, href: 'editor' },
    { title: 'SQL Editor', description: 'Write and execute queries', icon: Terminal, href: 'sql/1' },
    { title: 'Database', description: 'Tables, functions, triggers', icon: Database, href: 'database/tables' },
    { title: 'Extensions', description: 'PostgreSQL extensions', icon: PuzzleIcon, href: 'database/extensions' },
  ]

  const managementTools = [
    { title: 'Authentication', description: 'User management', icon: Users, href: 'auth/users' },
    { title: 'Storage', description: 'File management', icon: FolderOpen, href: 'storage/buckets' },
    { title: 'API Settings', description: 'Configure API access', icon: Key, href: 'settings/api' },
    { title: 'Project Settings', description: 'General configuration', icon: Settings, href: 'settings/general' },
  ]

  return (
    <div className="w-full h-full overflow-y-auto">
      <div className="px-6 py-8 max-w-7xl mx-auto space-y-10">

        {/* Hero Section */}
        <div className="text-center space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-400/10 border border-brand-400/20">
            <Sparkles className="h-4 w-4 text-brand-500" />
            <span className="text-sm font-medium text-brand-500">Powered by Rust + SIMD</span>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold text-foreground">
            RuVector Database
          </h1>

          <p className="text-lg text-foreground-light max-w-2xl mx-auto">
            Advanced vector database with AI-native capabilities
          </p>

          <div className="flex flex-wrap items-center justify-center gap-3 pt-4">
            <Link
              href={`/project/${ref}/editor`}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-brand-500 text-white font-medium text-sm hover:bg-brand-600 transition-colors"
            >
              <Table2 className="h-4 w-4" />
              Table Editor
            </Link>
            <Link
              href={`/project/${ref}/sql/1`}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-default bg-surface-100 text-foreground font-medium text-sm hover:bg-surface-200 transition-colors"
            >
              <Terminal className="h-4 w-4" />
              SQL Editor
            </Link>
            <Link
              href={`/project/${ref}/database/extensions`}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-default bg-surface-100 text-foreground font-medium text-sm hover:bg-surface-200 transition-colors"
            >
              <PuzzleIcon className="h-4 w-4" />
              Extensions
            </Link>
          </div>
        </div>

        {/* Live Performance Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="h-5 w-5 text-foreground-light" />
              <div>
                <h2 className="font-semibold text-foreground">Live Performance</h2>
                <p className="text-sm text-foreground-light">
                  Real-time PostgreSQL metrics {error && '(simulated)'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={refetch}
                className="p-2 rounded-lg hover:bg-surface-200 transition-colors"
                title="Refresh stats"
              >
                <RefreshCw className={cn("h-4 w-4 text-foreground-light", loading && "animate-spin")} />
              </button>
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-brand-400/10 border border-brand-400/20">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-500 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-500" />
                </span>
                <span className="text-xs font-medium text-brand-500">Live</span>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {liveStats.map((stat) => (
              <LiveStatCard
                key={stat.label}
                {...stat}
                loading={loading}
              />
            ))}
          </div>

          {/* Real-time Graph */}
          <div className="rounded-lg border border-default bg-surface-100 p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-foreground-light" />
                <span className="text-sm font-medium text-foreground">Activity Monitor</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setSelectedMetric('queries')}
                  className={cn(
                    "px-3 py-1 text-xs rounded-md transition-colors",
                    selectedMetric === 'queries'
                      ? "bg-brand-500 text-white"
                      : "bg-surface-200 text-foreground-light hover:bg-surface-300"
                  )}
                >
                  Queries/sec
                </button>
                <button
                  onClick={() => setSelectedMetric('cache')}
                  className={cn(
                    "px-3 py-1 text-xs rounded-md transition-colors",
                    selectedMetric === 'cache'
                      ? "bg-brand-500 text-white"
                      : "bg-surface-200 text-foreground-light hover:bg-surface-300"
                  )}
                >
                  Cache Hit %
                </button>
                <button
                  onClick={() => setSelectedMetric('tuples')}
                  className={cn(
                    "px-3 py-1 text-xs rounded-md transition-colors",
                    selectedMetric === 'tuples'
                      ? "bg-brand-500 text-white"
                      : "bg-surface-200 text-foreground-light hover:bg-surface-300"
                  )}
                >
                  Tuples/sec
                </button>
              </div>
            </div>
            <RealtimeGraph
              data={history}
              height={140}
              color={selectedMetric === 'cache' ? 'blue' : selectedMetric === 'tuples' ? 'orange' : 'brand'}
              label={selectedMetric === 'queries' ? 'Queries/sec' : selectedMetric === 'cache' ? 'Cache Hit %' : 'Tuples/sec'}
            />
            {history.length === 0 && (
              <div className="flex items-center justify-center h-32 text-foreground-muted text-sm">
                Collecting data...
              </div>
            )}
          </div>
        </div>

        {/* RuVector Capabilities */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Sparkles className="h-5 w-5 text-foreground-light" />
              <div>
                <h2 className="font-semibold text-foreground">RuVector Capabilities</h2>
                <p className="text-sm text-foreground-light">AI-native features powered by Rust</p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature) => (
              <FeatureCard key={feature.href} {...feature} />
            ))}
          </div>
        </div>

        {/* Tools Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Database Tools */}
          <div className="p-5 rounded-lg border border-default bg-surface-100">
            <div className="flex items-center gap-3 mb-4">
              <Server className="h-5 w-5 text-foreground-light" />
              <h3 className="font-semibold text-foreground">Database Tools</h3>
            </div>
            <div className="space-y-1">
              {databaseTools.map((tool) => (
                <ToolLink key={tool.href} {...tool} />
              ))}
            </div>
          </div>

          {/* Management Tools */}
          <div className="p-5 rounded-lg border border-default bg-surface-100">
            <div className="flex items-center gap-3 mb-4">
              <Settings className="h-5 w-5 text-foreground-light" />
              <h3 className="font-semibold text-foreground">Management</h3>
            </div>
            <div className="space-y-1">
              {managementTools.map((tool) => (
                <ToolLink key={tool.href} {...tool} />
              ))}
            </div>
          </div>
        </div>

        {/* Quick Start */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Code2 className="h-5 w-5 text-foreground-light" />
              <div>
                <h2 className="font-semibold text-foreground">Quick Start</h2>
                <p className="text-sm text-foreground-light">Get started with vector search</p>
              </div>
            </div>
            <Link
              href={`/project/${ref}/sql/1`}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-brand-400/10 text-brand-500 text-sm font-medium hover:bg-brand-400/20 transition-colors"
            >
              <Play className="h-4 w-4" />
              Open SQL Editor
            </Link>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <CodeBlock
              title="1. Create Vector Table"
              code={`CREATE TABLE embeddings (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(384)
);`}
            />
            <CodeBlock
              title="2. Add HNSW Index"
              code={`CREATE INDEX ON embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);`}
            />
            <CodeBlock
              title="3. Similarity Search"
              code={`SELECT content,
  1 - (embedding <=> query_vec) AS score
FROM embeddings
ORDER BY embedding <=> query_vec
LIMIT 5;`}
            />
          </div>
        </div>

        {/* Footer Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href={`/project/${ref}/auth/policies`}
            className="group p-5 rounded-lg border border-default bg-surface-100 hover:bg-surface-200 transition-colors"
          >
            <Shield className="h-6 w-6 text-foreground-light mb-3" />
            <h3 className="font-semibold text-foreground mb-1">Security</h3>
            <p className="text-sm text-foreground-light mb-3">Row-level security and access controls</p>
            <span className="text-sm text-brand-500 flex items-center gap-1">
              Manage Policies <ChevronRight className="h-4 w-4" />
            </span>
          </Link>

          <Link
            href={`/project/${ref}/database/functions`}
            className="group p-5 rounded-lg border border-default bg-surface-100 hover:bg-surface-200 transition-colors"
          >
            <Workflow className="h-6 w-6 text-foreground-light mb-3" />
            <h3 className="font-semibold text-foreground mb-1">Functions</h3>
            <p className="text-sm text-foreground-light mb-3">Database functions and triggers</p>
            <span className="text-sm text-brand-500 flex items-center gap-1">
              View Functions <ChevronRight className="h-4 w-4" />
            </span>
          </Link>

          <Link
            href={`/project/${ref}/database/indexes`}
            className="group p-5 rounded-lg border border-default bg-surface-100 hover:bg-surface-200 transition-colors"
          >
            <Layers className="h-6 w-6 text-foreground-light mb-3" />
            <h3 className="font-semibold text-foreground mb-1">Indexes</h3>
            <p className="text-sm text-foreground-light mb-3">HNSW and IVFFlat for vector similarity</p>
            <span className="text-sm text-brand-500 flex items-center gap-1">
              Manage Indexes <ChevronRight className="h-4 w-4" />
            </span>
          </Link>
        </div>

        {/* Footer */}
        <div className="text-center py-6 border-t border-default">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-surface-200 text-sm">
            <Star className="h-4 w-4 text-amber-500" />
            <span className="text-foreground-light">Built with</span>
            <span className="font-semibold text-foreground">Rust + SIMD</span>
            <span className="text-foreground-light">for maximum performance</span>
          </div>
        </div>

      </div>
    </div>
  )
}
