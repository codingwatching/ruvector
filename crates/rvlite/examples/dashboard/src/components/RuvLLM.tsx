import { useState } from 'react'

interface TrmConfig {
  iterations: number
  threshold: number
  maxContextLength: number
}

interface LoraConfig {
  rank: number
  alpha: number
  learningRate: number
  enabled: boolean
}

interface ModelState {
  loaded: boolean
  name: string
  size: string
}

export default function RuvLLM() {
  const [trmConfig, setTrmConfig] = useState<TrmConfig>({
    iterations: 3,
    threshold: 0.1,
    maxContextLength: 2048,
  })

  const [microLora, setMicroLora] = useState<LoraConfig>({
    rank: 2,
    alpha: 1.0,
    learningRate: 0.001,
    enabled: true,
  })

  const [baseLora, setBaseLora] = useState<LoraConfig>({
    rank: 8,
    alpha: 16,
    learningRate: 0.0001,
    enabled: false,
  })

  const [model, setModel] = useState<ModelState>({
    loaded: false,
    name: '',
    size: '',
  })

  const [modelUrl, setModelUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [response, setResponse] = useState('')

  async function loadModel() {
    if (!modelUrl) return

    setLoading(true)
    try {
      // Simulate model loading
      await new Promise((r) => setTimeout(r, 1500))

      const fileName = modelUrl.split('/').pop() || 'model.onnx'
      setModel({
        loaded: true,
        name: fileName,
        size: '~7M parameters',
      })
    } catch (err) {
      console.error('Failed to load model:', err)
    } finally {
      setLoading(false)
    }
  }

  function runInference() {
    if (!prompt.trim()) return

    // Simulate TRM reasoning
    const steps: string[] = []
    steps.push(`[TRM K=${trmConfig.iterations}] Processing prompt...`)
    for (let i = 1; i <= trmConfig.iterations; i++) {
      steps.push(`  Iteration ${i}: Refining answer (threshold: ${trmConfig.threshold})`)
    }
    steps.push('')
    steps.push('Response: This is a simulated response from the TRM engine.')
    steps.push('In production, this would run actual ONNX inference.')

    if (microLora.enabled) {
      steps.push('')
      steps.push(`[MicroLoRA] Applied rank-${microLora.rank} adaptation`)
    }

    setResponse(steps.join('\n'))
  }

  return (
    <div>
      <div className="page-header">
        <h2>RuvLLM Settings</h2>
        <p>Tiny Recursive Models with self-learning LoRA adapters</p>
      </div>

      <div className="grid-2">
        {/* Model Loading */}
        <div className="card">
          <div className="card-header flex justify-between items-center">
            <span>ONNX Model</span>
            {model.loaded && <span className="badge badge-success">Loaded</span>}
          </div>

          <div className="form-group">
            <label className="form-label">Model URL or Path</label>
            <input
              type="text"
              className="form-input"
              placeholder="https://example.com/model.onnx"
              value={modelUrl}
              onChange={(e) => setModelUrl(e.target.value)}
            />
          </div>

          <button
            className="btn btn-primary"
            onClick={loadModel}
            disabled={loading || !modelUrl}
          >
            {loading ? 'Loading...' : 'Load Model'}
          </button>

          {model.loaded && (
            <div style={{ marginTop: 12, fontSize: 13, color: 'var(--text-secondary)' }}>
              <div>Model: {model.name}</div>
              <div>Size: {model.size}</div>
            </div>
          )}
        </div>

        {/* TRM Configuration */}
        <div className="card">
          <div className="card-header">TRM Configuration</div>

          <div className="form-group">
            <label className="form-label">
              Iterations (K): {trmConfig.iterations}
            </label>
            <input
              type="range"
              className="form-range"
              min="1"
              max="10"
              value={trmConfig.iterations}
              onChange={(e) =>
                setTrmConfig({ ...trmConfig, iterations: Number(e.target.value) })
              }
            />
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              More iterations = better reasoning, higher latency
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              Convergence Threshold: {trmConfig.threshold}
            </label>
            <input
              type="range"
              className="form-range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={trmConfig.threshold}
              onChange={(e) =>
                setTrmConfig({ ...trmConfig, threshold: Number(e.target.value) })
              }
            />
          </div>

          <div className="form-group">
            <label className="form-label">Max Context Length</label>
            <select
              className="form-select"
              value={trmConfig.maxContextLength}
              onChange={(e) =>
                setTrmConfig({ ...trmConfig, maxContextLength: Number(e.target.value) })
              }
            >
              <option value="512">512 tokens</option>
              <option value="1024">1024 tokens</option>
              <option value="2048">2048 tokens</option>
              <option value="4096">4096 tokens</option>
            </select>
          </div>
        </div>
      </div>

      <div className="grid-2 mt-4">
        {/* MicroLoRA */}
        <div className="card">
          <div className="card-header flex justify-between items-center">
            <span>MicroLoRA (Per-Request)</span>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={microLora.enabled}
                onChange={(e) => setMicroLora({ ...microLora, enabled: e.target.checked })}
              />
              Enabled
            </label>
          </div>

          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
            Instant adaptation from user feedback (~100us latency)
          </div>

          <div className="form-group">
            <label className="form-label">Rank: {microLora.rank}</label>
            <input
              type="range"
              className="form-range"
              min="1"
              max="4"
              value={microLora.rank}
              onChange={(e) => setMicroLora({ ...microLora, rank: Number(e.target.value) })}
              disabled={!microLora.enabled}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Learning Rate: {microLora.learningRate}</label>
            <input
              type="range"
              className="form-range"
              min="0.0001"
              max="0.01"
              step="0.0001"
              value={microLora.learningRate}
              onChange={(e) =>
                setMicroLora({ ...microLora, learningRate: Number(e.target.value) })
              }
              disabled={!microLora.enabled}
            />
          </div>
        </div>

        {/* BaseLoRA */}
        <div className="card">
          <div className="card-header flex justify-between items-center">
            <span>BaseLoRA (Background)</span>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={baseLora.enabled}
                onChange={(e) => setBaseLora({ ...baseLora, enabled: e.target.checked })}
              />
              Enabled
            </label>
          </div>

          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
            Periodic background training for long-term adaptation
          </div>

          <div className="form-group">
            <label className="form-label">Rank: {baseLora.rank}</label>
            <input
              type="range"
              className="form-range"
              min="4"
              max="32"
              value={baseLora.rank}
              onChange={(e) => setBaseLora({ ...baseLora, rank: Number(e.target.value) })}
              disabled={!baseLora.enabled}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Alpha: {baseLora.alpha}</label>
            <input
              type="range"
              className="form-range"
              min="1"
              max="32"
              value={baseLora.alpha}
              onChange={(e) => setBaseLora({ ...baseLora, alpha: Number(e.target.value) })}
              disabled={!baseLora.enabled}
            />
          </div>
        </div>
      </div>

      {/* Test Playground */}
      <div className="card mt-4">
        <div className="card-header">Test Playground</div>

        <div className="form-group">
          <label className="form-label">Prompt</label>
          <textarea
            className="query-input"
            rows={3}
            placeholder="Enter a test prompt..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <button className="btn btn-primary" onClick={runInference} disabled={!prompt.trim()}>
          Run Inference
        </button>

        {response && (
          <div className="mt-4">
            <label className="form-label">Output</label>
            <pre className="results-area">{response}</pre>
          </div>
        )}
      </div>

      {/* Info Card */}
      <div className="card mt-4">
        <div className="card-header">About RuvLLM</div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          <p style={{ marginBottom: 8 }}>
            <strong>TRM (Tiny Recursive Models)</strong> - Samsung's approach using small ~7M
            parameter models with recursive reasoning (K iterations) to achieve GPT-4 level
            performance on complex tasks.
          </p>
          <p style={{ marginBottom: 8 }}>
            <strong>MicroLoRA</strong> - Rank 1-2 LoRA adapters for instant per-request
            adaptation based on user feedback (thumbs up/down). Adds ~100us latency.
          </p>
          <p>
            <strong>BaseLoRA</strong> - Rank 4-16 adapters trained in the background to
            accumulate long-term learning. Distilled from MicroLoRA feedback.
          </p>
        </div>
      </div>
    </div>
  )
}
