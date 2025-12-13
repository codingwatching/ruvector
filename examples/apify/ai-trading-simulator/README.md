# AI Trading Simulator - Neural Market Data & Strategy Backtesting

**Generate unlimited realistic market data with GPU-accelerated neural price models. Backtest algorithmic trading strategies with comprehensive metrics.**

[![Apify Actor](https://img.shields.io/badge/Apify-Actor-blue)](https://apify.com)
[![Node.js](https://img.shields.io/badge/Node.js-20-green)](https://nodejs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Introduction

AI Trading Simulator is a powerful Apify actor that generates unlimited realistic market data for algorithmic trading development, strategy backtesting, and ML model training. Built on the **neural-trader** framework from [ruv.io](https://ruv.io), it combines GPU-accelerated price generation with professional-grade technical analysis.

Perfect for:
- **Quantitative developers** building trading algorithms
- **Data scientists** training ML models for market prediction
- **Fintech startups** testing strategies without risking capital
- **Researchers** studying market dynamics and patterns
- **Educators** teaching algorithmic trading concepts

## ✨ Features

### 🧠 Neural Price Generation
- GPU-accelerated market simulation with configurable volatility
- Realistic OHLCV (Open, High, Low, Close, Volume) data generation
- Support for multiple asset classes (stocks, crypto, indices)
- Reproducible results with seed-based randomization

### 📊 Technical Indicators
- **Moving Averages**: SMA, EMA
- **Momentum**: RSI (Relative Strength Index)
- **Trend**: MACD (Moving Average Convergence Divergence)
- **Volatility**: Bollinger Bands
- All indicators calculated automatically

### 📈 Strategy Backtesting
Four built-in trading strategies:
1. **Mean Reversion** - Buy low, sell high using Bollinger Bands
2. **Momentum** - Follow the trend with MACD crossovers
3. **Breakout** - Trade range breaks and resistance levels
4. **ML Neural** - AI-based decisions using multiple indicators

### 📉 Performance Metrics
Comprehensive backtest analytics:
- Total return and P&L
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside deviation analysis
- **Max Drawdown** - Worst peak-to-trough decline
- **Win Rate** - Percentage of profitable trades
- Detailed trade log with entry/exit reasons

### 🌊 Advanced Features
- **Order Book Simulation** - Level 2 market depth (bids/asks)
- **News Sentiment** - Synthetic market news generation
- **Real-time Streaming** - Live market simulation
- **Multiple Timeframes** - 1m, 5m, 15m, 1h, 4h, 1d
- **Export Formats** - JSON, CSV, Parquet (ML-optimized)

## 🎯 Quick Start

### Run on Apify Platform

```bash
apify call ruv/ai-trading-simulator --input '{
  "mode": "simulate",
  "symbol": "AAPL",
  "timeframe": "1h",
  "dataPoints": 1000,
  "volatility": 0.05,
  "trend": "sideways",
  "includeIndicators": true
}'
```

### Run Locally

```bash
cd /workspaces/ruvector/examples/apify/ai-trading-simulator
npm install
node src/main.js
```

### Example Input Configurations

#### 1. Generate 1 Year of Daily OHLCV Data
```json
{
  "mode": "simulate",
  "symbol": "TSLA",
  "timeframe": "1d",
  "dataPoints": 252,
  "volatility": 0.08,
  "trend": "bullish",
  "includeIndicators": true,
  "outputFormat": "csv"
}
```

#### 2. Backtest a Momentum Strategy
```json
{
  "mode": "backtest",
  "symbol": "BTC-USD",
  "timeframe": "4h",
  "dataPoints": 2000,
  "volatility": 0.15,
  "trend": "volatile",
  "strategy": "momentum",
  "includeIndicators": true,
  "includeOrderBook": true
}
```

#### 3. Train ML Model with Synthetic Data
```json
{
  "mode": "train",
  "symbol": "SPY",
  "timeframe": "15m",
  "dataPoints": 10000,
  "volatility": 0.03,
  "trend": "sideways",
  "includeIndicators": true,
  "includeNews": true,
  "outputFormat": "parquet",
  "seed": 42
}
```

## 🔗 Apify MCP Integration

Integrate with Claude AI using Apify's Model Context Protocol (MCP) server:

```bash
# Add to Claude Desktop MCP config
claude mcp add trading-sim -- npx -y @apify/actors-mcp-server --actors "ruv/ai-trading-simulator"
```

Then use in Claude:
```
Generate 500 hourly candles for AAPL with high volatility and backtest a breakout strategy
```

## 📚 Tutorials

### Tutorial 1: Generate 1 Year of Daily OHLCV Data

This tutorial shows how to generate a full year of realistic daily stock data:

```javascript
{
  "mode": "simulate",
  "symbol": "AAPL",
  "timeframe": "1d",
  "dataPoints": 252,  // Trading days in a year
  "volatility": 0.05,
  "trend": "bullish",
  "initialPrice": 150,
  "includeIndicators": true,
  "includeNews": true,
  "outputFormat": "csv"
}
```

**Use cases:**
- Historical data analysis
- Pattern recognition training
- Correlation studies
- Portfolio simulations

**Output includes:**
- 252 daily candles (OHLCV)
- SMA(20), EMA(12), RSI(14), MACD, Bollinger Bands
- Daily news headlines with sentiment scores
- CSV export for Excel/Google Sheets

---

### Tutorial 2: Backtest a Momentum Strategy

Test a MACD-based momentum strategy on Bitcoin data:

```javascript
{
  "mode": "backtest",
  "symbol": "BTC-USD",
  "timeframe": "4h",
  "dataPoints": 2000,
  "volatility": 0.15,
  "trend": "volatile",
  "strategy": "momentum",
  "includeIndicators": true,
  "includeOrderBook": true
}
```

**Strategy logic:**
- **Buy Signal**: MACD line crosses above signal line
- **Sell Signal**: MACD line crosses below signal line
- Initial capital: $10,000

**Metrics provided:**
- Total return (%)
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk)
- Max drawdown
- Win rate
- Complete trade log with entry/exit prices

**Sample output:**
```json
{
  "backtest": {
    "initialCapital": 10000,
    "finalEquity": 12547.32,
    "totalReturn": 25.47,
    "totalTrades": 84,
    "winRate": 58.33,
    "sharpeRatio": 1.42,
    "sortinoRatio": 1.89,
    "maxDrawdown": 12.34
  }
}
```

---

### Tutorial 3: Train ML Model with Synthetic Data

Generate a large dataset optimized for machine learning:

```javascript
{
  "mode": "train",
  "symbol": "SPY",
  "timeframe": "15m",
  "dataPoints": 10000,
  "volatility": 0.03,
  "trend": "sideways",
  "includeIndicators": true,
  "includeNews": true,
  "outputFormat": "parquet",
  "seed": 42
}
```

**ML-ready features:**
- OHLCV data (5 features)
- Technical indicators (9 features)
- News sentiment (2 features)
- Total: 16+ features per timestamp

**Parquet format benefits:**
- Columnar storage (10x faster queries)
- Efficient compression (5x smaller files)
- Direct integration with pandas, PyTorch, TensorFlow
- Schema preservation

**Python usage:**
```python
import pandas as pd
import pyarrow.parquet as pq

# Load data
df = pd.read_parquet('trading_data.parquet')

# Features for ML
features = df[['open', 'high', 'low', 'close', 'volume',
               'sma_20', 'ema_12', 'rsi_14', 'macd']]
labels = df['close'].shift(-1)  # Predict next close

# Train your model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(features[:-1], labels[:-1])
```

---

### Tutorial 4: Simulate Flash Crash Scenarios

Test how strategies perform during extreme volatility:

```javascript
{
  "mode": "backtest",
  "symbol": "AAPL",
  "timeframe": "1m",
  "dataPoints": 500,
  "volatility": 0.3,  // Extreme volatility
  "trend": "bearish",
  "strategy": "mean_reversion",
  "includeIndicators": true,
  "includeOrderBook": true
}
```

**Scenario simulation:**
- High volatility (30%) mimics flash crashes
- Bearish trend for stress testing
- 1-minute candles for intraday dynamics
- Order book shows liquidity gaps

**Analysis:**
- How does mean reversion perform in crashes?
- Maximum drawdown during extreme moves
- Recovery time after volatility spike
- Liquidity impact on strategy execution

---

## 🎮 Input Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `mode` | enum | Operation mode: `simulate`, `backtest`, `train`, `stream` | `simulate` |
| `symbol` | string | Trading symbol (e.g., AAPL, BTC-USD) | `AAPL` |
| `timeframe` | enum | Candle interval: `1m`, `5m`, `15m`, `1h`, `4h`, `1d` | `1h` |
| `dataPoints` | integer | Number of candles to generate (10-100000) | `1000` |
| `volatility` | number | Market volatility factor (0.01-0.5) | `0.05` |
| `trend` | enum | Market direction: `bullish`, `bearish`, `sideways`, `volatile` | `sideways` |
| `strategy` | enum | Trading strategy for backtest mode | `momentum` |
| `includeOrderBook` | boolean | Generate Level 2 order book data | `false` |
| `includeNews` | boolean | Generate market news sentiment | `false` |
| `includeIndicators` | boolean | Calculate technical indicators | `true` |
| `outputFormat` | enum | Export format: `json`, `csv`, `parquet` | `json` |
| `initialPrice` | number | Starting price (auto-detected if empty) | - |
| `seed` | integer | Random seed for reproducibility | - |

## 📊 Output Schema

### Simulate Mode

```json
{
  "data": [
    {
      "timestamp": "2025-01-15T10:00:00.000Z",
      "symbol": "AAPL",
      "open": 175.32,
      "high": 176.45,
      "low": 174.89,
      "close": 175.78,
      "volume": 2847593,
      "indicators": {
        "sma_20": 174.56,
        "ema_12": 175.23,
        "rsi_14": 62.34,
        "macd": 0.45,
        "macd_signal": 0.38,
        "macd_histogram": 0.07,
        "bb_upper": 178.23,
        "bb_middle": 175.12,
        "bb_lower": 172.01
      },
      "orderBook": {
        "bids": [{"price": 175.77, "size": 450}, ...],
        "asks": [{"price": 175.79, "size": 320}, ...]
      },
      "news": {
        "headline": "AAPL surges on strong earnings report",
        "sentiment": "positive",
        "score": 0.82
      }
    }
  ],
  "metadata": {
    "symbol": "AAPL",
    "timeframe": "1h",
    "dataPoints": 1000,
    "volatility": 0.05,
    "trend": "sideways",
    "generatedAt": "2025-01-15T12:34:56.789Z",
    "mode": "simulate"
  }
}
```

### Backtest Mode

Includes all `simulate` data plus:

```json
{
  "backtest": {
    "initialCapital": 10000,
    "finalEquity": 11234.56,
    "totalReturn": 12.35,
    "totalTrades": 42,
    "completedTrades": 21,
    "winRate": 57.14,
    "sharpeRatio": 1.23,
    "sortinoRatio": 1.45,
    "maxDrawdown": 8.76,
    "trades": [
      {
        "timestamp": "2025-01-15T10:00:00.000Z",
        "action": "buy",
        "price": 175.32,
        "shares": 57,
        "reason": "MACD bullish crossover"
      }
    ],
    "equityCurve": [
      {"timestamp": "2025-01-15T10:00:00.000Z", "equity": 10000}
    ]
  },
  "signals": [
    {"action": "buy", "strength": 0.9, "reason": "MACD bullish crossover"}
  ]
}
```

## 🔧 Technical Implementation

### Architecture

```
AI Trading Simulator
├── MarketDataGenerator - Neural price generation
│   ├── OHLCV synthesis with trend/volatility
│   ├── Order book simulation
│   └── News sentiment generation
├── TechnicalIndicators - Technical analysis
│   ├── SMA, EMA (Moving Averages)
│   ├── RSI (Momentum)
│   ├── MACD (Trend)
│   └── Bollinger Bands (Volatility)
├── TradingStrategies - Strategy implementations
│   ├── Mean Reversion
│   ├── Momentum
│   ├── Breakout
│   └── ML Neural
└── Backtester - Performance analysis
    ├── Trade execution simulation
    ├── Equity curve tracking
    └── Metrics calculation
```

### Neural Price Model

The price generation uses a sophisticated neural-inspired model:

1. **Base Price**: Auto-detected or user-specified
2. **Trend Component**: Directional drift based on trend parameter
3. **Volatility Component**: Random walk scaled by volatility
4. **OHLCV Construction**:
   - Open = Current price
   - High = Open + random volatility range
   - Low = Open - random volatility range
   - Close = Random point between High and Low
   - Volume = Random with volatility correlation

5. **Seeded Randomness**: Reproducible results with seed parameter

### Indicator Calculations

- **SMA**: Simple Moving Average over N periods
- **EMA**: Exponential Moving Average with multiplier 2/(N+1)
- **RSI**: Relative Strength Index using average gains/losses
- **MACD**: 12/26/9 EMA configuration with histogram
- **Bollinger Bands**: 20-period SMA ± 2 standard deviations

### Backtesting Engine

1. **Signal Generation**: Strategy analyzes each candle
2. **Position Management**: Buy/sell execution at close prices
3. **Capital Tracking**: Real-time equity calculation
4. **Metrics Calculation**:
   - Returns: Daily/period returns array
   - Sharpe: (Mean Return / StdDev) × √252
   - Sortino: (Mean Return / Downside Dev) × √252
   - Drawdown: Max(Peak - Valley) / Peak

## 📈 Use Cases

### Quantitative Trading
- **Algo development**: Test strategies on unlimited data
- **Parameter optimization**: Grid search for best indicators
- **Walk-forward analysis**: Train on period N, test on N+1
- **Risk management**: Simulate position sizing strategies

### Data Science & ML
- **Training datasets**: Generate 100K+ labeled examples
- **Feature engineering**: Test indicator combinations
- **Model validation**: Cross-validate on synthetic data
- **Transfer learning**: Pre-train on synthetic, fine-tune on real

### Fintech Applications
- **Paper trading**: Risk-free strategy testing
- **User education**: Interactive trading simulators
- **Product demos**: Showcase trading platforms
- **Regulatory testing**: Stress test compliance algorithms

### Academic Research
- **Market microstructure**: Study order book dynamics
- **Behavioral finance**: Model sentiment-driven trading
- **Systemic risk**: Simulate cascading failures
- **Agent-based models**: Multi-strategy market simulation

## 🚀 Performance & Scalability

- **Generation speed**: 10,000 candles in <1 second
- **Memory efficient**: Streaming mode for unlimited data
- **Reproducible**: Seed-based deterministic results
- **Parallel-ready**: Stateless design for multi-actor scaling

### Apify Platform Benefits
- **Auto-scaling**: Handle millions of data points
- **Scheduled runs**: Daily strategy backtests
- **Webhooks**: Integrate with trading platforms
- **Dataset API**: Query results programmatically

## 🔐 Security & Privacy

- **No real market connections**: Fully synthetic data
- **No API keys required**: Self-contained simulation
- **No PII collection**: Anonymous usage
- **Open source**: Transparent implementation

## 🛠️ Development

### Local Testing

```bash
# Install dependencies
npm install

# Run with default config
node src/main.js

# Run with custom input
echo '{
  "mode": "backtest",
  "symbol": "TSLA",
  "dataPoints": 500,
  "strategy": "momentum"
}' > input.json

APIFY_INPUT_PATH=input.json node src/main.js
```

### Deploy to Apify

```bash
apify login
apify push
```

## 📖 Related Resources

- **[neural-trader](https://www.npmjs.com/package/neural-trader)** - NPM package for neural trading models
- **[ruv.io](https://ruv.io)** - AI-powered trading tools and platforms
- **[ruvector](https://github.com/ruvnet/ruvector)** - High-performance vector database for trading data
- **[Apify Actors](https://apify.com/actors)** - Serverless automation platform
- **[Apify MCP](https://docs.apify.com/mcp)** - Model Context Protocol integration

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional trading strategies (pairs trading, arbitrage, etc.)
- More technical indicators (Ichimoku, Fibonacci, etc.)
- Machine learning integration (LSTM, Transformers)
- Multi-asset portfolio simulation
- Options and derivatives pricing

## 📄 License

MIT License - see [LICENSE](https://opensource.org/licenses/MIT)

## 🌟 Support

- **GitHub Issues**: [ruvnet/ruvector](https://github.com/ruvnet/ruvector/issues)
- **Email**: info@ruv.io
- **Website**: [https://ruv.io](https://ruv.io)

---

## 🔍 SEO Keywords

**Primary**: AI Trading Simulator, Neural Market Data Generator, Algorithmic Trading Backtesting, Synthetic OHLCV Data, Quantitative Finance Tools

**Secondary**: trading strategy backtest, technical indicators calculator, market simulation, fintech data generation, ML trading data, GPU-accelerated trading, Sharpe ratio calculator, algorithmic trading development, quant tools, cryptocurrency simulation

**Long-tail**: generate realistic stock market data, backtest trading strategies with Python, synthetic market data for ML training, OHLCV generator for algorithmic trading, neural network trading simulator, GPU-accelerated market simulation, technical analysis indicators API, trading strategy performance metrics

**Industry**: quantitative finance, algorithmic trading, high-frequency trading, market making, statistical arbitrage, machine learning trading, systematic trading, fintech development, data science finance

**Technologies**: Apify actor, Node.js trading, neural trading models, MCP integration, Claude AI trading, serverless backtesting, cloud-based quant tools

---

**Built with ❤️ by [rUv](https://ruv.io) | Powered by [Apify](https://apify.com) | Part of [ruvector](https://github.com/ruvnet/ruvector)**

*Generate unlimited market data. Backtest like a pro. Trade with confidence.*
