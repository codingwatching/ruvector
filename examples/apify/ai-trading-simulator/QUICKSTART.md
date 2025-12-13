# AI Trading Simulator - Quick Start Guide

## 🚀 Local Testing

### Option 1: Standalone Test (No Apify SDK required)

```bash
# Run pre-configured test
node standalone-test.js

# Run multiple test scenarios
node test-configs.js
```

### Option 2: With Apify SDK

```bash
# Install dependencies
npm install

# Create input configuration
mkdir -p apify_storage/key_value_stores/default

cat > apify_storage/key_value_stores/default/INPUT.json << 'EOF'
{
  "mode": "backtest",
  "symbol": "AAPL",
  "timeframe": "1h",
  "dataPoints": 500,
  "volatility": 0.05,
  "trend": "sideways",
  "strategy": "momentum",
  "includeIndicators": true
}
EOF

# Run actor
APIFY_LOCAL_STORAGE_DIR=./apify_storage APIFY_TOKEN=test node src/main.js
```

## 📊 Example Configurations

### 1. Generate Market Data (Simulate Mode)

```json
{
  "mode": "simulate",
  "symbol": "BTC-USD",
  "timeframe": "4h",
  "dataPoints": 1000,
  "volatility": 0.12,
  "trend": "volatile",
  "includeIndicators": true,
  "includeOrderBook": true,
  "includeNews": true,
  "outputFormat": "json"
}
```

### 2. Backtest Momentum Strategy

```json
{
  "mode": "backtest",
  "symbol": "TSLA",
  "timeframe": "1h",
  "dataPoints": 2000,
  "volatility": 0.08,
  "trend": "bullish",
  "strategy": "momentum",
  "includeIndicators": true
}
```

### 3. Generate ML Training Data

```json
{
  "mode": "train",
  "symbol": "SPY",
  "timeframe": "15m",
  "dataPoints": 10000,
  "volatility": 0.03,
  "trend": "sideways",
  "includeIndicators": true,
  "outputFormat": "parquet",
  "seed": 42
}
```

## 🎯 Trading Strategies

### Momentum (MACD-based)
- **Buy**: MACD line crosses above signal line
- **Sell**: MACD line crosses below signal line
- Best for trending markets

### Mean Reversion (Bollinger Bands)
- **Buy**: Price touches or crosses below lower band
- **Sell**: Price touches or crosses above upper band
- Best for range-bound markets

### Breakout
- **Buy**: Price breaks above 20-period high
- **Sell**: Price breaks below 20-period low
- Best for volatile markets with clear support/resistance

### ML Neural
- **Buy/Sell**: Weighted decision based on RSI, MACD, and trend
- Multi-indicator approach
- Best for diverse market conditions

## 📈 Understanding Results

### Backtest Metrics

- **Total Return**: Percentage profit/loss from initial capital
- **Sharpe Ratio**: Risk-adjusted returns (>1 is good, >2 is excellent)
- **Sortino Ratio**: Like Sharpe but only considers downside risk
- **Max Drawdown**: Worst peak-to-trough decline (lower is better)
- **Win Rate**: Percentage of profitable trades

### Good Performance Indicators
✅ Total Return > 10%
✅ Sharpe Ratio > 1.0
✅ Win Rate > 50%
✅ Max Drawdown < 20%

### Warning Signs
⚠️ Win Rate < 40%
⚠️ Max Drawdown > 50%
⚠️ Sharpe Ratio < 0.5

## 🔧 Customization Tips

### Volatility Settings
- **Low (0.01-0.03)**: Large cap stocks, indices
- **Medium (0.04-0.08)**: Most stocks, ETFs
- **High (0.09-0.20)**: Small caps, volatile stocks
- **Extreme (0.20+)**: Crypto, meme stocks

### Trend Settings
- **Bullish**: Upward trending market (bull run)
- **Bearish**: Downward trending market (bear market)
- **Sideways**: Range-bound, consolidating
- **Volatile**: High fluctuation, no clear direction

### Data Points
- **100-500**: Quick tests, proof of concept
- **500-2000**: Standard backtests (few weeks to months)
- **2000-10000**: Long-term analysis (1-5 years)
- **10000+**: ML training datasets

## 🎓 Learning Path

1. **Start Simple**: Run standalone-test.js to see how it works
2. **Experiment**: Try different volatility and trend combinations
3. **Compare Strategies**: Test all 4 strategies on same data (use seed for reproducibility)
4. **Optimize**: Find best parameters for your strategy
5. **Deploy**: Use on Apify platform for production runs

## 🐛 Troubleshooting

### No trades generated
- Increase data points (try 500-1000+)
- Adjust volatility (higher volatility = more signals)
- Try different strategy
- Check MACD parameters (fast=12, slow=26, signal=9)

### Poor performance
- Wrong strategy for market condition
- Volatility too high/low
- Insufficient data points
- Over-optimization (use validation set)

### Apify local storage issues
- Use standalone-test.js instead
- Ensure INPUT.json is in correct location
- Check APIFY_LOCAL_STORAGE_DIR environment variable

## 📚 Next Steps

- Read full [README.md](./README.md) for comprehensive documentation
- Deploy to [Apify Platform](https://console.apify.com/)
- Integrate with Claude AI via MCP
- Customize strategies in [src/main.js](./src/main.js)
- Export to Python for ML training

## 🔗 Resources

- **Apify Documentation**: https://docs.apify.com/
- **ruv.io**: https://ruv.io
- **ruvector GitHub**: https://github.com/ruvnet/ruvector
- **Technical Analysis**: https://www.investopedia.com/technical-analysis-4689657

---

**Built with ❤️ by [rUv](https://ruv.io) | Powered by [Apify](https://apify.com)**
