# Financial Stress Testing Platform - VaR, Scenario Analysis & Risk Simulation

**Enterprise-grade financial stress testing for portfolio risk management, regulatory compliance, and quantitative analysis.**

## Introduction

The Financial Stress Testing Platform provides institutional-quality risk assessment tools for financial portfolios. Built on the powerful **ruvector** framework and leveraging **neural-trader** capabilities from [ruv.io](https://ruv.io), this Apify actor enables portfolio managers, quantitative analysts, risk officers, and compliance teams to:

- Calculate Value at Risk (VaR) using parametric, historical, and Monte Carlo methods
- Perform comprehensive scenario analysis with 10+ pre-built crisis scenarios
- Run Monte Carlo simulations with up to 100,000 paths
- Generate Basel III compliant regulatory reports
- Analyze tail risk, maximum drawdown, and recovery metrics
- Model asset correlations and portfolio diversification effects

Whether you're managing a hedge fund, running enterprise risk systems, or ensuring regulatory compliance, this platform delivers the sophisticated analytics you need.

## Features

### Value at Risk (VaR) Calculation

- **Parametric VaR**: Variance-covariance method with normal distribution assumptions
- **Historical VaR**: Non-parametric approach using actual return distributions
- **Monte Carlo VaR**: Simulation-based with configurable iterations (1,000 - 100,000)
- **Conditional VaR (CVaR)**: Expected Shortfall calculation for tail risk
- **Multi-horizon**: 1-day, 10-day, 30-day, and annual risk measurements
- **Confidence levels**: 90%, 95%, 99%, 99.5% support

### Crisis Scenario Analysis

Pre-built historical scenarios include:

1. **2008 Financial Crisis** - 55% equity decline, credit crunch, correlation breakdown
2. **COVID-19 Crash 2020** - 35% drop with rapid volatility spike and V-shaped recovery
3. **Flash Crash 2010** - Intraday liquidity shock with 10% decline
4. **Rapid Rate Hike** - 500bps rate increase, bond market crash
5. **Currency Crisis** - 30% FX depreciation, emerging market contagion
6. **Dot-com Bubble 2000** - 78% tech decline over 2+ years
7. **Black Monday 1987** - Single-day 22% equity crash
8. **European Debt Crisis** - Sovereign spread widening, euro weakness
9. **Oil Shock 1973** - 400% oil price increase, stagflation
10. **Stagflation Scenario** - High inflation + negative growth

Each scenario includes realistic timeline progression and multi-asset impacts.

### Monte Carlo Simulation

- Up to **100,000 simulation paths** for robust statistical analysis
- **Correlation modeling** using Cholesky decomposition
- Asset-specific volatility and expected return inputs
- Normal and custom return distribution support
- Parallel computation for performance

### Risk Metrics Suite

- **Maximum Drawdown** - Peak-to-trough decline analysis
- **Recovery Time** - Duration to regain peak values
- **Sharpe Ratio** - Risk-adjusted return measurement
- **Sortino Ratio** - Downside deviation-based performance
- **Skewness** - Return distribution asymmetry
- **Kurtosis** - Fat tail analysis (excess kurtosis)
- **Tail Risk** - Extreme event probability

### Regulatory Compliance

- **Basel III Market Risk Capital** - VaR-based capital requirements
- **Stressed VaR** - Capital under adverse scenarios
- **Incremental Risk Charge** - Default and migration risk
- **Comprehensive Risk Measure** - Correlation trading risks
- **Concentration Risk** - Herfindahl Index calculation
- **Model Validation** - Backtesting and independent review tracking

### Custom Shock Scenarios

Build your own stress tests:

- Equity market shocks (% changes)
- Interest rate shocks (basis points)
- FX rate movements (% changes)
- Volatility multipliers
- Combined multi-factor shocks

## Quick Start

### Basic VaR Calculation

```javascript
{
  "testType": "var_calculation",
  "portfolio": [
    { "symbol": "SPY", "weight": 0.6, "quantity": 200, "currentPrice": 450 },
    { "symbol": "TLT", "weight": 0.4, "quantity": 300, "currentPrice": 95 }
  ],
  "confidenceLevel": 0.99,
  "timeHorizon": "10d",
  "simulations": 10000,
  "includeCorrelations": true,
  "outputMetrics": ["var", "cvar", "max_drawdown"]
}
```

### Scenario Analysis

```javascript
{
  "testType": "scenario_analysis",
  "portfolio": [
    { "symbol": "SPY", "weight": 0.4, "quantity": 100, "currentPrice": 450 },
    { "symbol": "TLT", "weight": 0.3, "quantity": 150, "currentPrice": 95 },
    { "symbol": "GLD", "weight": 0.3, "quantity": 80, "currentPrice": 185 }
  ],
  "scenarios": ["2008_crisis", "covid_crash", "flash_crash"],
  "confidenceLevel": 0.95,
  "outputMetrics": ["stress_pnl", "max_drawdown", "recovery_time"],
  "generateRegulatoryReport": true
}
```

### Monte Carlo Stress Test

```javascript
{
  "testType": "monte_carlo",
  "portfolio": [
    { "symbol": "AAPL", "weight": 0.25, "quantity": 50, "currentPrice": 180, "volatility": 0.025 },
    { "symbol": "MSFT", "weight": 0.25, "quantity": 60, "currentPrice": 380, "volatility": 0.022 },
    { "symbol": "GOOGL", "weight": 0.25, "quantity": 40, "currentPrice": 140, "volatility": 0.028 },
    { "symbol": "AMZN", "weight": 0.25, "quantity": 35, "currentPrice": 175, "volatility": 0.030 }
  ],
  "confidenceLevel": 0.99,
  "timeHorizon": "30d",
  "simulations": 50000,
  "includeCorrelations": true,
  "outputMetrics": ["var", "cvar", "skewness", "kurtosis"]
}
```

## Apify MCP Integration

Integrate this actor with Claude Code and MCP for seamless risk analysis:

### Installation

```bash
# Add the Financial Stress Testing actor to your MCP server
claude mcp add stress-test -- npx -y @apify/actors-mcp-server --actors "ruv/financial-stress-test"
```

### Usage with Claude

Once installed, you can use natural language to run stress tests:

```
"Calculate 99% VaR for my portfolio with SPY, TLT, and GLD positions"

"Run a 2008 crisis scenario on my portfolio and show me the maximum drawdown"

"Perform a Monte Carlo simulation with 50,000 iterations including correlations"

"Generate a Basel III regulatory report for my current positions"
```

### MCP Tool Functions

The actor exposes these MCP tools:

- `run_var_calculation` - Calculate Value at Risk with multiple methodologies
- `run_scenario_analysis` - Apply historical crisis scenarios
- `run_monte_carlo` - Execute Monte Carlo simulations
- `generate_regulatory_report` - Create compliance reports
- `calculate_risk_metrics` - Compute Sharpe, Sortino, tail risk metrics

## Tutorials

### Tutorial 1: Calculate 99% VaR for Equity Portfolio

**Objective**: Measure potential loss with 99% confidence over 10 trading days.

```javascript
{
  "testType": "var_calculation",
  "portfolio": [
    { "symbol": "SPY", "weight": 0.50, "quantity": 222, "currentPrice": 450.00 },
    { "symbol": "QQQ", "weight": 0.30, "quantity": 200, "currentPrice": 375.00 },
    { "symbol": "IWM", "weight": 0.20, "quantity": 500, "currentPrice": 200.00 }
  ],
  "confidenceLevel": 0.99,
  "timeHorizon": "10d",
  "simulations": 25000,
  "includeCorrelations": true,
  "outputMetrics": ["var", "cvar", "sharpe_ratio"],
  "generateRegulatoryReport": false
}
```

**Expected Results**:
- VaR (99%): Approximately $15,000-$25,000 depending on volatility
- CVaR: 20-30% higher than VaR (tail risk premium)
- Sharpe Ratio: Portfolio risk-adjusted performance

**Interpretation**:
- There's a 1% chance of losing more than the VaR amount in 10 days
- CVaR shows average loss if you fall into the worst 1% of outcomes
- Use this to set position limits and capital allocation

### Tutorial 2: Run 2008 Crisis Scenario Replay

**Objective**: Understand portfolio behavior during severe market stress.

```javascript
{
  "testType": "scenario_analysis",
  "portfolio": [
    { "symbol": "SPY", "weight": 0.40, "quantity": 178, "currentPrice": 450.00 },
    { "symbol": "TLT", "weight": 0.30, "quantity": 316, "currentPrice": 95.00 },
    { "symbol": "GLD", "weight": 0.20, "quantity": 108, "currentPrice": 185.00 },
    { "symbol": "DBC", "weight": 0.10, "quantity": 400, "currentPrice": 25.00 }
  ],
  "scenarios": ["2008_crisis"],
  "outputMetrics": ["stress_pnl", "max_drawdown", "recovery_time"],
  "generateRegulatoryReport": true
}
```

**Expected Results**:
- Total P&L: -30% to -40% (bonds provide partial offset)
- Max Drawdown: 40-45% from peak
- Recovery Time: 400-500 trading days (18-24 months)

**Interpretation**:
- Equity-heavy portfolios suffer significantly in credit crises
- Bonds (TLT) and gold (GLD) provide diversification benefits
- Recovery takes 1.5-2 years historically
- Adjust allocation if drawdown exceeds risk tolerance

### Tutorial 3: Monte Carlo Stress Test with 10,000 Simulations

**Objective**: Model portfolio risk with realistic correlations.

```javascript
{
  "testType": "monte_carlo",
  "portfolio": [
    { "symbol": "AAPL", "weight": 0.20, "quantity": 111, "currentPrice": 180.00, "volatility": 0.025, "expectedReturn": 0.0005 },
    { "symbol": "MSFT", "weight": 0.20, "quantity": 53, "currentPrice": 380.00, "volatility": 0.022, "expectedReturn": 0.0004 },
    { "symbol": "GOOGL", "weight": 0.20, "quantity": 143, "currentPrice": 140.00, "volatility": 0.028, "expectedReturn": 0.0003 },
    { "symbol": "JPM", "weight": 0.20, "quantity": 139, "currentPrice": 145.00, "volatility": 0.020, "expectedReturn": 0.0003 },
    { "symbol": "JNJ", "weight": 0.20, "quantity": 125, "currentPrice": 160.00, "volatility": 0.015, "expectedReturn": 0.0002 }
  ],
  "confidenceLevel": 0.95,
  "timeHorizon": "10d",
  "simulations": 10000,
  "includeCorrelations": true,
  "outputMetrics": ["var", "cvar", "skewness", "kurtosis", "sharpe_ratio"]
}
```

**Expected Results**:
- VaR (95%): $8,000-$12,000 (well-diversified portfolio)
- Skewness: Slightly negative (left tail risk)
- Kurtosis: Positive (fat tails, extreme events more likely than normal distribution)
- Sharpe Ratio: 1.0-1.5 (depends on expected returns)

**Interpretation**:
- Correlations reduce diversification benefits during stress
- Negative skewness indicates asymmetric downside risk
- Positive kurtosis shows higher probability of extreme losses
- Monitor tail risk metrics regularly

### Tutorial 4: Generate Regulatory Risk Report

**Objective**: Create Basel III compliant documentation for auditors.

```javascript
{
  "testType": "var_calculation",
  "portfolio": [
    { "symbol": "SPY", "weight": 0.35, "quantity": 156, "currentPrice": 450.00 },
    { "symbol": "IEF", "weight": 0.25, "quantity": 250, "currentPrice": 100.00 },
    { "symbol": "GLD", "weight": 0.15, "quantity": 81, "currentPrice": 185.00 },
    { "symbol": "VNQ", "weight": 0.15, "quantity": 175, "currentPrice": 85.00 },
    { "symbol": "DBC", "weight": 0.10, "quantity": 400, "currentPrice": 25.00 }
  ],
  "confidenceLevel": 0.99,
  "timeHorizon": "10d",
  "simulations": 50000,
  "includeCorrelations": true,
  "scenarios": ["2008_crisis", "covid_crash", "rate_hike"],
  "outputMetrics": ["var", "cvar", "max_drawdown", "stress_pnl"],
  "generateRegulatoryReport": true,
  "exportFormat": "json"
}
```

**Expected Output**:
- Market Risk Capital calculation
- Stressed VaR and Incremental Risk Charge
- Concentration risk metrics (Herfindahl Index)
- Scenario stress test results
- Compliance recommendations

**Use Cases**:
- Quarterly regulatory filings
- Internal risk committee presentations
- Audit documentation
- Capital planning

## Output Schema

### VaR Results

```javascript
{
  "varResults": {
    "parametric": {
      "method": "parametric",
      "var": 15234.56,
      "dailyVar": 4821.33,
      "portfolioVolatility": 0.0187,
      "confidenceLevel": 0.99,
      "timeHorizon": 10
    },
    "historical": {
      "method": "historical",
      "var": 16789.22,
      "percentileReturn": -0.0234,
      "sampleSize": 1000
    },
    "monteCarlo": {
      "method": "monte_carlo",
      "var": 15987.44,
      "simulations": 10000,
      "correlationIncluded": true
    },
    "cvar": {
      "cvar": 21234.88,
      "avgTailReturn": -0.0298,
      "tailSize": 100
    }
  }
}
```

### Scenario Results

```javascript
{
  "scenarioResults": {
    "2008_crisis": {
      "scenarioName": "2008 Financial Crisis",
      "initialValue": 100000,
      "finalValue": 62000,
      "totalPnL": -38000,
      "totalPnLPercent": -38.0,
      "maxDrawdown": 0.42,
      "duration": 504,
      "timeline": [
        { "day": 0, "portfolioValue": 100000, "pnl": 0 },
        { "day": 252, "portfolioValue": 62000, "pnl": -38000 }
      ]
    }
  }
}
```

### Regulatory Report

```javascript
{
  "regulatoryReport": {
    "reportType": "Basel III Market Risk Report",
    "reportDate": "2025-12-13T00:00:00Z",
    "marketRiskCapital": {
      "var99_10day": 15987.44,
      "stressedVar": 38000.00,
      "incrementalRiskCharge": 23981.16,
      "comprehensiveRiskMeasure": 31974.88
    },
    "portfolioSummary": {
      "totalValue": 100000,
      "positions": 5,
      "concentrationRisk": {
        "herfindahlIndex": 0.22,
        "effectivePositions": 4.54,
        "concentrated": false
      }
    },
    "recommendations": [
      {
        "severity": "MEDIUM",
        "category": "Diversification",
        "message": "Consider increasing allocation to defensive assets"
      }
    ]
  }
}
```

## API Reference

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `testType` | enum | Yes | Type of stress test: var_calculation, scenario_analysis, monte_carlo, historical_replay, custom_shock |
| `portfolio` | array | Yes | Array of portfolio positions with symbol, weight, quantity, currentPrice |
| `confidenceLevel` | number | No | Statistical confidence level: 0.90, 0.95, 0.99, 0.995 (default: 0.95) |
| `timeHorizon` | string | No | Risk measurement horizon: 1d, 10d, 30d, 252d (default: 10d) |
| `simulations` | integer | No | Monte Carlo iterations: 1000-100000 (default: 10000) |
| `includeCorrelations` | boolean | No | Model asset correlations (default: true) |
| `scenarios` | array | No | Crisis scenarios to apply (default: ["2008_crisis", "covid_crash"]) |
| `outputMetrics` | array | No | Metrics to calculate: var, cvar, max_drawdown, stress_pnl, recovery_time, sharpe_ratio, sortino_ratio, skewness, kurtosis |
| `generateRegulatoryReport` | boolean | No | Generate Basel III report (default: true) |
| `exportFormat` | string | No | Output format: json, csv, excel, pdf_report (default: json) |

### Output Fields

- `portfolioValue` - Total portfolio value in base currency
- `varResults` - VaR calculations (parametric, historical, Monte Carlo, CVaR)
- `scenarioResults` - Crisis scenario P&L and drawdown analysis
- `monteCarloResults` - Monte Carlo simulation statistics
- `additionalMetrics` - Sharpe, Sortino, skewness, kurtosis
- `regulatoryReport` - Basel III compliance report
- `timestamp` - Calculation timestamp (ISO 8601)

## Performance & Scalability

- **Small portfolios** (<10 assets): ~5 seconds for 10,000 simulations
- **Medium portfolios** (10-50 assets): ~15 seconds for 25,000 simulations
- **Large portfolios** (50-200 assets): ~60 seconds for 50,000 simulations
- **Memory usage**: 512MB minimum, scales with simulation count
- **Parallel processing**: Multi-core Monte Carlo computation

## Use Cases

### Hedge Funds & Asset Managers
- Daily VaR monitoring and limit compliance
- Quarterly stress testing for investor reports
- Risk-adjusted performance attribution
- Portfolio construction and optimization

### Banks & Financial Institutions
- Basel III regulatory capital calculation
- Internal risk model validation
- Trading desk risk limits
- Enterprise-wide stress testing

### Corporate Treasury
- FX and interest rate risk measurement
- Pension fund risk management
- Liquidity stress testing
- Investment policy compliance

### Compliance & Audit
- Regulatory filing preparation
- Model validation documentation
- Risk control effectiveness testing
- Audit trail generation

## Integration Examples

### Python Integration

```python
from apify_client import ApifyClient

client = ApifyClient("<YOUR_APIFY_API_TOKEN>")

run_input = {
    "testType": "var_calculation",
    "portfolio": [
        {"symbol": "SPY", "weight": 0.6, "quantity": 200, "currentPrice": 450},
        {"symbol": "TLT", "weight": 0.4, "quantity": 300, "currentPrice": 95}
    ],
    "confidenceLevel": 0.99,
    "simulations": 10000
}

run = client.actor("ruv/financial-stress-test").call(run_input=run_input)

for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    print(f"VaR (99%): ${item['varResults']['var99']:,.2f}")
    print(f"CVaR: ${item['varResults']['cvar']['cvar']:,.2f}")
```

### JavaScript Integration

```javascript
import { ApifyClient } from 'apify-client';

const client = new ApifyClient({ token: '<YOUR_APIFY_API_TOKEN>' });

const input = {
    testType: 'scenario_analysis',
    portfolio: [
        { symbol: 'SPY', weight: 0.5, quantity: 100, currentPrice: 450 },
        { symbol: 'TLT', weight: 0.5, quantity: 200, currentPrice: 95 }
    ],
    scenarios: ['2008_crisis', 'covid_crash']
};

const run = await client.actor('ruv/financial-stress-test').call(input);
const { items } = await client.dataset(run.defaultDatasetId).listItems();

items.forEach(item => {
    console.log(`Portfolio Value: $${item.portfolioValue.toLocaleString()}`);
    Object.entries(item.scenarioResults).forEach(([name, result]) => {
        console.log(`${name}: ${result.totalPnLPercent.toFixed(2)}% P&L`);
    });
});
```

## Advanced Topics

### Custom Correlation Matrices

For advanced users, you can provide custom correlation matrices in portfolio input:

```javascript
{
  "portfolio": [
    {
      "symbol": "AAPL",
      "weight": 0.5,
      "correlations": [1.0, 0.7, 0.5]  // Correlation with AAPL, MSFT, GOOGL
    },
    {
      "symbol": "MSFT",
      "weight": 0.3,
      "correlations": [0.7, 1.0, 0.6]
    },
    {
      "symbol": "GOOGL",
      "weight": 0.2,
      "correlations": [0.5, 0.6, 1.0]
    }
  ]
}
```

### Multi-Currency Portfolios

Include FX rates for multi-currency risk:

```javascript
{
  "portfolio": [
    { "symbol": "SPY", "currency": "USD", "quantity": 100, "currentPrice": 450 },
    { "symbol": "EFA", "currency": "EUR", "quantity": 200, "currentPrice": 75, "fxRate": 1.08 }
  ],
  "baseCurrency": "USD"
}
```

### Backtesting VaR Models

Validate VaR accuracy with historical backtesting:

```javascript
{
  "testType": "var_calculation",
  "backtest": {
    "enabled": true,
    "period": "1y",
    "validateExceptions": true
  }
}
```

## SEO & Keywords

**Financial Risk Management**: This actor provides comprehensive stress testing, Value at Risk calculation, scenario analysis, and regulatory reporting for institutional portfolios. Essential for quantitative risk managers, compliance officers, and portfolio managers.

**Value at Risk (VaR)**: Calculate VaR using parametric (variance-covariance), historical simulation, and Monte Carlo methods. Support for 90%, 95%, 99%, and 99.5% confidence levels across multiple time horizons.

**Stress Testing**: Apply historical crisis scenarios including 2008 financial crisis, COVID-19 crash, flash crashes, and custom market shocks. Measure maximum drawdown, recovery time, and tail risk.

**Monte Carlo Simulation**: Run up to 100,000 simulation paths with full correlation modeling using Cholesky decomposition. Generate realistic return distributions for robust risk measurement.

**Regulatory Compliance**: Generate Basel III compliant market risk capital reports including VaR, stressed VaR, incremental risk charge, and comprehensive risk measure calculations.

**Portfolio Risk**: Measure portfolio-level risk metrics including Sharpe ratio, Sortino ratio, maximum drawdown, skewness, kurtosis, and tail risk. Essential for risk-adjusted performance evaluation.

**Scenario Analysis**: Model portfolio behavior under extreme market conditions with pre-built crisis scenarios and custom shock builders. Understand potential losses and recovery dynamics.

**Risk Capital**: Calculate regulatory capital requirements for market risk under Basel III framework. Support for internal models approach and standardized approach.

**Quantitative Finance**: Advanced mathematical models for financial risk including normal distribution methods, historical simulation, and Monte Carlo techniques with correlation matrices.

**Risk Dashboard**: Integration with Apify MCP enables natural language risk queries and automated stress testing workflows with Claude AI.

## Related Tools

- **[Market Data Fetcher](https://apify.com/ruv/market-data)** - Real-time and historical market data
- **[Portfolio Optimizer](https://apify.com/ruv/portfolio-optimizer)** - Mean-variance optimization
- **[Backtesting Engine](https://apify.com/ruv/backtesting)** - Strategy validation
- **[Neural Trader](https://ruv.io)** - AI-powered trading at ruv.io

## Support & Resources

- **Documentation**: [https://github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- **Neural Trading Platform**: [https://ruv.io](https://ruv.io)
- **Issue Tracker**: [GitHub Issues](https://github.com/ruvnet/ruvector/issues)
- **Email Support**: info@ruv.io
- **Community**: [Discord](https://discord.gg/ruvector)

## License

MIT License - Copyright (c) 2025 rUv

## Footer

**Powered by rUv** - Building the future of AI-driven financial technology.

- Website: [https://ruv.io](https://ruv.io)
- GitHub: [ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- Neural Trader Platform: Advanced AI trading at [ruv.io](https://ruv.io)

**ruvector** - High-performance vector database and quantitative finance toolkit powering the next generation of financial applications.

---

*Built with Apify Actor Platform | Deployed on Global Infrastructure | Enterprise Support Available*
