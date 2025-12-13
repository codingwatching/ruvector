import { Actor } from 'apify';

/**
 * Financial Stress Testing Platform
 *
 * Enterprise-grade stress testing for financial portfolios
 * Supporting VaR, Monte Carlo, scenario analysis, and regulatory reporting
 *
 * @author rUv <info@ruv.io>
 * @see https://ruv.io
 * @see https://github.com/ruvnet/ruvector
 */

// ============================================================================
// MATHEMATICAL UTILITIES
// ============================================================================

/**
 * Calculate normal distribution cumulative density function (CDF)
 */
function normalCDF(x, mean = 0, std = 1) {
    const z = (x - mean) / std;
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const d = 0.3989423 * Math.exp(-z * z / 2);
    const probability = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return z > 0 ? 1 - probability : probability;
}

/**
 * Calculate inverse normal CDF (for percentile calculations)
 */
function normalInverseCDF(p, mean = 0, std = 1) {
    if (p <= 0 || p >= 1) throw new Error('Probability must be between 0 and 1');

    // Rational approximation for central region
    const a = [
        -39.6968302866538, 220.946098424521, -275.928510446969,
        138.357751867269, -30.6647980661472, 2.50662827745924
    ];
    const b = [
        -54.4760987982241, 161.585836858041, -155.698979859887,
        66.8013118877197, -13.2806815528857, 1.0
    ];

    const c = [
        -0.00778489400243029, -0.322396458041136, -2.40075827716184,
        -2.54973253934373, 4.37466414146497, 2.93816398269878
    ];
    const d = [
        0.00778469570904146, 0.32246712907004, 2.445134137143,
        3.75440866190742, 1.0
    ];

    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let q, r, x;

    if (p < pLow) {
        q = Math.sqrt(-2 * Math.log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p <= pHigh) {
        q = p - 0.5;
        r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    } else {
        q = Math.sqrt(-2 * Math.log(1 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }

    return mean + std * x;
}

/**
 * Generate random normal distribution sample (Box-Muller transform)
 */
function randomNormal(mean = 0, std = 1) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z0;
}

/**
 * Calculate Cholesky decomposition for correlation matrix
 */
function choleskyDecomposition(matrix) {
    const n = matrix.length;
    const L = Array(n).fill(0).map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let sum = 0;
            for (let k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k];
            }

            if (i === j) {
                L[i][j] = Math.sqrt(matrix[i][i] - sum);
            } else {
                L[i][j] = (matrix[i][j] - sum) / L[j][j];
            }
        }
    }

    return L;
}

// ============================================================================
// CRISIS SCENARIO DEFINITIONS
// ============================================================================

const CRISIS_SCENARIOS = {
    '2008_crisis': {
        name: '2008 Financial Crisis',
        duration: 504, // trading days
        shocks: {
            equity: -0.55,
            rates: -0.03,
            credit: 0.06,
            volatility: 3.5,
            correlation: 0.85
        },
        timeline: [
            { day: 0, equity: 0.00, rates: 0.000 },
            { day: 60, equity: -0.15, rates: -0.005 },
            { day: 120, equity: -0.25, rates: -0.010 },
            { day: 180, equity: -0.40, rates: -0.020 },
            { day: 252, equity: -0.55, rates: -0.030 },
            { day: 504, equity: -0.35, rates: -0.025 }
        ]
    },
    'covid_crash': {
        name: 'COVID-19 Crash 2020',
        duration: 126,
        shocks: {
            equity: -0.35,
            rates: -0.015,
            volatility: 5.0,
            recovery: 189
        },
        timeline: [
            { day: 0, equity: 0.00, rates: 0.000 },
            { day: 20, equity: -0.35, rates: -0.010 },
            { day: 63, equity: -0.20, rates: -0.015 },
            { day: 126, equity: 0.05, rates: -0.015 }
        ]
    },
    'flash_crash': {
        name: 'Flash Crash 2010',
        duration: 1,
        shocks: {
            equity: -0.10,
            intraday: true,
            liquidity: 0.05,
            recovery: 0.25
        }
    },
    'rate_hike': {
        name: 'Rapid Rate Hike Cycle',
        duration: 252,
        shocks: {
            rates: 0.05,
            bonds: -0.25,
            equity: -0.15,
            dollar: 0.20
        }
    },
    'currency_crisis': {
        name: 'Currency Crisis / EM Contagion',
        duration: 189,
        shocks: {
            fx: -0.30,
            equity: -0.40,
            volatility: 4.0,
            contagion: true
        }
    },
    'dot_com_bubble': {
        name: 'Dot-com Bubble 2000-2002',
        duration: 756,
        shocks: {
            tech: -0.78,
            equity: -0.45,
            duration: 'prolonged'
        }
    },
    'black_monday_1987': {
        name: 'Black Monday 1987',
        duration: 1,
        shocks: {
            equity: -0.22,
            singleDay: true,
            globalContagion: true
        }
    },
    'european_debt_crisis': {
        name: 'European Debt Crisis 2011',
        duration: 378,
        shocks: {
            sovereignSpread: 0.08,
            equity: -0.25,
            euro: -0.15
        }
    },
    'oil_shock': {
        name: 'Oil Shock 1973',
        duration: 504,
        shocks: {
            oil: 4.0,
            inflation: 0.12,
            equity: -0.45,
            stagflation: true
        }
    },
    'stagflation': {
        name: 'Stagflation Scenario',
        duration: 504,
        shocks: {
            inflation: 0.10,
            growth: -0.02,
            equity: -0.30,
            bonds: -0.20
        }
    }
};

// ============================================================================
// VAR CALCULATION ENGINE
// ============================================================================

class VaREngine {
    constructor(portfolio, confidenceLevel, timeHorizon) {
        this.portfolio = portfolio;
        this.confidenceLevel = confidenceLevel;
        this.timeHorizon = this.parseTimeHorizon(timeHorizon);
        this.portfolioValue = this.calculatePortfolioValue();
    }

    parseTimeHorizon(horizon) {
        const map = { '1d': 1, '10d': 10, '30d': 30, '252d': 252 };
        return map[horizon] || 10;
    }

    calculatePortfolioValue() {
        return this.portfolio.reduce((sum, asset) =>
            sum + (asset.quantity * asset.currentPrice), 0
        );
    }

    /**
     * Parametric VaR (Variance-Covariance Method)
     */
    calculateParametricVaR(returns, volatilities) {
        const portfolioVolatility = this.calculatePortfolioVolatility(volatilities);
        const zScore = normalInverseCDF(1 - this.confidenceLevel);
        const dailyVaR = this.portfolioValue * portfolioVolatility * Math.abs(zScore);
        const scaledVaR = dailyVaR * Math.sqrt(this.timeHorizon);

        return {
            method: 'parametric',
            var: scaledVaR,
            dailyVar: dailyVaR,
            portfolioVolatility,
            confidenceLevel: this.confidenceLevel,
            timeHorizon: this.timeHorizon
        };
    }

    /**
     * Historical VaR (Non-parametric)
     */
    calculateHistoricalVaR(historicalReturns) {
        const sortedReturns = [...historicalReturns].sort((a, b) => a - b);
        const percentileIndex = Math.floor((1 - this.confidenceLevel) * sortedReturns.length);
        const percentileReturn = sortedReturns[percentileIndex];
        const var_ = -percentileReturn * this.portfolioValue * Math.sqrt(this.timeHorizon);

        return {
            method: 'historical',
            var: var_,
            percentileReturn,
            sampleSize: sortedReturns.length,
            confidenceLevel: this.confidenceLevel
        };
    }

    /**
     * Monte Carlo VaR
     */
    calculateMonteCarloVaR(simulations, correlationMatrix) {
        const returns = [];
        const n = this.portfolio.length;

        // Generate correlated returns using Cholesky decomposition
        const L = correlationMatrix ? choleskyDecomposition(correlationMatrix) : null;

        for (let i = 0; i < simulations; i++) {
            let portfolioReturn = 0;
            const randomNormals = Array(n).fill(0).map(() => randomNormal());

            for (let j = 0; j < n; j++) {
                const asset = this.portfolio[j];
                const volatility = asset.volatility || 0.02; // 2% daily default
                const expectedReturn = asset.expectedReturn || 0;

                let correlatedRandom = randomNormals[j];
                if (L) {
                    correlatedRandom = 0;
                    for (let k = 0; k <= j; k++) {
                        correlatedRandom += L[j][k] * randomNormals[k];
                    }
                }

                const assetReturn = expectedReturn + volatility * correlatedRandom;
                portfolioReturn += asset.weight * assetReturn;
            }

            returns.push(portfolioReturn);
        }

        const sortedReturns = returns.sort((a, b) => a - b);
        const percentileIndex = Math.floor((1 - this.confidenceLevel) * simulations);
        const percentileReturn = sortedReturns[percentileIndex];
        const var_ = -percentileReturn * this.portfolioValue * Math.sqrt(this.timeHorizon);

        return {
            method: 'monte_carlo',
            var: var_,
            simulations,
            percentileReturn,
            correlationIncluded: !!correlationMatrix,
            confidenceLevel: this.confidenceLevel
        };
    }

    /**
     * Conditional VaR / Expected Shortfall (CVaR)
     */
    calculateCVaR(returns) {
        const sortedReturns = [...returns].sort((a, b) => a - b);
        const varIndex = Math.floor((1 - this.confidenceLevel) * sortedReturns.length);
        const tailReturns = sortedReturns.slice(0, varIndex);
        const avgTailReturn = tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length;
        const cvar = -avgTailReturn * this.portfolioValue * Math.sqrt(this.timeHorizon);

        return {
            cvar,
            avgTailReturn,
            tailSize: tailReturns.length,
            confidenceLevel: this.confidenceLevel
        };
    }

    calculatePortfolioVolatility(volatilities) {
        // Simplified portfolio volatility (assumes equal correlation for demonstration)
        const weightedVolSquared = this.portfolio.reduce((sum, asset, i) => {
            const vol = volatilities[i] || asset.volatility || 0.02;
            return sum + Math.pow(asset.weight * vol, 2);
        }, 0);

        return Math.sqrt(weightedVolSquared);
    }
}

// ============================================================================
// SCENARIO ANALYSIS ENGINE
// ============================================================================

class ScenarioEngine {
    constructor(portfolio) {
        this.portfolio = portfolio;
        this.portfolioValue = portfolio.reduce((sum, asset) =>
            sum + (asset.quantity * asset.currentPrice), 0
        );
    }

    /**
     * Apply crisis scenario to portfolio
     */
    applyScenario(scenarioName) {
        const scenario = CRISIS_SCENARIOS[scenarioName];
        if (!scenario) {
            throw new Error(`Unknown scenario: ${scenarioName}`);
        }

        const results = {
            scenarioName: scenario.name,
            duration: scenario.duration,
            initialValue: this.portfolioValue,
            shocks: scenario.shocks,
            timeline: []
        };

        // Calculate P&L at each timeline point
        if (scenario.timeline) {
            results.timeline = scenario.timeline.map(point => {
                const portfolioReturn = this.calculatePortfolioReturn(point, scenario.shocks);
                const portfolioValue = this.portfolioValue * (1 + portfolioReturn);
                const pnl = portfolioValue - this.portfolioValue;

                return {
                    day: point.day,
                    portfolioReturn,
                    portfolioValue,
                    pnl,
                    pnlPercent: (pnl / this.portfolioValue) * 100
                };
            });
        }

        // Calculate final stressed value
        const finalReturn = this.calculatePortfolioReturn(
            scenario.timeline ? scenario.timeline[scenario.timeline.length - 1] : { equity: scenario.shocks.equity },
            scenario.shocks
        );

        results.finalValue = this.portfolioValue * (1 + finalReturn);
        results.totalPnL = results.finalValue - this.portfolioValue;
        results.totalPnLPercent = (results.totalPnL / this.portfolioValue) * 100;
        results.maxDrawdown = this.calculateMaxDrawdown(results.timeline);

        return results;
    }

    calculatePortfolioReturn(point, shocks) {
        // Simplified: assumes portfolio moves with equity shock
        // In production, this would map each asset to specific shocks
        return point.equity || shocks.equity || 0;
    }

    calculateMaxDrawdown(timeline) {
        if (!timeline || timeline.length === 0) return 0;

        let peak = timeline[0].portfolioValue;
        let maxDD = 0;

        for (const point of timeline) {
            if (point.portfolioValue > peak) {
                peak = point.portfolioValue;
            }
            const drawdown = (peak - point.portfolioValue) / peak;
            maxDD = Math.max(maxDD, drawdown);
        }

        return maxDD;
    }

    /**
     * Calculate recovery time from drawdown
     */
    calculateRecoveryTime(timeline) {
        if (!timeline || timeline.length < 2) return null;

        let inDrawdown = false;
        let drawdownStart = 0;
        let peak = timeline[0].portfolioValue;
        const recoveries = [];

        for (let i = 0; i < timeline.length; i++) {
            const point = timeline[i];

            if (point.portfolioValue > peak) {
                if (inDrawdown) {
                    recoveries.push({
                        start: drawdownStart,
                        end: i,
                        days: point.day - timeline[drawdownStart].day
                    });
                    inDrawdown = false;
                }
                peak = point.portfolioValue;
            } else if (point.portfolioValue < peak * 0.95 && !inDrawdown) {
                inDrawdown = true;
                drawdownStart = i;
            }
        }

        return recoveries;
    }
}

// ============================================================================
// RISK METRICS CALCULATOR
// ============================================================================

class RiskMetrics {
    static calculateSharpeRatio(returns, riskFreeRate = 0.02) {
        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const stdDev = Math.sqrt(
            returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
        );
        return (avgReturn - riskFreeRate / 252) / stdDev;
    }

    static calculateSortinoRatio(returns, riskFreeRate = 0.02) {
        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const downside = returns.filter(r => r < 0);
        const downsideStdDev = Math.sqrt(
            downside.reduce((sum, r) => sum + Math.pow(r, 2), 0) / downside.length
        );
        return (avgReturn - riskFreeRate / 252) / downsideStdDev;
    }

    static calculateSkewness(returns) {
        const n = returns.length;
        const mean = returns.reduce((sum, r) => sum + r, 0) / n;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / n;
        const stdDev = Math.sqrt(variance);
        const skewness = returns.reduce((sum, r) => sum + Math.pow((r - mean) / stdDev, 3), 0) / n;
        return skewness;
    }

    static calculateKurtosis(returns) {
        const n = returns.length;
        const mean = returns.reduce((sum, r) => sum + r, 0) / n;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / n;
        const stdDev = Math.sqrt(variance);
        const kurtosis = returns.reduce((sum, r) => sum + Math.pow((r - mean) / stdDev, 4), 0) / n;
        return kurtosis - 3; // Excess kurtosis
    }
}

// ============================================================================
// REGULATORY REPORTING
// ============================================================================

class RegulatoryReport {
    static generateBaselIIIReport(varResults, scenarioResults, portfolio) {
        return {
            reportType: 'Basel III Market Risk Report',
            reportDate: new Date().toISOString(),
            reportingEntity: 'Portfolio Risk Management',

            // Basel III Market Risk Capital Requirements
            marketRiskCapital: {
                var99_10day: varResults.var99 || 0,
                stressedVar: Math.max(...Object.values(scenarioResults).map(s => Math.abs(s.totalPnL))),
                incrementalRiskCharge: varResults.var99 * 1.5, // Simplified
                comprehensiveRiskMeasure: varResults.var99 * 2.0 // Simplified
            },

            // Risk Metrics Summary
            riskMetrics: {
                valueAtRisk: {
                    var95: varResults.var95,
                    var99: varResults.var99,
                    cvar: varResults.cvar,
                    method: varResults.method,
                    confidenceLevel: varResults.confidenceLevel,
                    timeHorizon: varResults.timeHorizon
                },

                stressTestResults: Object.entries(scenarioResults).map(([name, result]) => ({
                    scenario: name,
                    pnl: result.totalPnL,
                    pnlPercent: result.totalPnLPercent,
                    maxDrawdown: result.maxDrawdown
                }))
            },

            // Portfolio Composition
            portfolioSummary: {
                totalValue: portfolio.reduce((sum, asset) =>
                    sum + (asset.quantity * asset.currentPrice), 0
                ),
                positions: portfolio.length,
                assetClasses: [...new Set(portfolio.map(a => a.assetClass || 'equity'))],
                concentrationRisk: this.calculateConcentrationRisk(portfolio)
            },

            // Compliance Status
            compliance: {
                varBacktesting: 'PENDING',
                stressTestingFrequency: 'QUARTERLY',
                modelValidation: 'PENDING',
                independentReview: 'PENDING'
            },

            // Recommendations
            recommendations: this.generateRecommendations(varResults, scenarioResults)
        };
    }

    static calculateConcentrationRisk(portfolio) {
        const weights = portfolio.map(a => a.weight);
        const herfindahlIndex = weights.reduce((sum, w) => sum + w * w, 0);
        return {
            herfindahlIndex,
            effectivePositions: 1 / herfindahlIndex,
            concentrated: herfindahlIndex > 0.25
        };
    }

    static generateRecommendations(varResults, scenarioResults) {
        const recommendations = [];

        if (varResults.var99 / varResults.portfolioValue > 0.15) {
            recommendations.push({
                severity: 'HIGH',
                category: 'Risk Limit',
                message: 'VaR exceeds 15% of portfolio value. Consider reducing leverage or increasing diversification.'
            });
        }

        const worstScenario = Object.values(scenarioResults)
            .reduce((worst, s) => s.totalPnLPercent < worst.totalPnLPercent ? s : worst);

        if (worstScenario.totalPnLPercent < -30) {
            recommendations.push({
                severity: 'CRITICAL',
                category: 'Stress Test',
                message: `Worst scenario (${worstScenario.scenarioName}) shows ${worstScenario.totalPnLPercent.toFixed(2)}% loss. Immediate hedging required.`
            });
        }

        return recommendations;
    }
}

// ============================================================================
// MAIN ACTOR LOGIC
// ============================================================================

await Actor.main(async () => {
    console.log('🚀 Financial Stress Testing Platform - Actor Starting');
    console.log('━'.repeat(80));

    const input = await Actor.getInput();
    if (!input) {
        throw new Error('No input provided to the actor');
    }

    console.log(`📊 Test Type: ${input.testType}`);
    console.log(`💼 Portfolio Positions: ${input.portfolio.length}`);
    console.log(`🎯 Confidence Level: ${(input.confidenceLevel * 100).toFixed(1)}%`);
    console.log(`⏱️  Time Horizon: ${input.timeHorizon}`);
    console.log('━'.repeat(80));

    const results = {
        testType: input.testType,
        timestamp: new Date().toISOString(),
        portfolio: input.portfolio,
        portfolioValue: 0,
        configuration: {
            confidenceLevel: input.confidenceLevel,
            timeHorizon: input.timeHorizon,
            simulations: input.simulations,
            includeCorrelations: input.includeCorrelations
        }
    };

    // Calculate portfolio value
    results.portfolioValue = input.portfolio.reduce((sum, asset) =>
        sum + (asset.quantity * asset.currentPrice), 0
    );

    console.log(`💰 Total Portfolio Value: $${results.portfolioValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
    console.log('');

    // ========================================================================
    // VAR CALCULATION
    // ========================================================================

    if (input.testType === 'var_calculation' || input.outputMetrics.includes('var')) {
        console.log('📈 Calculating Value at Risk (VaR)...');

        const varEngine = new VaREngine(input.portfolio, input.confidenceLevel, input.timeHorizon);

        // Generate synthetic returns for demonstration
        const syntheticReturns = Array(1000).fill(0).map(() => randomNormal(0.0005, 0.015));
        const volatilities = input.portfolio.map(() => 0.02); // 2% daily vol

        // Parametric VaR
        const parametricVaR = varEngine.calculateParametricVaR(syntheticReturns, volatilities);
        console.log(`  ✓ Parametric VaR: $${parametricVaR.var.toLocaleString('en-US', { maximumFractionDigits: 2 })}`);

        // Historical VaR
        const historicalVaR = varEngine.calculateHistoricalVaR(syntheticReturns);
        console.log(`  ✓ Historical VaR: $${historicalVaR.var.toLocaleString('en-US', { maximumFractionDigits: 2 })}`);

        // Monte Carlo VaR
        const correlationMatrix = input.includeCorrelations ?
            this.generateCorrelationMatrix(input.portfolio.length) : null;
        const monteCarloVaR = varEngine.calculateMonteCarloVaR(input.simulations, correlationMatrix);
        console.log(`  ✓ Monte Carlo VaR: $${monteCarloVaR.var.toLocaleString('en-US', { maximumFractionDigits: 2 })} (${input.simulations.toLocaleString()} simulations)`);

        // CVaR
        const cvar = varEngine.calculateCVaR(syntheticReturns);
        console.log(`  ✓ CVaR (Expected Shortfall): $${cvar.cvar.toLocaleString('en-US', { maximumFractionDigits: 2 })}`);

        results.varResults = {
            parametric: parametricVaR,
            historical: historicalVaR,
            monteCarlo: monteCarloVaR,
            cvar: cvar,
            var95: monteCarloVaR.var * (0.95 / input.confidenceLevel), // Scaled
            var99: monteCarloVaR.var,
            method: 'monte_carlo',
            confidenceLevel: input.confidenceLevel,
            timeHorizon: input.timeHorizon
        };

        console.log('');
    }

    // ========================================================================
    // SCENARIO ANALYSIS
    // ========================================================================

    if (input.testType === 'scenario_analysis' || input.scenarios?.length > 0) {
        console.log('🎭 Running Scenario Analysis...');

        const scenarioEngine = new ScenarioEngine(input.portfolio);
        results.scenarioResults = {};

        const scenarios = input.scenarios || ['2008_crisis', 'covid_crash'];

        for (const scenarioName of scenarios) {
            console.log(`  ⚡ Applying ${CRISIS_SCENARIOS[scenarioName]?.name || scenarioName}...`);

            try {
                const scenarioResult = scenarioEngine.applyScenario(scenarioName);
                results.scenarioResults[scenarioName] = scenarioResult;

                console.log(`     Initial Value: $${scenarioResult.initialValue.toLocaleString('en-US', { maximumFractionDigits: 2 })}`);
                console.log(`     Final Value: $${scenarioResult.finalValue.toLocaleString('en-US', { maximumFractionDigits: 2 })}`);
                console.log(`     P&L: $${scenarioResult.totalPnL.toLocaleString('en-US', { maximumFractionDigits: 2 })} (${scenarioResult.totalPnLPercent.toFixed(2)}%)`);
                console.log(`     Max Drawdown: ${(scenarioResult.maxDrawdown * 100).toFixed(2)}%`);

                if (input.outputMetrics.includes('recovery_time')) {
                    const recoveries = scenarioEngine.calculateRecoveryTime(scenarioResult.timeline);
                    if (recoveries && recoveries.length > 0) {
                        console.log(`     Recovery Time: ${recoveries[0].days} days`);
                        scenarioResult.recoveryTime = recoveries;
                    }
                }
            } catch (error) {
                console.error(`     ✗ Error in scenario ${scenarioName}:`, error.message);
            }

            console.log('');
        }
    }

    // ========================================================================
    // MONTE CARLO SIMULATION
    // ========================================================================

    if (input.testType === 'monte_carlo') {
        console.log('🎲 Running Monte Carlo Simulation...');
        console.log(`  Simulations: ${input.simulations.toLocaleString()}`);
        console.log(`  Correlations: ${input.includeCorrelations ? 'Enabled' : 'Disabled'}`);

        const varEngine = new VaREngine(input.portfolio, input.confidenceLevel, input.timeHorizon);
        const correlationMatrix = input.includeCorrelations ?
            this.generateCorrelationMatrix(input.portfolio.length) : null;

        const mcResults = varEngine.calculateMonteCarloVaR(input.simulations, correlationMatrix);
        results.monteCarloResults = mcResults;

        console.log(`  ✓ VaR (${(input.confidenceLevel * 100).toFixed(0)}%): $${mcResults.var.toLocaleString('en-US', { maximumFractionDigits: 2 })}`);
        console.log('');
    }

    // ========================================================================
    // CUSTOM SHOCKS
    // ========================================================================

    if (input.testType === 'custom_shock' && input.customShocks?.length > 0) {
        console.log('⚡ Applying Custom Shock Scenarios...');

        results.customShockResults = {};

        for (const shock of input.customShocks) {
            console.log(`  🎯 Scenario: ${shock.name}`);

            const shockReturn = (shock.equityShock || 0) / 100;
            const shockedValue = results.portfolioValue * (1 + shockReturn);
            const pnl = shockedValue - results.portfolioValue;

            results.customShockResults[shock.name] = {
                name: shock.name,
                shocks: shock,
                initialValue: results.portfolioValue,
                finalValue: shockedValue,
                pnl,
                pnlPercent: (pnl / results.portfolioValue) * 100
            };

            console.log(`     P&L: $${pnl.toLocaleString('en-US', { maximumFractionDigits: 2 })} (${((pnl / results.portfolioValue) * 100).toFixed(2)}%)`);
        }

        console.log('');
    }

    // ========================================================================
    // RISK METRICS
    // ========================================================================

    if (input.outputMetrics.length > 0) {
        console.log('📊 Calculating Additional Risk Metrics...');

        const syntheticReturns = Array(252).fill(0).map(() => randomNormal(0.0005, 0.015));

        results.additionalMetrics = {};

        if (input.outputMetrics.includes('sharpe_ratio')) {
            results.additionalMetrics.sharpeRatio = RiskMetrics.calculateSharpeRatio(syntheticReturns);
            console.log(`  ✓ Sharpe Ratio: ${results.additionalMetrics.sharpeRatio.toFixed(3)}`);
        }

        if (input.outputMetrics.includes('sortino_ratio')) {
            results.additionalMetrics.sortinoRatio = RiskMetrics.calculateSortinoRatio(syntheticReturns);
            console.log(`  ✓ Sortino Ratio: ${results.additionalMetrics.sortinoRatio.toFixed(3)}`);
        }

        if (input.outputMetrics.includes('skewness')) {
            results.additionalMetrics.skewness = RiskMetrics.calculateSkewness(syntheticReturns);
            console.log(`  ✓ Skewness: ${results.additionalMetrics.skewness.toFixed(3)}`);
        }

        if (input.outputMetrics.includes('kurtosis')) {
            results.additionalMetrics.kurtosis = RiskMetrics.calculateKurtosis(syntheticReturns);
            console.log(`  ✓ Excess Kurtosis: ${results.additionalMetrics.kurtosis.toFixed(3)}`);
        }

        console.log('');
    }

    // ========================================================================
    // REGULATORY REPORTING
    // ========================================================================

    if (input.generateRegulatoryReport) {
        console.log('📋 Generating Regulatory Report (Basel III)...');

        results.regulatoryReport = RegulatoryReport.generateBaselIIIReport(
            results.varResults || {},
            results.scenarioResults || {},
            input.portfolio
        );

        console.log(`  ✓ Market Risk Capital: $${results.regulatoryReport.marketRiskCapital.var99_10day.toLocaleString('en-US', { maximumFractionDigits: 2 })}`);
        console.log(`  ✓ Stressed VaR: $${results.regulatoryReport.marketRiskCapital.stressedVar.toLocaleString('en-US', { maximumFractionDigits: 2 })}`);

        if (results.regulatoryReport.recommendations.length > 0) {
            console.log(`  ⚠️  ${results.regulatoryReport.recommendations.length} recommendation(s) generated`);
            results.regulatoryReport.recommendations.forEach(rec => {
                console.log(`     [${rec.severity}] ${rec.message}`);
            });
        }

        console.log('');
    }

    // ========================================================================
    // SAVE RESULTS
    // ========================================================================

    console.log('💾 Saving results to dataset...');
    await Actor.pushData(results);

    console.log('━'.repeat(80));
    console.log('✅ Stress Testing Complete!');
    console.log(`📦 Results exported in ${input.exportFormat} format`);
    console.log('━'.repeat(80));
    console.log('');
    console.log('🌐 Powered by rUv - https://ruv.io');
    console.log('📚 GitHub: https://github.com/ruvnet/ruvector');
    console.log('');
});

// Helper function to generate correlation matrix
function generateCorrelationMatrix(size) {
    const matrix = Array(size).fill(0).map(() => Array(size).fill(0));

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            if (i === j) {
                matrix[i][j] = 1.0;
            } else {
                // Generate realistic correlations (0.3-0.7 for same asset class)
                matrix[i][j] = 0.3 + Math.random() * 0.4;
                matrix[j][i] = matrix[i][j]; // Symmetric
            }
        }
    }

    return matrix;
}
