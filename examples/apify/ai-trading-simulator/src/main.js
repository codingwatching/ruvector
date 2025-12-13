import { Actor } from 'apify';

// ============================================================================
// TECHNICAL INDICATORS
// ============================================================================

class TechnicalIndicators {
    static sma(data, period) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                result.push(null);
                continue;
            }
            const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            result.push(sum / period);
        }
        return result;
    }

    static ema(data, period) {
        const result = [];
        const multiplier = 2 / (period + 1);
        let ema = data[0];
        result.push(ema);

        for (let i = 1; i < data.length; i++) {
            ema = (data[i] - ema) * multiplier + ema;
            result.push(ema);
        }
        return result;
    }

    static rsi(data, period = 14) {
        const result = [];
        const changes = [];

        for (let i = 1; i < data.length; i++) {
            changes.push(data[i] - data[i - 1]);
        }

        for (let i = 0; i < changes.length; i++) {
            if (i < period - 1) {
                result.push(null);
                continue;
            }

            const gains = changes.slice(i - period + 1, i + 1).filter(c => c > 0);
            const losses = changes.slice(i - period + 1, i + 1).filter(c => c < 0).map(c => Math.abs(c));

            const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / period : 0;
            const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / period : 0;

            if (avgLoss === 0) {
                result.push(100);
            } else {
                const rs = avgGain / avgLoss;
                const rsi = 100 - (100 / (1 + rs));
                result.push(rsi);
            }
        }

        result.unshift(null);
        return result;
    }

    static macd(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
        const fastEMA = this.ema(data, fastPeriod);
        const slowEMA = this.ema(data, slowPeriod);
        const macdLine = fastEMA.map((fast, i) => fast - slowEMA[i]);
        const signalLine = this.ema(macdLine, signalPeriod);
        const histogram = macdLine.map((macd, i) => macd - signalLine[i]);

        return { macdLine, signalLine, histogram };
    }

    static bollingerBands(data, period = 20, stdDev = 2) {
        const sma = this.sma(data, period);
        const upper = [];
        const lower = [];

        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                upper.push(null);
                lower.push(null);
                continue;
            }

            const slice = data.slice(i - period + 1, i + 1);
            const mean = sma[i];
            const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
            const std = Math.sqrt(variance);

            upper.push(mean + (stdDev * std));
            lower.push(mean - (stdDev * std));
        }

        return { upper, middle: sma, lower };
    }
}

// ============================================================================
// MARKET DATA GENERATOR
// ============================================================================

class MarketDataGenerator {
    constructor(config) {
        this.config = config;
        this.rng = this.seededRandom(config.seed || Date.now());
    }

    seededRandom(seed) {
        let state = seed;
        return () => {
            state = (state * 1664525 + 1013904223) % 4294967296;
            return state / 4294967296;
        };
    }

    generateOHLCV() {
        const { dataPoints, volatility, trend, initialPrice, symbol } = this.config;
        const basePrice = initialPrice || this.getDefaultPrice(symbol);
        const data = [];

        let currentPrice = basePrice;
        const trendFactor = this.getTrendFactor(trend);

        const now = new Date();
        const interval = this.getIntervalMs(this.config.timeframe);

        for (let i = 0; i < dataPoints; i++) {
            const timestamp = new Date(now.getTime() - (dataPoints - i) * interval);

            // Apply trend
            const trendChange = currentPrice * trendFactor * (this.rng() - 0.5) * 2;
            currentPrice += trendChange;

            // Generate OHLCV
            const open = currentPrice;
            const volatilityRange = currentPrice * volatility;

            const high = open + (this.rng() * volatilityRange);
            const low = open - (this.rng() * volatilityRange);
            const close = low + this.rng() * (high - low);
            const volume = Math.floor((1000000 + this.rng() * 5000000) * (1 + volatility));

            data.push({
                timestamp: timestamp.toISOString(),
                symbol: symbol,
                open: parseFloat(open.toFixed(2)),
                high: parseFloat(high.toFixed(2)),
                low: parseFloat(low.toFixed(2)),
                close: parseFloat(close.toFixed(2)),
                volume: volume
            });

            currentPrice = close;
        }

        return data;
    }

    getTrendFactor(trend) {
        switch (trend) {
            case 'bullish': return 0.002;
            case 'bearish': return -0.002;
            case 'volatile': return 0.001;
            case 'sideways':
            default: return 0.0005;
        }
    }

    getDefaultPrice(symbol) {
        const defaults = {
            'AAPL': 175,
            'TSLA': 250,
            'MSFT': 380,
            'GOOGL': 140,
            'AMZN': 155,
            'BTC-USD': 45000,
            'ETH-USD': 2500,
            'SPY': 450
        };
        return defaults[symbol] || 100;
    }

    getIntervalMs(timeframe) {
        const intervals = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        };
        return intervals[timeframe] || intervals['1h'];
    }

    generateOrderBook(price) {
        const bids = [];
        const asks = [];
        const spread = price * 0.001; // 0.1% spread

        for (let i = 0; i < 10; i++) {
            bids.push({
                price: parseFloat((price - spread * (i + 1)).toFixed(2)),
                size: Math.floor(100 + this.rng() * 900)
            });
            asks.push({
                price: parseFloat((price + spread * (i + 1)).toFixed(2)),
                size: Math.floor(100 + this.rng() * 900)
            });
        }

        return { bids, asks };
    }

    generateNews(price, previousPrice) {
        const sentiment = price > previousPrice ? 'positive' : price < previousPrice ? 'negative' : 'neutral';
        const templates = {
            positive: [
                `${this.config.symbol} surges on strong earnings report`,
                `Analysts upgrade ${this.config.symbol} price target`,
                `${this.config.symbol} announces breakthrough innovation`
            ],
            negative: [
                `${this.config.symbol} drops amid regulatory concerns`,
                `Analysts downgrade ${this.config.symbol} outlook`,
                `${this.config.symbol} faces market headwinds`
            ],
            neutral: [
                `${this.config.symbol} trades sideways in quiet session`,
                `Market awaits ${this.config.symbol} earnings report`,
                `${this.config.symbol} consolidates recent gains`
            ]
        };

        const headlines = templates[sentiment];
        const headline = headlines[Math.floor(this.rng() * headlines.length)];

        return {
            headline,
            sentiment,
            score: sentiment === 'positive' ? 0.6 + this.rng() * 0.4 :
                   sentiment === 'negative' ? this.rng() * 0.4 :
                   0.4 + this.rng() * 0.2
        };
    }
}

// ============================================================================
// TRADING STRATEGIES
// ============================================================================

class TradingStrategies {
    static meanReversion(data, indicators) {
        const signals = [];
        const { upper, lower, middle } = indicators.bb;

        for (let i = 0; i < data.length; i++) {
            if (!upper[i] || !lower[i]) {
                signals.push({ action: 'hold', strength: 0 });
                continue;
            }

            const close = data[i].close;
            if (close <= lower[i]) {
                signals.push({ action: 'buy', strength: 0.8, reason: 'Price below lower BB' });
            } else if (close >= upper[i]) {
                signals.push({ action: 'sell', strength: 0.8, reason: 'Price above upper BB' });
            } else {
                signals.push({ action: 'hold', strength: 0 });
            }
        }

        return signals;
    }

    static momentum(data, indicators) {
        const signals = [];
        const { macdLine, signalLine } = indicators.macd;

        for (let i = 1; i < data.length; i++) {
            if (!macdLine[i] || !signalLine[i] || !macdLine[i-1] || !signalLine[i-1]) {
                signals.push({ action: 'hold', strength: 0 });
                continue;
            }

            const crossover = macdLine[i-1] <= signalLine[i-1] && macdLine[i] > signalLine[i];
            const crossunder = macdLine[i-1] >= signalLine[i-1] && macdLine[i] < signalLine[i];

            if (crossover) {
                signals.push({ action: 'buy', strength: 0.9, reason: 'MACD bullish crossover' });
            } else if (crossunder) {
                signals.push({ action: 'sell', strength: 0.9, reason: 'MACD bearish crossunder' });
            } else {
                signals.push({ action: 'hold', strength: 0 });
            }
        }

        signals.unshift({ action: 'hold', strength: 0 });
        return signals;
    }

    static breakout(data, indicators) {
        const signals = [];
        const period = 20;

        for (let i = 0; i < data.length; i++) {
            if (i < period) {
                signals.push({ action: 'hold', strength: 0 });
                continue;
            }

            const recentData = data.slice(i - period, i);
            const highestHigh = Math.max(...recentData.map(d => d.high));
            const lowestLow = Math.min(...recentData.map(d => d.low));

            if (data[i].close > highestHigh) {
                signals.push({ action: 'buy', strength: 0.85, reason: 'Breakout above resistance' });
            } else if (data[i].close < lowestLow) {
                signals.push({ action: 'sell', strength: 0.85, reason: 'Breakdown below support' });
            } else {
                signals.push({ action: 'hold', strength: 0 });
            }
        }

        return signals;
    }

    static mlNeural(data, indicators) {
        const signals = [];

        // Simple neural-inspired model using multiple indicators
        for (let i = 0; i < data.length; i++) {
            if (!indicators.rsi[i] || !indicators.macd.macdLine[i]) {
                signals.push({ action: 'hold', strength: 0 });
                continue;
            }

            const rsi = indicators.rsi[i];
            const macdHist = indicators.macd.histogram[i];
            const sma = indicators.sma[i];
            const close = data[i].close;

            // Weighted decision
            let score = 0;

            // RSI signals
            if (rsi < 30) score += 0.4; // Oversold
            if (rsi > 70) score -= 0.4; // Overbought

            // MACD signals
            if (macdHist > 0) score += 0.3;
            if (macdHist < 0) score -= 0.3;

            // Trend signals
            if (sma && close > sma) score += 0.3;
            if (sma && close < sma) score -= 0.3;

            if (score > 0.5) {
                signals.push({ action: 'buy', strength: Math.min(score, 1), reason: 'ML neural buy signal' });
            } else if (score < -0.5) {
                signals.push({ action: 'sell', strength: Math.min(Math.abs(score), 1), reason: 'ML neural sell signal' });
            } else {
                signals.push({ action: 'hold', strength: 0 });
            }
        }

        return signals;
    }
}

// ============================================================================
// BACKTESTER
// ============================================================================

class Backtester {
    constructor(initialCapital = 10000) {
        this.initialCapital = initialCapital;
        this.capital = initialCapital;
        this.position = 0;
        this.trades = [];
        this.equity = [];
    }

    run(data, signals) {
        for (let i = 0; i < data.length; i++) {
            const signal = signals[i];
            const price = data[i].close;

            if (signal.action === 'buy' && this.position === 0 && this.capital > 0) {
                // Buy
                const shares = Math.floor(this.capital / price);
                if (shares > 0) {
                    this.position = shares;
                    this.capital -= shares * price;
                    this.trades.push({
                        timestamp: data[i].timestamp,
                        action: 'buy',
                        price: price,
                        shares: shares,
                        reason: signal.reason
                    });
                }
            } else if (signal.action === 'sell' && this.position > 0) {
                // Sell
                this.capital += this.position * price;
                this.trades.push({
                    timestamp: data[i].timestamp,
                    action: 'sell',
                    price: price,
                    shares: this.position,
                    reason: signal.reason
                });
                this.position = 0;
            }

            // Track equity
            const currentEquity = this.capital + (this.position * price);
            this.equity.push({
                timestamp: data[i].timestamp,
                equity: currentEquity
            });
        }

        return this.calculateMetrics();
    }

    calculateMetrics() {
        const finalEquity = this.equity[this.equity.length - 1].equity;
        const totalReturn = ((finalEquity - this.initialCapital) / this.initialCapital) * 100;

        // Calculate returns for Sharpe ratio
        const returns = [];
        for (let i = 1; i < this.equity.length; i++) {
            const ret = (this.equity[i].equity - this.equity[i-1].equity) / this.equity[i-1].equity;
            returns.push(ret);
        }

        const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
        const stdDev = Math.sqrt(variance);
        const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0; // Annualized

        // Calculate max drawdown
        let maxDrawdown = 0;
        let peak = this.equity[0].equity;
        for (const point of this.equity) {
            if (point.equity > peak) peak = point.equity;
            const drawdown = ((peak - point.equity) / peak) * 100;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        }

        // Win rate
        const buyTrades = this.trades.filter(t => t.action === 'buy');
        const sellTrades = this.trades.filter(t => t.action === 'sell');
        const completedTrades = Math.min(buyTrades.length, sellTrades.length);
        let wins = 0;

        for (let i = 0; i < completedTrades; i++) {
            if (sellTrades[i].price > buyTrades[i].price) wins++;
        }

        const winRate = completedTrades > 0 ? (wins / completedTrades) * 100 : 0;

        // Sortino ratio (downside deviation)
        const downsideReturns = returns.filter(r => r < 0);
        const downsideVariance = downsideReturns.length > 0
            ? downsideReturns.reduce((sum, ret) => sum + Math.pow(ret, 2), 0) / downsideReturns.length
            : 0;
        const downsideDev = Math.sqrt(downsideVariance);
        const sortinoRatio = downsideDev > 0 ? (avgReturn / downsideDev) * Math.sqrt(252) : 0;

        return {
            initialCapital: this.initialCapital,
            finalEquity: parseFloat(finalEquity.toFixed(2)),
            totalReturn: parseFloat(totalReturn.toFixed(2)),
            totalTrades: this.trades.length,
            completedTrades: completedTrades,
            winRate: parseFloat(winRate.toFixed(2)),
            sharpeRatio: parseFloat(sharpeRatio.toFixed(4)),
            sortinoRatio: parseFloat(sortinoRatio.toFixed(4)),
            maxDrawdown: parseFloat(maxDrawdown.toFixed(2)),
            trades: this.trades,
            equityCurve: this.equity
        };
    }
}

// ============================================================================
// MAIN ACTOR
// ============================================================================

await Actor.main(async () => {
    const rawInput = await Actor.getInput();

    // Provide defaults for local testing
    const input = {
        mode: 'simulate',
        symbol: 'AAPL',
        timeframe: '1h',
        dataPoints: 1000,
        volatility: 0.05,
        trend: 'sideways',
        strategy: 'momentum',
        includeIndicators: true,
        includeOrderBook: false,
        includeNews: false,
        outputFormat: 'json',
        ...rawInput
    };

    console.log('🚀 AI Trading Simulator - Starting...');
    console.log(`Mode: ${input.mode}`);
    console.log(`Symbol: ${input.symbol}`);
    console.log(`Timeframe: ${input.timeframe}`);
    console.log(`Data Points: ${input.dataPoints}`);

    // Generate market data
    const generator = new MarketDataGenerator(input);
    const ohlcvData = generator.generateOHLCV();

    console.log(`✅ Generated ${ohlcvData.length} OHLCV candles`);

    // Calculate technical indicators
    let indicators = null;
    if (input.includeIndicators) {
        const closes = ohlcvData.map(d => d.close);
        indicators = {
            sma: TechnicalIndicators.sma(closes, 20),
            ema: TechnicalIndicators.ema(closes, 12),
            rsi: TechnicalIndicators.rsi(closes, 14),
            macd: TechnicalIndicators.macd(closes),
            bb: TechnicalIndicators.bollingerBands(closes)
        };
        console.log('✅ Calculated technical indicators (SMA, EMA, RSI, MACD, BB)');
    }

    // Add indicators to data
    const enrichedData = ohlcvData.map((candle, i) => {
        const result = { ...candle };

        if (indicators) {
            result.indicators = {
                sma_20: indicators.sma[i],
                ema_12: indicators.ema[i],
                rsi_14: indicators.rsi[i],
                macd: indicators.macd.macdLine[i],
                macd_signal: indicators.macd.signalLine[i],
                macd_histogram: indicators.macd.histogram[i],
                bb_upper: indicators.bb.upper[i],
                bb_middle: indicators.bb.middle[i],
                bb_lower: indicators.bb.lower[i]
            };
        }

        if (input.includeOrderBook) {
            result.orderBook = generator.generateOrderBook(candle.close);
        }

        if (input.includeNews && i > 0) {
            result.news = generator.generateNews(candle.close, ohlcvData[i-1].close);
        }

        return result;
    });

    // Mode-specific processing
    let results = { data: enrichedData };

    if (input.mode === 'backtest' && indicators) {
        console.log(`📊 Backtesting strategy: ${input.strategy}`);

        // Generate signals
        let signals;
        switch (input.strategy) {
            case 'mean_reversion':
                signals = TradingStrategies.meanReversion(ohlcvData, indicators);
                break;
            case 'momentum':
                signals = TradingStrategies.momentum(ohlcvData, indicators);
                break;
            case 'breakout':
                signals = TradingStrategies.breakout(ohlcvData, indicators);
                break;
            case 'ml_neural':
                signals = TradingStrategies.mlNeural(ohlcvData, indicators);
                break;
            default:
                signals = TradingStrategies.momentum(ohlcvData, indicators);
        }

        // Run backtest
        const backtester = new Backtester(10000);
        const metrics = backtester.run(ohlcvData, signals);

        console.log('✅ Backtest complete');
        console.log(`   Total Return: ${metrics.totalReturn}%`);
        console.log(`   Sharpe Ratio: ${metrics.sharpeRatio}`);
        console.log(`   Win Rate: ${metrics.winRate}%`);
        console.log(`   Max Drawdown: ${metrics.maxDrawdown}%`);

        results.backtest = metrics;
        results.signals = signals;
    }

    if (input.mode === 'train') {
        console.log('🧠 Generating ML training dataset');
        results.mlReady = true;
        results.features = ['open', 'high', 'low', 'close', 'volume'];
        if (indicators) {
            results.features.push('sma_20', 'ema_12', 'rsi_14', 'macd', 'bb_upper', 'bb_lower');
        }
    }

    if (input.mode === 'stream') {
        console.log('📡 Stream mode - generating real-time simulation');
        results.streamInfo = {
            updateInterval: generator.getIntervalMs(input.timeframe),
            isLive: true
        };
    }

    // Add metadata
    results.metadata = {
        symbol: input.symbol,
        timeframe: input.timeframe,
        dataPoints: ohlcvData.length,
        volatility: input.volatility,
        trend: input.trend,
        generatedAt: new Date().toISOString(),
        mode: input.mode,
        outputFormat: input.outputFormat
    };

    // Push to dataset
    await Actor.pushData(results);

    console.log('✅ Results saved to dataset');
    console.log(`📊 Total records: ${enrichedData.length}`);

    if (input.mode === 'backtest') {
        console.log('\n📈 Backtest Summary:');
        console.log(`   Strategy: ${input.strategy}`);
        console.log(`   Initial Capital: $${results.backtest.initialCapital}`);
        console.log(`   Final Equity: $${results.backtest.finalEquity}`);
        console.log(`   Total Return: ${results.backtest.totalReturn}%`);
        console.log(`   Total Trades: ${results.backtest.totalTrades}`);
        console.log(`   Win Rate: ${results.backtest.winRate}%`);
        console.log(`   Sharpe Ratio: ${results.backtest.sharpeRatio}`);
        console.log(`   Sortino Ratio: ${results.backtest.sortinoRatio}`);
        console.log(`   Max Drawdown: ${results.backtest.maxDrawdown}%`);
    }

    console.log('\n🎉 AI Trading Simulator - Complete!');
    console.log('Visit https://ruv.io for more neural trading tools');
});
