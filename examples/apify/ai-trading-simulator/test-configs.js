#!/usr/bin/env node

// Test multiple configurations to demonstrate functionality

import { execSync } from 'child_process';
import fs from 'fs';

const tests = [
    {
        name: 'High Volatility BTC Momentum',
        config: {
            mode: 'backtest',
            symbol: 'BTC-USD',
            timeframe: '4h',
            dataPoints: 500,
            volatility: 0.08,
            trend: 'bullish',
            strategy: 'momentum',
            initialPrice: 45000,
            seed: 12345
        }
    },
    {
        name: 'TSLA Volatile Breakout',
        config: {
            mode: 'backtest',
            symbol: 'TSLA',
            timeframe: '1h',
            dataPoints: 1000,
            volatility: 0.15,
            trend: 'volatile',
            strategy: 'momentum',
            initialPrice: 250,
            seed: 67890
        }
    }
];

console.log('🧪 Running AI Trading Simulator Tests\n');

tests.forEach((test, i) => {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Test ${i + 1}: ${test.name}`);
    console.log('='.repeat(60));

    // Create modified standalone test
    let testContent = fs.readFileSync('standalone-test.js', 'utf-8');

    // Replace config
    const configStr = JSON.stringify(test.config, null, 4).replace(/\n/g, '\n    ');
    testContent = testContent.replace(
        /const config = \{[\s\S]*?\};/,
        `const config = ${configStr};`
    );

    // Write temporary test file
    fs.writeFileSync('temp-test.js', testContent);

    // Run test
    try {
        const output = execSync('node temp-test.js', { encoding: 'utf-8' });
        console.log(output);
    } catch (err) {
        console.error('Test failed:', err.message);
    }

    // Cleanup
    fs.unlinkSync('temp-test.js');
});

console.log('\n✅ All tests complete!');
