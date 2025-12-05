#!/usr/bin/env npx ts-node
/**
 * SWE-Bench Runner for @ruvector/ruvllm
 *
 * Quick runner script with sensible defaults for benchmarking small models.
 *
 * Usage:
 *   npx ts-node benchmarks/run-swe-bench.ts
 *   npx ts-node benchmarks/run-swe-bench.ts --tasks 50 --folds 3
 *   npx ts-node benchmarks/run-swe-bench.ts --quick
 *
 * Environment Variables:
 *   OPENAI_API_KEY      - For GPT-4.1-nano/mini, GPT-5-nano
 *   ANTHROPIC_API_KEY   - For Claude 3 Haiku
 *   OPENROUTER_API_KEY  - For open-source models
 */

import {
  SWEBenchHarness,
  generateSyntheticSWEBenchTasks,
  getSmallModelRegistry,
  SmallModelConfig,
} from './swe-bench-harness';

// ============================================================================
// CLI Argument Parsing
// ============================================================================

interface CLIArgs {
  tasks: number;
  folds: number;
  quick: boolean;
  models: string[];
  seed: string;
  output: string;
}

function parseArgs(): CLIArgs {
  const args: CLIArgs = {
    tasks: 100,
    folds: 5,
    quick: false,
    models: [],
    seed: `swe-bench-${Date.now()}`,
    output: './benchmarks/results',
  };

  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];

    if (arg === '--quick') {
      args.quick = true;
      args.tasks = 30;
      args.folds = 3;
    } else if (arg === '--tasks' && process.argv[i + 1]) {
      args.tasks = parseInt(process.argv[++i], 10);
    } else if (arg === '--folds' && process.argv[i + 1]) {
      args.folds = parseInt(process.argv[++i], 10);
    } else if (arg === '--seed' && process.argv[i + 1]) {
      args.seed = process.argv[++i];
    } else if (arg === '--output' && process.argv[i + 1]) {
      args.output = process.argv[++i];
    } else if (arg === '--models' && process.argv[i + 1]) {
      args.models = process.argv[++i].split(',');
    } else if (arg === '--help') {
      printHelp();
      process.exit(0);
    }
  }

  return args;
}

function printHelp(): void {
  console.log(`
SWE-Bench Small Model Benchmark Runner
@ruvector/ruvllm - December 2025

USAGE:
  npx ts-node benchmarks/run-swe-bench.ts [OPTIONS]

OPTIONS:
  --tasks <n>      Number of synthetic tasks (default: 100)
  --folds <n>      K-fold cross-validation folds (default: 5)
  --quick          Quick mode: 30 tasks, 3 folds
  --models <list>  Comma-separated model names to test
  --seed <string>  Random seed for reproducibility
  --output <dir>   Output directory (default: ./benchmarks/results)
  --help           Show this help

EXAMPLES:
  # Full benchmark with defaults
  npx ts-node benchmarks/run-swe-bench.ts

  # Quick benchmark
  npx ts-node benchmarks/run-swe-bench.ts --quick

  # Test specific models only
  npx ts-node benchmarks/run-swe-bench.ts --models "GPT-4.1-nano,Qwen2.5-Coder-7B"

  # Custom configuration
  npx ts-node benchmarks/run-swe-bench.ts --tasks 200 --folds 10 --seed "fixed-seed"

ENVIRONMENT VARIABLES:
  OPENAI_API_KEY      GPT-4.1 and GPT-5 models
  ANTHROPIC_API_KEY   Claude 3 Haiku
  OPENROUTER_API_KEY  Open-source models via OpenRouter

ANTI-OVERFITTING MEASURES:
  - Stratified train/validation/test splits (60/20/20)
  - K-fold cross-validation with confidence intervals
  - Holdout set for final evaluation (10% of tasks)
  - Statistical significance testing between models
  - Overfit gap analysis (train - test performance)

SMALL MODEL FOCUS (December 2025):
  Models under ~24B parameters, prioritizing:
  - Qwen2.5-Coder-1.5B (1.5B) - Ultra tiny
  - Qwen2.5-Coder-7B (7B) - Best tiny coder
  - GPT-4.1-nano (~8B) - OpenAI fastest
  - GPT-4.1-mini (~20B) - OpenAI balanced
  - DeepSeek-Coder-V2-Lite (16B MoE, 2.4B active)
  - Devstral-Small (24B) - Top open model at 53.6% SWE-bench

For more details, see: https://github.com/ruvnet/ruvector
`);
}

// ============================================================================
// Focused Small Model Selection
// ============================================================================

/**
 * Get models focused on smallest/most efficient for SWE-bench
 */
function getTinyModelRegistry(): SmallModelConfig[] {
  const openaiKey = process.env.OPENAI_API_KEY;
  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  const openrouterKey = process.env.OPENROUTER_API_KEY;

  // Focus on truly small models (under 10B parameters or highly efficient)
  return [
    {
      name: 'Qwen2.5-Coder-1.5B',
      provider: 'openrouter',
      modelId: 'qwen/qwen-2.5-coder-1.5b-instruct',
      apiKey: openrouterKey,
      maxTokens: 32768,
      parameterCount: '1.5B',
      costPer1kTokens: { input: 0.00002, output: 0.00002 },
      description: 'Smallest viable code model',
    },
    {
      name: 'DeepSeek-Coder-1.3B',
      provider: 'openrouter',
      modelId: 'deepseek/deepseek-coder-1.3b-instruct',
      apiKey: openrouterKey,
      maxTokens: 16384,
      parameterCount: '1.3B',
      costPer1kTokens: { input: 0.00001, output: 0.00001 },
      description: 'DeepSeek tiny coder',
    },
    {
      name: 'StarCoder2-3B',
      provider: 'openrouter',
      modelId: 'bigcode/starcoder2-3b',
      apiKey: openrouterKey,
      maxTokens: 16384,
      parameterCount: '3B',
      costPer1kTokens: { input: 0.00003, output: 0.00003 },
      description: 'BigCode 3B model',
    },
    {
      name: 'Qwen2.5-Coder-7B',
      provider: 'openrouter',
      modelId: 'qwen/qwen-2.5-coder-7b-instruct',
      apiKey: openrouterKey,
      maxTokens: 32768,
      parameterCount: '7B',
      costPer1kTokens: { input: 0.00007, output: 0.00007 },
      description: 'Best 7B code model',
    },
    {
      name: 'CodeLlama-7B',
      provider: 'openrouter',
      modelId: 'meta-llama/codellama-7b-instruct',
      apiKey: openrouterKey,
      maxTokens: 16384,
      parameterCount: '7B',
      costPer1kTokens: { input: 0.00007, output: 0.00007 },
      description: 'Meta CodeLlama 7B',
    },
    {
      name: 'GPT-4.1-nano',
      provider: 'openai',
      modelId: 'gpt-4.1-nano-2025-04-14',
      apiKey: openaiKey,
      maxTokens: 16384,
      parameterCount: '~8B',
      costPer1kTokens: { input: 0.0001, output: 0.0004 },
      description: 'OpenAI fastest model',
    },
    {
      name: 'GPT-5-nano',
      provider: 'openai',
      modelId: 'gpt-5-nano',
      apiKey: openaiKey,
      maxTokens: 32768,
      parameterCount: '~10B',
      costPer1kTokens: { input: 0.00015, output: 0.0006 },
      description: 'Latest OpenAI nano',
    },
  ];
}

// ============================================================================
// Main Runner
// ============================================================================

async function main() {
  const args = parseArgs();

  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║     🔬 SWE-Bench Small Model Benchmark                         ║');
  console.log('║     @ruvector/ruvllm - December 2025                           ║');
  console.log('╚════════════════════════════════════════════════════════════════╝\n');

  console.log('Configuration:');
  console.log(`  Tasks: ${args.tasks}`);
  console.log(`  K-Folds: ${args.folds}`);
  console.log(`  Seed: ${args.seed}`);
  console.log(`  Output: ${args.output}`);
  console.log(`  Mode: ${args.quick ? 'Quick' : 'Full'}\n`);

  // Check for API keys
  const hasOpenAI = !!process.env.OPENAI_API_KEY;
  const hasAnthropic = !!process.env.ANTHROPIC_API_KEY;
  const hasOpenRouter = !!process.env.OPENROUTER_API_KEY;

  console.log('API Keys Detected:');
  console.log(`  OPENAI_API_KEY:     ${hasOpenAI ? '✓' : '✗ (will use mock)'}`);
  console.log(`  ANTHROPIC_API_KEY:  ${hasAnthropic ? '✓' : '✗ (will use mock)'}`);
  console.log(`  OPENROUTER_API_KEY: ${hasOpenRouter ? '✓' : '✗ (will use mock)'}\n`);

  // Create harness
  const harness = new SWEBenchHarness({
    tasks: generateSyntheticSWEBenchTasks(args.tasks),
    kFolds: args.folds,
    seed: args.seed,
    outputDir: args.output,
  });

  // Select model registry
  const registry = args.quick ? getTinyModelRegistry() : getSmallModelRegistry();

  // Filter models if specified
  const modelsToTest = args.models.length > 0
    ? registry.filter(m => args.models.includes(m.name))
    : registry;

  if (modelsToTest.length === 0) {
    console.error('❌ No matching models found. Available models:');
    registry.forEach(m => console.log(`  - ${m.name}`));
    process.exit(1);
  }

  // Register models
  console.log(`\n📋 Registering ${modelsToTest.length} models:\n`);
  for (const config of modelsToTest) {
    harness.addModel(config);
  }

  // Run benchmark
  console.log('\n⏳ Starting benchmark... (this may take a while)\n');
  const startTime = Date.now();

  try {
    const report = await harness.runBenchmark();

    const duration = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
    console.log(`\n✅ Benchmark completed in ${duration} minutes!`);

    // Print summary
    console.log('\n' + '═'.repeat(80));
    console.log('\n📊 DECEMBER 2025 SMALL MODEL RANKINGS\n');

    console.log('🏆 Top 3 by Resolve Rate (Cross-Validated):');
    report.rankings.byResolveRate.slice(0, 3).forEach((name, i) => {
      const m = report.modelMetrics.find(x => x.modelName === name)!;
      console.log(`   ${i + 1}. ${name.padEnd(25)} ${(m.cvMean * 100).toFixed(1)}% ± ${(m.cvStd * 100).toFixed(1)}%`);
    });

    console.log('\n💰 Top 3 by Cost Efficiency:');
    report.rankings.byCostEfficiency.slice(0, 3).forEach((name, i) => {
      const m = report.modelMetrics.find(x => x.modelName === name)!;
      console.log(`   ${i + 1}. ${name.padEnd(25)} $${m.costPerResolvedTask.toFixed(4)}/resolved`);
    });

    console.log('\n⚡ Top 3 by Latency:');
    report.rankings.byLatency.slice(0, 3).forEach((name, i) => {
      const m = report.modelMetrics.find(x => x.modelName === name)!;
      console.log(`   ${i + 1}. ${name.padEnd(25)} ${m.avgLatencyMs.toFixed(0)}ms avg`);
    });

    console.log('\n🎯 RECOMMENDED BEST TINY MODEL:');
    console.log(`   ${report.bestSmallModel.name}`);
    console.log(`   CV Rate: ${(report.bestSmallModel.metrics.cvMean * 100).toFixed(1)}%`);
    console.log(`   95% CI: [${(report.bestSmallModel.metrics.cvConfidenceInterval[0] * 100).toFixed(1)}%, ${(report.bestSmallModel.metrics.cvConfidenceInterval[1] * 100).toFixed(1)}%]`);
    console.log(`   Overfit Gap: ${(report.bestSmallModel.metrics.overfitGap * 100).toFixed(1)}%`);

    console.log('\n' + '═'.repeat(80));

    // Return for programmatic use
    return report;

  } catch (error) {
    console.error('\n❌ Benchmark failed:', error);
    process.exit(1);
  }
}

// Run
main().catch(console.error);
