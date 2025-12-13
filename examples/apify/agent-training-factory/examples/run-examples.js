/**
 * Example configurations for running the Agent Training Factory
 * These demonstrate various use cases and dataset types
 */

// Example 1: Basic Customer Support Conversations
export const customerSupportExample = {
  datasetType: 'conversations',
  domain: 'customer_support',
  count: 100,
  complexity: 'moderate',
  includeMetadata: true,
  outputFormat: 'jsonl',
};

// Example 2: Coding Q&A with Embeddings
export const codingQAExample = {
  datasetType: 'qa_pairs',
  domain: 'coding',
  count: 500,
  complexity: 'complex',
  generateEmbeddings: true,
  embeddingModel: 'text-embedding-3-small',
  outputFormat: 'huggingface',
};

// Example 3: Research Conversations Grounded in Real Data
export const groundedResearchExample = {
  datasetType: 'conversations',
  domain: 'research',
  count: 200,
  complexity: 'expert',
  groundingActorId: 'apify/web-scraper',
  includeMetadata: true,
  outputFormat: 'jsonl',
};

// Example 4: Tool Call Sequences for Agent Training
export const toolCallsExample = {
  datasetType: 'tool_calls',
  domain: 'coding',
  count: 300,
  complexity: 'expert',
  includeMetadata: true,
  outputFormat: 'jsonl',
};

// Example 5: Reasoning Chains with Chain-of-Thought
export const reasoningChainsExample = {
  datasetType: 'reasoning_chains',
  domain: 'research',
  count: 150,
  complexity: 'complex',
  includeMetadata: true,
  outputFormat: 'jsonl',
};

// Example 6: Memory Patterns for Agent State Management
export const memoryPatternsExample = {
  datasetType: 'memory_patterns',
  domain: 'customer_support',
  count: 100,
  complexity: 'moderate',
  includeMetadata: true,
  outputFormat: 'jsonl',
};

// Example 7: High-Quality Embeddings with OpenAI
export const embeddingsExample = {
  datasetType: 'embeddings',
  domain: 'coding',
  count: 1000,
  complexity: 'complex',
  generateEmbeddings: true,
  embeddingModel: 'text-embedding-3-large',
  openaiApiKey: process.env.OPENAI_API_KEY,
  outputFormat: 'jsonl',
};

// Example 8: Reproducible Dataset with Seed
export const reproducibleExample = {
  datasetType: 'conversations',
  domain: 'customer_support',
  count: 100,
  complexity: 'moderate',
  seed: 42,
  includeMetadata: true,
  outputFormat: 'jsonl',
};

// Example 9: Custom Prompts for Specialized Generation
export const customPromptsExample = {
  datasetType: 'conversations',
  domain: 'healthcare',
  count: 200,
  complexity: 'expert',
  customPrompts: [
    {
      role: 'system',
      content: 'Generate realistic patient-doctor conversations focusing on mental health support',
    },
  ],
  includeMetadata: true,
  outputFormat: 'jsonl',
};

// Example 10: Large-Scale Dataset for Fine-Tuning
export const fineTuningExample = {
  datasetType: 'qa_pairs',
  domain: 'coding',
  count: 10000,
  complexity: 'complex',
  includeMetadata: true,
  outputFormat: 'huggingface',
};

/**
 * Run an example configuration
 * Usage: node examples/run-examples.js [exampleName]
 */
if (import.meta.url === `file://${process.argv[1]}`) {
  const exampleName = process.argv[2] || 'customerSupportExample';

  const examples = {
    customerSupportExample,
    codingQAExample,
    groundedResearchExample,
    toolCallsExample,
    reasoningChainsExample,
    memoryPatternsExample,
    embeddingsExample,
    reproducibleExample,
    customPromptsExample,
    fineTuningExample,
  };

  const config = examples[exampleName];

  if (!config) {
    console.error(`Unknown example: ${exampleName}`);
    console.log('Available examples:', Object.keys(examples).join(', '));
    process.exit(1);
  }

  console.log('Running example:', exampleName);
  console.log('Configuration:', JSON.stringify(config, null, 2));
  console.log('\nTo run this in Apify:');
  console.log('1. Copy the configuration above');
  console.log('2. Paste into the Input tab');
  console.log('3. Click "Start"');
}
