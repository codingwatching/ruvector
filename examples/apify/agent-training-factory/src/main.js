import { Actor } from 'apify';
import { generateConversations } from './generators/conversations.js';
import { generateQAPairs } from './generators/qa_pairs.js';
import { generateToolCalls } from './generators/tool_calls.js';
import { generateReasoningChains } from './generators/reasoning_chains.js';
import { generateMemoryPatterns } from './generators/memory_patterns.js';
import { generateEmbeddings } from './generators/embeddings.js';
import { fetchGroundingData } from './utils/grounding.js';
import { formatOutput } from './utils/formatters.js';

/**
 * Agent Training Data Factory - Main Entry Point
 * Generates training datasets for AI agents with various patterns
 */
await Actor.main(async () => {
    // Get input configuration
    const input = await Actor.getInput();

    if (!input) {
        throw new Error('Missing input configuration!');
    }

    const {
        datasetType,
        domain,
        count = 100,
        complexity = 'moderate',
        includeMetadata = true,
        groundingActorId,
        groundingDatasetId,
        outputFormat = 'jsonl',
        generateEmbeddings: embeddings = false,
        embeddingModel = 'text-embedding-3-small',
        openaiApiKey,
        customPrompts,
        seed,
    } = input;

    console.log(`Starting Agent Training Data Factory...`);
    console.log(`Dataset Type: ${datasetType}`);
    console.log(`Domain: ${domain}`);
    console.log(`Count: ${count}`);
    console.log(`Complexity: ${complexity}`);

    // Set random seed if provided
    if (seed !== null && seed !== undefined) {
        Math.seedrandom = (s) => {
            let seed = s;
            return function() {
                seed = (seed * 9301 + 49297) % 233280;
                return seed / 233280;
            };
        };
        Math.random = Math.seedrandom(seed);
    }

    // Fetch grounding data if actor ID or dataset ID provided
    let groundingData = null;
    if (groundingActorId || groundingDatasetId) {
        console.log('Fetching grounding data...');
        groundingData = await fetchGroundingData({
            actorId: groundingActorId,
            datasetId: groundingDatasetId,
        });
        console.log(`Fetched ${groundingData.length} grounding examples`);
    }

    // Generate data based on type
    let generatedData = [];
    const generatorConfig = {
        domain,
        count,
        complexity,
        includeMetadata,
        groundingData,
        customPrompts,
    };

    console.log(`Generating ${count} ${datasetType} examples...`);

    switch (datasetType) {
        case 'conversations':
            generatedData = await generateConversations(generatorConfig);
            break;
        case 'qa_pairs':
            generatedData = await generateQAPairs(generatorConfig);
            break;
        case 'tool_calls':
            generatedData = await generateToolCalls(generatorConfig);
            break;
        case 'reasoning_chains':
            generatedData = await generateReasoningChains(generatorConfig);
            break;
        case 'memory_patterns':
            generatedData = await generateMemoryPatterns(generatorConfig);
            break;
        case 'embeddings':
            generatedData = await generateEmbeddings({
                ...generatorConfig,
                embeddingModel,
                openaiApiKey,
            });
            break;
        default:
            throw new Error(`Unknown dataset type: ${datasetType}`);
    }

    console.log(`Generated ${generatedData.length} examples`);

    // Generate embeddings if requested (for non-embedding dataset types)
    if (embeddings && datasetType !== 'embeddings') {
        console.log('Generating embeddings...');
        const { generateEmbeddingsForData } = await import('./utils/embeddings.js');
        generatedData = await generateEmbeddingsForData(generatedData, {
            model: embeddingModel,
            apiKey: openaiApiKey,
        });
    }

    // Format output based on requested format
    console.log(`Formatting output as ${outputFormat}...`);
    const formattedData = await formatOutput(generatedData, outputFormat);

    // Push to dataset
    console.log('Pushing to dataset...');
    await Actor.pushData(formattedData);

    // Save statistics
    const stats = {
        datasetType,
        domain,
        totalExamples: generatedData.length,
        complexity,
        hasEmbeddings: embeddings || datasetType === 'embeddings',
        outputFormat,
        timestamp: new Date().toISOString(),
    };

    console.log('Generation complete!');
    console.log('Statistics:', JSON.stringify(stats, null, 2));

    // Store stats as key-value
    await Actor.setValue('OUTPUT_STATS', stats);

    console.log('Actor finished successfully!');
});
