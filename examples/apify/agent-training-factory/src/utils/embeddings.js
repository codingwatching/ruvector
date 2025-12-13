/**
 * Embeddings Utilities
 * Generate embeddings for existing data
 */

/**
 * Generate embeddings for data array
 */
export async function generateEmbeddingsForData(data, options = {}) {
    const { model = 'text-embedding-3-small', apiKey } = options;

    console.log(`Generating embeddings for ${data.length} items...`);

    const enhanced = [];

    for (let i = 0; i < data.length; i++) {
        if (i % 10 === 0) {
            console.log(`Progress: ${i}/${data.length}`);
        }

        const item = data[i];
        const text = extractTextFromItem(item);

        let vector;
        if (apiKey) {
            vector = await generateRealEmbedding(text, model, apiKey);
        } else {
            vector = generateMockEmbedding(model);
        }

        enhanced.push({
            ...item,
            embedding: {
                model,
                vector,
                dimension: vector.length,
            },
        });
    }

    console.log('Embeddings generation complete');
    return enhanced;
}

/**
 * Extract text content from data item
 */
function extractTextFromItem(item) {
    // Try common text fields
    if (item.text) return item.text;
    if (item.content) return item.content;
    if (item.question && item.answer) return `${item.question}\n${item.answer}`;

    // For conversations, concatenate messages
    if (item.messages && Array.isArray(item.messages)) {
        return item.messages.map(m => m.content).join('\n');
    }

    // For tool calls, use function name and arguments
    if (item.calls && Array.isArray(item.calls)) {
        return item.calls.map(c => `${c.function?.name || ''}: ${c.function?.arguments || ''}`).join('\n');
    }

    // For reasoning chains, concatenate steps
    if (item.steps && Array.isArray(item.steps)) {
        return item.steps.map(s => s.content).join('\n');
    }

    // Default: stringify the object
    return JSON.stringify(item);
}

/**
 * Generate real embedding using OpenAI API
 */
async function generateRealEmbedding(text, model, apiKey) {
    try {
        const { default: OpenAI } = await import('openai');
        const openai = new OpenAI({ apiKey });

        const response = await openai.embeddings.create({
            model,
            input: text,
        });

        return response.data[0].embedding;
    } catch (error) {
        console.warn('Failed to generate real embedding:', error.message);
        return generateMockEmbedding(model);
    }
}

/**
 * Generate mock embedding vector
 */
function generateMockEmbedding(model) {
    const dimensions = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    };

    const dimension = dimensions[model] || 1536;

    // Generate normalized random vector
    const vector = [];
    let sumSquares = 0;

    for (let i = 0; i < dimension; i++) {
        const value = (Math.random() - 0.5) * 2;
        vector.push(value);
        sumSquares += value * value;
    }

    const magnitude = Math.sqrt(sumSquares);
    return vector.map(v => v / magnitude);
}

/**
 * Batch embeddings generation with rate limiting
 */
export async function generateEmbeddingsBatch(texts, options = {}) {
    const {
        model = 'text-embedding-3-small',
        apiKey,
        batchSize = 10,
        delayMs = 1000,
    } = options;

    const results = [];

    for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);
        console.log(`Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(texts.length / batchSize)}`);

        const batchResults = await Promise.all(
            batch.map(text =>
                apiKey
                    ? generateRealEmbedding(text, model, apiKey)
                    : Promise.resolve(generateMockEmbedding(model))
            )
        );

        results.push(...batchResults);

        // Rate limiting delay
        if (i + batchSize < texts.length) {
            await new Promise(resolve => setTimeout(resolve, delayMs));
        }
    }

    return results;
}
