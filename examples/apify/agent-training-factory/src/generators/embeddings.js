/**
 * Embeddings Generator
 * Generates vector embeddings for RAG systems
 */

/**
 * Generate embedding datasets
 */
export async function generateEmbeddings(config) {
    const {
        domain,
        count,
        complexity,
        includeMetadata,
        groundingData,
        embeddingModel,
        openaiApiKey,
    } = config;

    const embeddings = [];

    for (let i = 0; i < count; i++) {
        const embedding = await generateEmbedding({
            complexity,
            groundingData,
            index: i,
            domain,
            embeddingModel,
            openaiApiKey,
        });

        if (includeMetadata) {
            embedding.metadata = {
                id: `emb_${i.toString().padStart(6, '0')}`,
                domain,
                complexity,
                model: embeddingModel,
                generated_at: new Date().toISOString(),
                grounded: !!groundingData,
            };
        }

        embeddings.push(embedding);
    }

    return embeddings;
}

/**
 * Generate a single embedding entry
 */
async function generateEmbedding({
    complexity,
    groundingData,
    index,
    domain,
    embeddingModel,
    openaiApiKey,
}) {
    // Generate text content
    const text = generateTextContent(domain, complexity, index, groundingData);

    // Generate vector (mock or real)
    let vector;
    if (openaiApiKey) {
        vector = await generateRealEmbedding(text, embeddingModel, openaiApiKey);
    } else {
        vector = generateMockEmbedding(embeddingModel);
    }

    return {
        type: 'embedding',
        text,
        vector,
        model: embeddingModel,
        dimension: vector.length,
        complexity,
    };
}

/**
 * Generate text content for embedding
 */
function generateTextContent(domain, complexity, index, groundingData) {
    const templates = {
        customer_support: [
            "How can I reset my password if I don't have access to my email?",
            "What is your refund policy for items purchased during a sale?",
            "I need help troubleshooting connection issues with your service.",
        ],
        coding: [
            "Implementing authentication with JWT tokens in Node.js Express applications",
            "Best practices for handling async operations in JavaScript with Promises",
            "Understanding the differences between SQL and NoSQL database design patterns",
        ],
        research: [
            "The impact of climate change on Arctic ecosystems and biodiversity",
            "Machine learning applications in medical diagnosis and treatment planning",
            "Quantum computing principles and their potential applications in cryptography",
        ],
    };

    const domainTemplates = templates[domain] || templates.customer_support;
    let text = domainTemplates[index % domainTemplates.length];

    // Enhance based on complexity
    if (complexity === 'complex' || complexity === 'expert') {
        text += " This requires consideration of multiple factors including edge cases, performance implications, and scalability requirements.";
    }

    // Use grounding data if available
    if (groundingData && groundingData.length > 0) {
        const groundingSample = groundingData[index % groundingData.length];
        if (groundingSample.text || groundingSample.content) {
            text = groundingSample.text || groundingSample.content;
        }
    }

    return text;
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
        console.warn('Failed to generate real embedding, falling back to mock:', error.message);
        return generateMockEmbedding(model);
    }
}

/**
 * Generate mock embedding vector
 */
function generateMockEmbedding(model) {
    // Determine dimension based on model
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
        const value = (Math.random() - 0.5) * 2; // Range [-1, 1]
        vector.push(value);
        sumSquares += value * value;
    }

    // Normalize to unit vector
    const magnitude = Math.sqrt(sumSquares);
    return vector.map(v => v / magnitude);
}
