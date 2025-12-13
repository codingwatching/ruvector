import { Actor } from 'apify';
import { CheerioCrawler } from 'crawlee';
import * as cheerio from 'cheerio';
import crypto from 'crypto';

/**
 * RAG Knowledge Base Builder
 * Crawls content, chunks intelligently, generates embeddings, and exports to vector databases
 */

class ChunkingEngine {
    constructor(strategy, size, overlap) {
        this.strategy = strategy;
        this.size = size;
        this.overlap = overlap;
    }

    /**
     * Approximate token count (rough estimation: 1 token ≈ 4 characters)
     */
    countTokens(text) {
        return Math.ceil(text.length / 4);
    }

    /**
     * Split text into chunks based on strategy
     */
    chunk(text, metadata = {}) {
        switch (this.strategy) {
            case 'fixed_size':
                return this.fixedSizeChunk(text, metadata);
            case 'semantic':
                return this.semanticChunk(text, metadata);
            case 'paragraph':
                return this.paragraphChunk(text, metadata);
            case 'sentence':
                return this.sentenceChunk(text, metadata);
            default:
                return this.fixedSizeChunk(text, metadata);
        }
    }

    /**
     * Fixed-size chunking with overlap
     */
    fixedSizeChunk(text, metadata) {
        const chunks = [];
        const words = text.split(/\s+/);
        const wordsPerChunk = Math.floor(this.size * 4 / 5); // Rough estimation
        const overlapWords = Math.floor(this.overlap * 4 / 5);

        for (let i = 0; i < words.length; i += (wordsPerChunk - overlapWords)) {
            const chunk = words.slice(i, i + wordsPerChunk).join(' ');
            if (chunk.trim().length > 0) {
                chunks.push({
                    text: chunk,
                    tokenCount: this.countTokens(chunk),
                    metadata: { ...metadata, chunkIndex: chunks.length }
                });
            }
        }

        return chunks;
    }

    /**
     * Semantic chunking by topic/section
     */
    semanticChunk(text, metadata) {
        const chunks = [];
        // Split by double newlines (paragraphs) and headings
        const sections = text.split(/\n\s*\n|\n#{1,6}\s+/);

        let currentChunk = '';
        let currentTokens = 0;

        for (const section of sections) {
            const sectionTokens = this.countTokens(section);

            if (currentTokens + sectionTokens > this.size && currentChunk) {
                chunks.push({
                    text: currentChunk.trim(),
                    tokenCount: currentTokens,
                    metadata: { ...metadata, chunkIndex: chunks.length }
                });
                currentChunk = section;
                currentTokens = sectionTokens;
            } else {
                currentChunk += (currentChunk ? '\n\n' : '') + section;
                currentTokens += sectionTokens;
            }
        }

        if (currentChunk.trim()) {
            chunks.push({
                text: currentChunk.trim(),
                tokenCount: currentTokens,
                metadata: { ...metadata, chunkIndex: chunks.length }
            });
        }

        return chunks;
    }

    /**
     * Paragraph-based chunking
     */
    paragraphChunk(text, metadata) {
        const chunks = [];
        const paragraphs = text.split(/\n\s*\n/);

        for (const para of paragraphs) {
            const trimmed = para.trim();
            if (trimmed.length > 0) {
                const tokens = this.countTokens(trimmed);

                // Split large paragraphs
                if (tokens > this.size) {
                    const subChunks = this.fixedSizeChunk(trimmed, metadata);
                    chunks.push(...subChunks);
                } else {
                    chunks.push({
                        text: trimmed,
                        tokenCount: tokens,
                        metadata: { ...metadata, chunkIndex: chunks.length }
                    });
                }
            }
        }

        return chunks;
    }

    /**
     * Sentence-based chunking
     */
    sentenceChunk(text, metadata) {
        const chunks = [];
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];

        let currentChunk = '';
        let currentTokens = 0;

        for (const sentence of sentences) {
            const sentenceTokens = this.countTokens(sentence);

            if (currentTokens + sentenceTokens > this.size && currentChunk) {
                chunks.push({
                    text: currentChunk.trim(),
                    tokenCount: currentTokens,
                    metadata: { ...metadata, chunkIndex: chunks.length }
                });
                currentChunk = sentence;
                currentTokens = sentenceTokens;
            } else {
                currentChunk += (currentChunk ? ' ' : '') + sentence;
                currentTokens += sentenceTokens;
            }
        }

        if (currentChunk.trim()) {
            chunks.push({
                text: currentChunk.trim(),
                tokenCount: currentTokens,
                metadata: { ...metadata, chunkIndex: chunks.length }
            });
        }

        return chunks;
    }
}

class EmbeddingEngine {
    constructor(model, apiKey) {
        this.model = model;
        this.apiKey = apiKey;
    }

    /**
     * Generate embeddings for text
     * In production, this would call OpenAI/Cohere APIs
     */
    async generateEmbedding(text) {
        if (this.model.startsWith('text-embedding-3')) {
            return this.openaiEmbedding(text);
        } else if (this.model.startsWith('cohere')) {
            return this.cohereEmbedding(text);
        } else {
            return this.simulatedEmbedding(text);
        }
    }

    async openaiEmbedding(text) {
        // Production implementation would use OpenAI API
        // For now, simulated
        if (!this.apiKey) {
            console.log('⚠️  No OpenAI API key provided, using simulated embeddings');
            return this.simulatedEmbedding(text);
        }

        // TODO: Implement actual OpenAI API call
        // const response = await fetch('https://api.openai.com/v1/embeddings', {
        //     method: 'POST',
        //     headers: {
        //         'Authorization': `Bearer ${this.apiKey}`,
        //         'Content-Type': 'application/json'
        //     },
        //     body: JSON.stringify({
        //         model: this.model,
        //         input: text
        //     })
        // });

        return this.simulatedEmbedding(text);
    }

    async cohereEmbedding(text) {
        // Production implementation would use Cohere API
        if (!this.apiKey) {
            console.log('⚠️  No Cohere API key provided, using simulated embeddings');
            return this.simulatedEmbedding(text);
        }

        // TODO: Implement actual Cohere API call
        return this.simulatedEmbedding(text);
    }

    /**
     * Simulated embedding for testing
     * Creates a deterministic vector based on text content
     */
    simulatedEmbedding(text) {
        const dimension = this.model === 'text-embedding-3-large' ? 3072 : 1536;
        const hash = crypto.createHash('sha256').update(text).digest();

        const embedding = [];
        for (let i = 0; i < dimension; i++) {
            const value = (hash[i % hash.length] / 255) * 2 - 1; // Normalize to [-1, 1]
            embedding.push(value);
        }

        return embedding;
    }
}

class DeduplicationEngine {
    constructor(threshold = 0.95) {
        this.threshold = threshold;
        this.hashes = new Set();
    }

    /**
     * Generate similarity hash for text
     */
    generateHash(text) {
        // Use first 100 chars for quick similarity check
        const normalized = text.toLowerCase().replace(/\s+/g, ' ').trim();
        const sample = normalized.substring(0, 100);
        return crypto.createHash('md5').update(sample).digest('hex');
    }

    /**
     * Check if chunk is duplicate
     */
    isDuplicate(chunk) {
        const hash = this.generateHash(chunk.text);

        if (this.hashes.has(hash)) {
            return true;
        }

        this.hashes.add(hash);
        return false;
    }

    /**
     * Calculate Jaccard similarity between two texts
     */
    jaccardSimilarity(text1, text2) {
        const words1 = new Set(text1.toLowerCase().split(/\s+/));
        const words2 = new Set(text2.toLowerCase().split(/\s+/));

        const intersection = new Set([...words1].filter(x => words2.has(x)));
        const union = new Set([...words1, ...words2]);

        return intersection.size / union.size;
    }
}

class OutputFormatter {
    constructor(format) {
        this.format = format;
    }

    /**
     * Format chunks for target vector database
     */
    formatChunk(chunk, embedding) {
        switch (this.format) {
            case 'pinecone':
                return this.pineconeFormat(chunk, embedding);
            case 'weaviate':
                return this.weaviateFormat(chunk, embedding);
            case 'qdrant':
                return this.qdrantFormat(chunk, embedding);
            case 'chroma':
                return this.chromaFormat(chunk, embedding);
            case 'agentdb':
                return this.agentdbFormat(chunk, embedding);
            default:
                return this.jsonlFormat(chunk, embedding);
        }
    }

    pineconeFormat(chunk, embedding) {
        return {
            id: chunk.id,
            values: embedding,
            metadata: {
                text: chunk.text,
                ...chunk.metadata,
                tokenCount: chunk.tokenCount
            }
        };
    }

    weaviateFormat(chunk, embedding) {
        return {
            class: 'KnowledgeChunk',
            id: chunk.id,
            properties: {
                text: chunk.text,
                tokenCount: chunk.tokenCount,
                ...chunk.metadata
            },
            vector: embedding
        };
    }

    qdrantFormat(chunk, embedding) {
        return {
            id: chunk.id,
            vector: embedding,
            payload: {
                text: chunk.text,
                tokenCount: chunk.tokenCount,
                ...chunk.metadata
            }
        };
    }

    chromaFormat(chunk, embedding) {
        return {
            ids: [chunk.id],
            embeddings: [embedding],
            metadatas: [{
                text: chunk.text,
                tokenCount: chunk.tokenCount,
                ...chunk.metadata
            }],
            documents: [chunk.text]
        };
    }

    agentdbFormat(chunk, embedding) {
        return {
            id: chunk.id,
            vector: embedding,
            text: chunk.text,
            metadata: {
                tokenCount: chunk.tokenCount,
                ...chunk.metadata,
                indexed_at: new Date().toISOString(),
                source: 'apify-rag-builder'
            }
        };
    }

    jsonlFormat(chunk, embedding) {
        return {
            id: chunk.id,
            text: chunk.text,
            embedding: embedding,
            tokenCount: chunk.tokenCount,
            metadata: chunk.metadata
        };
    }
}

/**
 * Main actor entry point
 */
await Actor.main(async () => {
    console.log('🚀 RAG Knowledge Base Builder starting...');

    const input = await Actor.getInput();
    const {
        sourceType = 'urls',
        urls = [],
        sitemapUrl,
        crawlDepth = 1,
        maxPages = 100,
        chunkStrategy = 'fixed_size',
        chunkSize = 512,
        chunkOverlap = 128,
        embeddingModel = 'simulated',
        openaiApiKey,
        cohereApiKey,
        outputFormat = 'jsonl',
        includeMetadata = true,
        deduplication = true,
        similarityThreshold = 0.95,
        excludeSelectors = ['nav', 'header', 'footer', '.sidebar', '.advertisement'],
        includeSelectors = [],
        metadata: customMetadata = {}
    } = input;

    // Initialize engines
    const chunkingEngine = new ChunkingEngine(chunkStrategy, chunkSize, chunkOverlap);
    const embeddingEngine = new EmbeddingEngine(embeddingModel, openaiApiKey || cohereApiKey);
    const deduplicationEngine = deduplication ? new DeduplicationEngine(similarityThreshold) : null;
    const outputFormatter = new OutputFormatter(outputFormat);

    let totalChunks = 0;
    let totalPages = 0;
    let duplicatesRemoved = 0;

    // Configure crawler
    const crawler = new CheerioCrawler({
        maxRequestsPerCrawl: maxPages > 0 ? maxPages : undefined,
        maxConcurrency: 5,
        async requestHandler({ request, $, enqueueLinks }) {
            totalPages++;
            console.log(`📄 Processing [${totalPages}]: ${request.url}`);

            // Remove excluded elements
            excludeSelectors.forEach(selector => $(selector).remove());

            // Extract content
            let content;
            if (includeSelectors.length > 0) {
                content = includeSelectors.map(selector => $(selector).text()).join('\n\n');
            } else {
                content = $('body').text();
            }

            // Clean content
            const cleanedContent = content
                .replace(/\s+/g, ' ')
                .replace(/\n\s*\n\s*\n/g, '\n\n')
                .trim();

            if (!cleanedContent) {
                console.log(`⚠️  No content extracted from ${request.url}`);
                return;
            }

            // Extract metadata
            const title = $('title').text() || $('h1').first().text() || 'Untitled';
            const description = $('meta[name="description"]').attr('content') || '';

            const pageMetadata = {
                source: request.url,
                title: title.trim(),
                description: description.trim(),
                crawledAt: new Date().toISOString(),
                ...customMetadata
            };

            // Chunk content
            const chunks = chunkingEngine.chunk(cleanedContent, includeMetadata ? pageMetadata : {});
            console.log(`  ✂️  Created ${chunks.length} chunks from ${request.url}`);

            // Process each chunk
            for (const chunk of chunks) {
                // Check for duplicates
                if (deduplicationEngine && deduplicationEngine.isDuplicate(chunk)) {
                    duplicatesRemoved++;
                    continue;
                }

                // Generate unique ID
                chunk.id = crypto.randomUUID();

                // Generate embedding
                const embedding = await embeddingEngine.generateEmbedding(chunk.text);

                // Format for output
                const formattedChunk = outputFormatter.formatChunk(chunk, embedding);

                // Save to dataset
                await Actor.pushData(formattedChunk);
                totalChunks++;
            }

            // Enqueue links if depth allows
            if (request.userData.depth < crawlDepth) {
                await enqueueLinks({
                    userData: { depth: (request.userData.depth || 0) + 1 },
                    strategy: 'same-domain'
                });
            }
        },
        failedRequestHandler({ request, error }) {
            console.error(`❌ Request failed: ${request.url}`, error.message);
        }
    });

    // Add start URLs
    const startUrls = urls.map(url => ({
        url: typeof url === 'string' ? url : url.url,
        userData: { depth: 0 }
    }));

    if (startUrls.length === 0) {
        console.error('❌ No URLs provided');
        return;
    }

    console.log(`📋 Processing ${startUrls.length} start URLs with depth ${crawlDepth}`);
    console.log(`📐 Chunking strategy: ${chunkStrategy} (${chunkSize} tokens, ${chunkOverlap} overlap)`);
    console.log(`🧠 Embedding model: ${embeddingModel}`);
    console.log(`💾 Output format: ${outputFormat}`);

    // Run crawler
    await crawler.run(startUrls);

    // Final statistics
    console.log('\n✅ RAG Knowledge Base Builder completed!');
    console.log(`📊 Statistics:`);
    console.log(`   - Pages processed: ${totalPages}`);
    console.log(`   - Chunks created: ${totalChunks}`);
    if (deduplication) {
        console.log(`   - Duplicates removed: ${duplicatesRemoved}`);
    }
    console.log(`   - Average chunks per page: ${(totalChunks / totalPages).toFixed(2)}`);

    // Set output
    await Actor.setValue('OUTPUT', {
        status: 'success',
        statistics: {
            pagesProcessed: totalPages,
            chunksCreated: totalChunks,
            duplicatesRemoved: duplicatesRemoved,
            averageChunksPerPage: totalChunks / totalPages,
            chunkingStrategy: chunkStrategy,
            embeddingModel: embeddingModel,
            outputFormat: outputFormat
        },
        configuration: {
            chunkSize,
            chunkOverlap,
            crawlDepth,
            deduplication
        }
    });

    console.log('\n🎉 Knowledge base ready for RAG! Check the dataset for indexed chunks.');
});
