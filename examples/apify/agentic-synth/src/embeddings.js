/**
 * ONNX-based Embeddings Module
 * Uses @xenova/transformers (Hugging Face Transformers.js) for ONNX Runtime embeddings
 */

import { log } from 'apify';

let embeddingPipeline = null;
let currentModel = null;

export const EMBEDDING_MODELS = {
  'all-MiniLM-L6-v2': { id: 'Xenova/all-MiniLM-L6-v2', dimensions: 384, speed: 'fast', quality: 'good' },
  'bge-small-en-v1.5': { id: 'Xenova/bge-small-en-v1.5', dimensions: 384, speed: 'fast', quality: 'excellent' },
  'all-mpnet-base-v2': { id: 'Xenova/all-mpnet-base-v2', dimensions: 768, speed: 'medium', quality: 'excellent' },
  'e5-small-v2': { id: 'Xenova/e5-small-v2', dimensions: 384, speed: 'fast', quality: 'very-good' },
  'gte-small': { id: 'Xenova/gte-small', dimensions: 384, speed: 'fast', quality: 'very-good' }
};

export async function initEmbeddingPipeline(modelName = 'all-MiniLM-L6-v2') {
  const modelConfig = EMBEDDING_MODELS[modelName];
  if (!modelConfig) throw new Error(`Unknown model: ${modelName}`);
  if (embeddingPipeline && currentModel === modelName) return embeddingPipeline;

  const { pipeline } = await import('@xenova/transformers');
  log.info(`Loading ONNX embedding model: ${modelConfig.id}...`);
  embeddingPipeline = await pipeline('feature-extraction', modelConfig.id, { quantized: true });
  currentModel = modelName;
  return embeddingPipeline;
}

export async function generateEmbedding(text, options = {}) {
  const { modelName = 'all-MiniLM-L6-v2', normalize = true } = options;
  const pipe = await initEmbeddingPipeline(modelName);
  const output = await pipe(text.substring(0, 8000), { pooling: 'mean', normalize });
  return Array.from(output.data);
}

export async function generateEmbeddingsBatch(texts, options = {}) {
  const { modelName = 'all-MiniLM-L6-v2', batchSize = 32, onProgress = null } = options;
  const pipe = await initEmbeddingPipeline(modelName);
  const embeddings = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const batchEmbeddings = await Promise.all(
      batch.map(async (text) => {
        const output = await pipe((text || '').substring(0, 8000), { pooling: 'mean', normalize: true });
        return Array.from(output.data);
      })
    );
    embeddings.push(...batchEmbeddings);
    if (onProgress) onProgress({ processed: Math.min(i + batchSize, texts.length), total: texts.length });
  }
  return embeddings;
}

export async function addEmbeddingsToRecords(records, options = {}) {
  const { modelName = 'all-MiniLM-L6-v2', textFields = ['title', 'description', 'text', 'content', 'caption', 'body', 'name'] } = options;
  if (!records?.length) return records;

  const modelConfig = EMBEDDING_MODELS[modelName];
  log.info(`Generating ONNX embeddings for ${records.length} records with ${modelName}`);

  const texts = records.map(record => {
    const parts = textFields.map(f => {
      const v = record[f] || record.data?.[f];
      return typeof v === 'string' ? v : Array.isArray(v) ? v.join(' ') : '';
    }).filter(Boolean);
    return parts.join(' ') || 'empty';
  });

  const embeddings = await generateEmbeddingsBatch(texts, { modelName });
  return records.map((record, i) => ({ ...record, embedding: embeddings[i], embeddingModel: modelName, embeddingDimensions: modelConfig.dimensions }));
}

export function generateRandomEmbedding(dimensions, random = Math.random) {
  const embedding = [];
  let norm = 0;
  for (let i = 0; i < dimensions; i++) {
    const val = random() * 2 - 1;
    embedding.push(val);
    norm += val * val;
  }
  norm = Math.sqrt(norm);
  return embedding.map(v => Math.round((v / norm) * 1000000) / 1000000);
}

export function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
