/**
 * Grounding Utilities
 * Fetch real data from Apify actors/datasets to ground synthetic data
 */

import { Actor } from 'apify';

/**
 * Fetch grounding data from Apify actor or dataset
 */
export async function fetchGroundingData({ actorId, datasetId }) {
    if (datasetId) {
        return fetchFromDataset(datasetId);
    } else if (actorId) {
        return fetchFromActor(actorId);
    }

    return null;
}

/**
 * Fetch data from existing Apify dataset
 */
async function fetchFromDataset(datasetId) {
    try {
        console.log(`Fetching grounding data from dataset: ${datasetId}`);

        const client = Actor.apifyClient;
        const dataset = await client.dataset(datasetId);
        const { items } = await dataset.listItems({ limit: 100 });

        console.log(`Fetched ${items.length} items from dataset`);
        return items;
    } catch (error) {
        console.error('Error fetching from dataset:', error.message);
        return [];
    }
}

/**
 * Fetch data from Apify actor run
 */
async function fetchFromActor(actorId) {
    try {
        console.log(`Fetching grounding data from actor: ${actorId}`);

        const client = Actor.apifyClient;

        // Get the most recent successful run
        const actor = await client.actor(actorId);
        const { items: runs } = await actor.runs().list({
            limit: 1,
            status: 'SUCCEEDED',
        });

        if (runs.length === 0) {
            console.warn('No successful runs found for actor');
            return [];
        }

        const lastRun = runs[0];
        const dataset = await client.dataset(lastRun.defaultDatasetId);
        const { items } = await dataset.listItems({ limit: 100 });

        console.log(`Fetched ${items.length} items from actor run`);
        return items;
    } catch (error) {
        console.error('Error fetching from actor:', error.message);
        return [];
    }
}

/**
 * Extract relevant features from grounding data
 */
export function extractFeatures(groundingData) {
    if (!groundingData || groundingData.length === 0) {
        return {};
    }

    const features = {
        common_fields: extractCommonFields(groundingData),
        text_patterns: extractTextPatterns(groundingData),
        structural_patterns: extractStructuralPatterns(groundingData),
    };

    return features;
}

/**
 * Extract common fields across grounding data
 */
function extractCommonFields(data) {
    const fieldCounts = {};

    data.forEach(item => {
        Object.keys(item).forEach(key => {
            fieldCounts[key] = (fieldCounts[key] || 0) + 1;
        });
    });

    // Return fields present in at least 50% of items
    const threshold = data.length * 0.5;
    return Object.entries(fieldCounts)
        .filter(([_, count]) => count >= threshold)
        .map(([field, _]) => field);
}

/**
 * Extract text patterns from grounding data
 */
function extractTextPatterns(data) {
    const patterns = {
        avgLength: 0,
        commonWords: new Map(),
        topics: new Set(),
    };

    let totalLength = 0;
    let textCount = 0;

    data.forEach(item => {
        // Find text fields
        const textFields = ['text', 'content', 'description', 'title', 'body'];

        textFields.forEach(field => {
            if (item[field] && typeof item[field] === 'string') {
                const text = item[field];
                totalLength += text.length;
                textCount++;

                // Extract words
                const words = text.toLowerCase().match(/\b\w+\b/g) || [];
                words.forEach(word => {
                    patterns.commonWords.set(word, (patterns.commonWords.get(word) || 0) + 1);
                });
            }
        });
    });

    patterns.avgLength = textCount > 0 ? totalLength / textCount : 0;

    return patterns;
}

/**
 * Extract structural patterns from grounding data
 */
function extractStructuralPatterns(data) {
    return {
        itemCount: data.length,
        hasArrays: data.some(item => Object.values(item).some(Array.isArray)),
        hasNested: data.some(item => Object.values(item).some(v => typeof v === 'object' && v !== null)),
        avgFieldCount: data.reduce((sum, item) => sum + Object.keys(item).length, 0) / data.length,
    };
}
