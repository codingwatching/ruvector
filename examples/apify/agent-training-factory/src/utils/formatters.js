/**
 * Output Formatters
 * Format generated data into various output formats
 */

/**
 * Format output based on requested format
 */
export async function formatOutput(data, format) {
    switch (format) {
        case 'jsonl':
            return formatAsJSONL(data);
        case 'parquet':
            return formatAsParquet(data);
        case 'huggingface':
            return formatAsHuggingFace(data);
        default:
            return data;
    }
}

/**
 * Format as JSONL (JSON Lines)
 */
function formatAsJSONL(data) {
    // For Apify dataset, we just return the data as-is
    // Apify will handle the actual formatting when downloaded
    return data;
}

/**
 * Format as Parquet-compatible structure
 */
function formatAsParquet(data) {
    // Convert nested structures to Parquet-compatible format
    return data.map(item => flattenForParquet(item));
}

/**
 * Flatten nested objects for Parquet
 */
function flattenForParquet(obj, prefix = '') {
    const flattened = {};

    for (const [key, value] of Object.entries(obj)) {
        const fullKey = prefix ? `${prefix}.${key}` : key;

        if (value === null || value === undefined) {
            flattened[fullKey] = null;
        } else if (Array.isArray(value)) {
            // Convert arrays to JSON strings for Parquet
            flattened[fullKey] = JSON.stringify(value);
        } else if (typeof value === 'object') {
            // Recursively flatten nested objects
            Object.assign(flattened, flattenForParquet(value, fullKey));
        } else {
            flattened[fullKey] = value;
        }
    }

    return flattened;
}

/**
 * Format as HuggingFace datasets compatible format
 */
function formatAsHuggingFace(data) {
    return data.map(item => {
        const formatted = { ...item };

        // HuggingFace expects specific structure for certain dataset types
        if (item.type === 'conversation') {
            formatted.conversations = item.messages || [];
            delete formatted.messages;
        } else if (item.type === 'qa_pair') {
            formatted.question = item.question || '';
            formatted.answer = item.answer || '';
        } else if (item.type === 'embedding') {
            formatted.embeddings = item.vector || [];
            delete formatted.vector;
        }

        // Add HuggingFace metadata
        formatted._hf_metadata = {
            dataset_name: 'agent_training_data',
            version: '1.0.0',
            license: 'apache-2.0',
        };

        return formatted;
    });
}

/**
 * Format embeddings for vector database ingestion
 */
export function formatForVectorDB(data, options = {}) {
    const { namespace = 'default', includeMetadata = true } = options;

    return data.map((item, index) => {
        const vector = item.vector || item.embeddings || [];

        const entry = {
            id: item.metadata?.id || `vec_${index}`,
            vector,
            namespace,
        };

        if (includeMetadata && item.metadata) {
            entry.metadata = {
                ...item.metadata,
                text: item.text,
                type: item.type,
            };
        }

        return entry;
    });
}

/**
 * Format conversations for fine-tuning
 */
export function formatForFineTuning(data, modelType = 'openai') {
    switch (modelType) {
        case 'openai':
            return formatForOpenAI(data);
        case 'anthropic':
            return formatForAnthropic(data);
        case 'huggingface':
            return formatAsHuggingFace(data);
        default:
            return data;
    }
}

/**
 * Format for OpenAI fine-tuning
 */
function formatForOpenAI(data) {
    return data.map(item => {
        if (item.type === 'conversation' && item.messages) {
            return {
                messages: item.messages.map(msg => ({
                    role: msg.role,
                    content: msg.content,
                })),
            };
        } else if (item.type === 'qa_pair') {
            return {
                messages: [
                    { role: 'user', content: item.question },
                    { role: 'assistant', content: item.answer },
                ],
            };
        }
        return item;
    });
}

/**
 * Format for Anthropic fine-tuning
 */
function formatForAnthropic(data) {
    return data.map(item => {
        if (item.type === 'conversation' && item.messages) {
            return {
                conversation: item.messages,
            };
        } else if (item.type === 'qa_pair') {
            return {
                conversation: [
                    { role: 'user', content: item.question },
                    { role: 'assistant', content: item.answer },
                ],
            };
        }
        return item;
    });
}
