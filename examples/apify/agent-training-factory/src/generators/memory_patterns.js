/**
 * Memory Patterns Generator
 * Generates memory patterns for AI agents (short-term, long-term, episodic)
 */

/**
 * Generate memory pattern datasets
 */
export async function generateMemoryPatterns(config) {
    const {
        domain,
        count,
        complexity,
        includeMetadata,
        groundingData,
    } = config;

    const memoryPatterns = [];

    for (let i = 0; i < count; i++) {
        const pattern = await generateMemoryPattern({
            complexity,
            groundingData,
            index: i,
            domain,
        });

        if (includeMetadata) {
            pattern.metadata = {
                id: `memory_${i.toString().padStart(6, '0')}`,
                domain,
                complexity,
                memory_type: pattern.memory_type,
                generated_at: new Date().toISOString(),
                grounded: !!groundingData,
            };
        }

        memoryPatterns.push(pattern);
    }

    return memoryPatterns;
}

/**
 * Generate a single memory pattern
 */
async function generateMemoryPattern({ complexity, groundingData, index, domain }) {
    const memoryType = selectMemoryType(complexity);

    const pattern = {
        type: 'memory_pattern',
        memory_type: memoryType,
        complexity,
    };

    switch (memoryType) {
        case 'short_term':
            Object.assign(pattern, generateShortTermMemory(domain, index));
            break;
        case 'long_term':
            Object.assign(pattern, generateLongTermMemory(domain, index));
            break;
        case 'episodic':
            Object.assign(pattern, generateEpisodicMemory(domain, index));
            break;
        case 'semantic':
            Object.assign(pattern, generateSemanticMemory(domain, index));
            break;
        case 'procedural':
            Object.assign(pattern, generateProceduralMemory(domain, index));
            break;
        default:
            Object.assign(pattern, generateShortTermMemory(domain, index));
    }

    return pattern;
}

/**
 * Select memory type based on complexity
 */
function selectMemoryType(complexity) {
    const types = {
        simple: ['short_term'],
        moderate: ['short_term', 'long_term'],
        complex: ['short_term', 'long_term', 'episodic'],
        expert: ['short_term', 'long_term', 'episodic', 'semantic', 'procedural'],
    };

    const availableTypes = types[complexity] || ['short_term'];
    return availableTypes[Math.floor(Math.random() * availableTypes.length)];
}

/**
 * Generate short-term memory pattern
 */
function generateShortTermMemory(domain, index) {
    return {
        context: `Current conversation in ${domain}`,
        working_memory: [
            {
                key: 'last_user_input',
                value: 'User asked about feature X',
                timestamp: Date.now() - 1000,
                ttl: 300000, // 5 minutes
            },
            {
                key: 'conversation_state',
                value: 'awaiting_clarification',
                timestamp: Date.now(),
                ttl: 600000, // 10 minutes
            },
        ],
        capacity: 7, // Miller's Law: 7±2 items
        retrieval_time: 'immediate',
    };
}

/**
 * Generate long-term memory pattern
 */
function generateLongTermMemory(domain, index) {
    return {
        context: `Historical data for ${domain}`,
        stored_facts: [
            {
                fact: 'User prefers detailed explanations',
                confidence: 0.85,
                reinforcement_count: 5,
                first_learned: Date.now() - 86400000 * 30, // 30 days ago
                last_accessed: Date.now() - 86400000, // 1 day ago
            },
            {
                fact: 'Common issue: authentication problems',
                confidence: 0.92,
                reinforcement_count: 15,
                first_learned: Date.now() - 86400000 * 60,
                last_accessed: Date.now() - 3600000,
            },
        ],
        consolidation_status: 'consolidated',
        retrieval_time: 'seconds',
    };
}

/**
 * Generate episodic memory pattern
 */
function generateEpisodicMemory(domain, index) {
    return {
        context: `Event sequence in ${domain}`,
        episode: {
            id: `episode_${index}`,
            sequence: [
                {
                    timestamp: Date.now() - 3600000,
                    event: 'user_login',
                    details: { user_id: 'user_123', method: 'oauth' },
                },
                {
                    timestamp: Date.now() - 3000000,
                    event: 'feature_accessed',
                    details: { feature: 'dashboard', duration: 300 },
                },
                {
                    timestamp: Date.now() - 1800000,
                    event: 'issue_encountered',
                    details: { type: 'error', code: 500 },
                },
                {
                    timestamp: Date.now() - 600000,
                    event: 'support_contacted',
                    details: { channel: 'chat', resolution_time: 900 },
                },
            ],
            context_tags: ['error_handling', 'user_support', 'dashboard_issue'],
            emotional_valence: -0.3, // Negative due to error
        },
        retrieval_cues: ['error 500', 'dashboard', 'support'],
        retrieval_time: 'seconds_to_minutes',
    };
}

/**
 * Generate semantic memory pattern
 */
function generateSemanticMemory(domain, index) {
    return {
        context: `Knowledge graph for ${domain}`,
        concepts: [
            {
                concept: 'authentication',
                related_concepts: ['security', 'oauth', 'jwt', 'sessions'],
                definition: 'Process of verifying user identity',
                examples: ['login', 'SSO', 'MFA'],
                hierarchical_level: 'intermediate',
            },
            {
                concept: 'API',
                related_concepts: ['REST', 'GraphQL', 'endpoints', 'requests'],
                definition: 'Application Programming Interface for system communication',
                examples: ['REST API', 'GraphQL API', 'WebSocket'],
                hierarchical_level: 'fundamental',
            },
        ],
        relationships: [
            { from: 'authentication', to: 'API', type: 'uses' },
            { from: 'API', to: 'endpoints', type: 'contains' },
        ],
        abstraction_level: 'conceptual',
        retrieval_time: 'immediate_to_seconds',
    };
}

/**
 * Generate procedural memory pattern
 */
function generateProceduralMemory(domain, index) {
    return {
        context: `Learned procedures for ${domain}`,
        procedures: [
            {
                name: 'handle_authentication_error',
                steps: [
                    { step: 1, action: 'verify_credentials', automatic: true },
                    { step: 2, action: 'check_session_validity', automatic: true },
                    { step: 3, action: 'attempt_token_refresh', automatic: false },
                    { step: 4, action: 'fallback_to_login', automatic: false },
                ],
                proficiency: 0.85,
                execution_count: 47,
                success_rate: 0.92,
            },
            {
                name: 'process_user_query',
                steps: [
                    { step: 1, action: 'parse_intent', automatic: true },
                    { step: 2, action: 'extract_entities', automatic: true },
                    { step: 3, action: 'retrieve_context', automatic: true },
                    { step: 4, action: 'generate_response', automatic: false },
                    { step: 5, action: 'validate_response', automatic: false },
                ],
                proficiency: 0.93,
                execution_count: 234,
                success_rate: 0.95,
            },
        ],
        automaticity_level: 'mixed',
        retrieval_time: 'automatic',
    };
}
