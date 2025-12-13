/**
 * Tool Calls Generator
 * Generates function calling sequences for AI agents
 */

const TOOL_SCHEMAS = {
    web_search: {
        name: 'web_search',
        description: 'Search the web for information',
        parameters: {
            type: 'object',
            properties: {
                query: { type: 'string', description: 'Search query' },
                max_results: { type: 'integer', description: 'Maximum results to return' },
            },
            required: ['query'],
        },
    },
    code_execution: {
        name: 'execute_code',
        description: 'Execute code in a sandboxed environment',
        parameters: {
            type: 'object',
            properties: {
                language: { type: 'string', enum: ['python', 'javascript', 'bash'] },
                code: { type: 'string', description: 'Code to execute' },
            },
            required: ['language', 'code'],
        },
    },
    api_call: {
        name: 'call_api',
        description: 'Make an API request',
        parameters: {
            type: 'object',
            properties: {
                url: { type: 'string', description: 'API endpoint URL' },
                method: { type: 'string', enum: ['GET', 'POST', 'PUT', 'DELETE'] },
                headers: { type: 'object', description: 'Request headers' },
                body: { type: 'object', description: 'Request body' },
            },
            required: ['url', 'method'],
        },
    },
    database_query: {
        name: 'query_database',
        description: 'Query a database',
        parameters: {
            type: 'object',
            properties: {
                query: { type: 'string', description: 'SQL query' },
                database: { type: 'string', description: 'Database name' },
            },
            required: ['query'],
        },
    },
    file_operation: {
        name: 'file_operation',
        description: 'Perform file operations',
        parameters: {
            type: 'object',
            properties: {
                operation: { type: 'string', enum: ['read', 'write', 'delete', 'list'] },
                path: { type: 'string', description: 'File path' },
                content: { type: 'string', description: 'File content (for write operation)' },
            },
            required: ['operation', 'path'],
        },
    },
};

/**
 * Generate tool call sequences
 */
export async function generateToolCalls(config) {
    const {
        domain,
        count,
        complexity,
        includeMetadata,
        groundingData,
    } = config;

    const toolCalls = [];

    for (let i = 0; i < count; i++) {
        const sequence = await generateToolCallSequence({
            complexity,
            groundingData,
            index: i,
            domain,
        });

        if (includeMetadata) {
            sequence.metadata = {
                id: `tool_${i.toString().padStart(6, '0')}`,
                domain,
                complexity,
                tool_count: sequence.calls.length,
                generated_at: new Date().toISOString(),
                grounded: !!groundingData,
            };
        }

        toolCalls.push(sequence);
    }

    return toolCalls;
}

/**
 * Generate a sequence of tool calls
 */
async function generateToolCallSequence({ complexity, groundingData, index, domain }) {
    const calls = [];
    const callCount = getToolCallCount(complexity);

    // Select tools based on domain
    const availableTools = selectToolsForDomain(domain);

    for (let i = 0; i < callCount; i++) {
        const tool = availableTools[i % availableTools.length];
        const call = generateToolCall(tool, complexity, i);
        calls.push(call);
    }

    return {
        type: 'tool_call_sequence',
        calls,
        complexity,
    };
}

/**
 * Get number of tool calls based on complexity
 */
function getToolCallCount(complexity) {
    const countMap = {
        simple: 1,
        moderate: 2,
        complex: 3,
        expert: 5,
    };
    return countMap[complexity] || 1;
}

/**
 * Select tools based on domain
 */
function selectToolsForDomain(domain) {
    const domainTools = {
        customer_support: ['web_search', 'database_query', 'api_call'],
        coding: ['code_execution', 'file_operation', 'web_search'],
        research: ['web_search', 'api_call', 'database_query'],
    };

    const toolNames = domainTools[domain] || ['web_search', 'api_call'];
    return toolNames.map(name => TOOL_SCHEMAS[name]);
}

/**
 * Generate a single tool call
 */
function generateToolCall(toolSchema, complexity, index) {
    const call = {
        id: `call_${index}`,
        type: 'function',
        function: {
            name: toolSchema.name,
            arguments: generateArguments(toolSchema, complexity),
        },
    };

    // Add execution result for more complete training data
    call.result = generateMockResult(toolSchema, complexity);

    return call;
}

/**
 * Generate arguments for a tool call
 */
function generateArguments(toolSchema, complexity) {
    const args = {};
    const params = toolSchema.parameters.properties;

    for (const [paramName, paramDef] of Object.entries(params)) {
        if (toolSchema.parameters.required?.includes(paramName) || Math.random() > 0.5) {
            args[paramName] = generateArgumentValue(paramDef, complexity);
        }
    }

    return JSON.stringify(args);
}

/**
 * Generate a value for a parameter
 */
function generateArgumentValue(paramDef, complexity) {
    switch (paramDef.type) {
        case 'string':
            if (paramDef.enum) {
                return paramDef.enum[Math.floor(Math.random() * paramDef.enum.length)];
            }
            return generateStringValue(paramDef, complexity);
        case 'integer':
            return Math.floor(Math.random() * 100) + 1;
        case 'object':
            return { key: 'value' };
        default:
            return null;
    }
}

/**
 * Generate string value based on parameter definition
 */
function generateStringValue(paramDef, complexity) {
    const examples = {
        query: ['latest news', 'how to implement feature', 'best practices'],
        code: ['print("Hello, World!")', 'const x = 10;', 'echo "test"'],
        url: ['https://api.example.com/data', 'https://api.github.com/repos'],
        path: ['/tmp/file.txt', '/data/output.json', '/home/user/document.pdf'],
    };

    for (const [key, values] of Object.entries(examples)) {
        if (paramDef.description?.toLowerCase().includes(key)) {
            return values[Math.floor(Math.random() * values.length)];
        }
    }

    return 'example value';
}

/**
 * Generate mock result for a tool call
 */
function generateMockResult(toolSchema, complexity) {
    const resultExamples = {
        web_search: {
            results: [
                { title: 'Example Result', url: 'https://example.com', snippet: 'This is an example result...' },
            ],
        },
        execute_code: {
            output: 'Hello, World!\n',
            exit_code: 0,
        },
        call_api: {
            status: 200,
            data: { success: true, message: 'API call successful' },
        },
        query_database: {
            rows: [{ id: 1, name: 'Example' }],
            count: 1,
        },
        file_operation: {
            success: true,
            message: 'File operation completed',
        },
    };

    return resultExamples[toolSchema.name] || { success: true };
}
