/**
 * Conversation Dataset Generator
 * Generates multi-turn conversation datasets for training AI agents
 */

const CONVERSATION_TEMPLATES = {
    customer_support: [
        {
            user: "I can't log into my account. It keeps saying invalid password.",
            assistant: "I understand you're having trouble logging in. Let me help you with that. First, could you confirm the email address associated with your account?",
        },
        {
            user: "My order hasn't arrived yet and it's been 2 weeks.",
            assistant: "I apologize for the delay with your order. Let me look into this for you. Could you please provide your order number?",
        },
    ],
    coding: [
        {
            user: "How do I sort an array of objects by a property in JavaScript?",
            assistant: "You can use the `sort()` method with a compare function. Here's an example:\n\n```javascript\nconst items = [{name: 'apple', price: 2}, {name: 'banana', price: 1}];\nitems.sort((a, b) => a.price - b.price);\n```\n\nThis sorts by price in ascending order.",
        },
        {
            user: "What's the difference between let and const?",
            assistant: "`let` allows you to reassign the variable, while `const` creates a constant reference. With `const`, you can't reassign the variable, but you can still modify object properties or array elements.",
        },
    ],
    research: [
        {
            user: "What are the main causes of climate change?",
            assistant: "The main causes of climate change include: 1) Greenhouse gas emissions from burning fossil fuels, 2) Deforestation reducing CO2 absorption, 3) Industrial processes, 4) Agriculture and livestock production. The primary driver is human activity increasing atmospheric CO2 levels.",
        },
        {
            user: "How does machine learning differ from traditional programming?",
            assistant: "Traditional programming uses explicit instructions, while machine learning learns patterns from data. In traditional programming, you write rules; in ML, the algorithm discovers rules by analyzing examples. ML excels at tasks where rules are complex or unknown.",
        },
    ],
};

/**
 * Generate realistic conversation datasets
 */
export async function generateConversations(config) {
    const {
        domain,
        count,
        complexity,
        includeMetadata,
        groundingData,
        customPrompts,
    } = config;

    const conversations = [];
    const templates = CONVERSATION_TEMPLATES[domain] || CONVERSATION_TEMPLATES.customer_support;

    for (let i = 0; i < count; i++) {
        const conversation = await generateSingleConversation({
            templates,
            complexity,
            groundingData,
            index: i,
        });

        if (includeMetadata) {
            conversation.metadata = {
                id: `conv_${i.toString().padStart(6, '0')}`,
                domain,
                complexity,
                turn_count: conversation.messages.length,
                generated_at: new Date().toISOString(),
                grounded: !!groundingData,
            };
        }

        conversations.push(conversation);
    }

    return conversations;
}

/**
 * Generate a single conversation with varying complexity
 */
async function generateSingleConversation({ templates, complexity, groundingData, index }) {
    const baseTemplate = templates[index % templates.length];
    const messages = [];

    // Start with template
    messages.push({
        role: 'user',
        content: varyContent(baseTemplate.user, complexity),
    });

    messages.push({
        role: 'assistant',
        content: varyContent(baseTemplate.assistant, complexity),
    });

    // Add more turns based on complexity
    const additionalTurns = getAdditionalTurns(complexity);
    for (let i = 0; i < additionalTurns; i++) {
        // Generate follow-up based on context
        const followUp = generateFollowUp(messages, complexity, groundingData);
        messages.push(...followUp);
    }

    return {
        type: 'conversation',
        messages,
    };
}

/**
 * Vary content based on complexity
 */
function varyContent(content, complexity) {
    switch (complexity) {
        case 'simple':
            return content;
        case 'moderate':
            return addContext(content);
        case 'complex':
            return addContextAndDetails(content);
        case 'expert':
            return addExpertLevelDetails(content);
        default:
            return content;
    }
}

/**
 * Get number of additional conversation turns based on complexity
 */
function getAdditionalTurns(complexity) {
    const turnMap = {
        simple: 0,
        moderate: 1,
        complex: 2,
        expert: 3,
    };
    return turnMap[complexity] || 0;
}

/**
 * Generate follow-up messages based on conversation context
 */
function generateFollowUp(messages, complexity, groundingData) {
    const lastMessage = messages[messages.length - 1];
    const followUps = [];

    if (lastMessage.role === 'assistant') {
        // User follow-up
        const userFollowUps = [
            "Thanks, that helps! Can you provide more details?",
            "I see. What about edge cases?",
            "Could you explain that differently?",
            "That makes sense. Is there anything else I should know?",
        ];

        followUps.push({
            role: 'user',
            content: userFollowUps[Math.floor(Math.random() * userFollowUps.length)],
        });

        // Assistant response
        const assistantFollowUps = [
            "Certainly! Let me elaborate on that...",
            "Good question. Here's more information...",
            "Of course. The key points to consider are...",
            "Yes, there are a few important considerations...",
        ];

        followUps.push({
            role: 'assistant',
            content: assistantFollowUps[Math.floor(Math.random() * assistantFollowUps.length)],
        });
    }

    return followUps;
}

/**
 * Add contextual information to content
 */
function addContext(content) {
    const contexts = [
        "I've been using this for a while now, but ",
        "Just to clarify, ",
        "Following up on this: ",
    ];
    return contexts[Math.floor(Math.random() * contexts.length)] + content;
}

/**
 * Add context and details
 */
function addContextAndDetails(content) {
    return addContext(content) + " I've tried several approaches already.";
}

/**
 * Add expert-level details
 */
function addExpertLevelDetails(content) {
    return addContextAndDetails(content) + " I'm particularly interested in the underlying mechanisms and best practices.";
}
