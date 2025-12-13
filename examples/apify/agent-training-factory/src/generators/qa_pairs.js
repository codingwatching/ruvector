/**
 * Q&A Pairs Generator
 * Generates question-answer pairs for training and fine-tuning
 */

const QA_TEMPLATES = {
    customer_support: [
        { q: "How do I reset my password?", a: "To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the link sent to you." },
        { q: "What is your return policy?", a: "We offer a 30-day return policy for most items. Products must be unused and in original packaging." },
        { q: "How long does shipping take?", a: "Standard shipping takes 5-7 business days. Express shipping is available for 2-3 day delivery." },
    ],
    coding: [
        { q: "What is a closure in JavaScript?", a: "A closure is a function that has access to variables in its outer (enclosing) lexical scope, even after the outer function has returned." },
        { q: "How do you reverse a string in Python?", a: "Use slicing: `reversed_string = original[::-1]` or the reversed() function: `''.join(reversed(original))`" },
        { q: "What is the difference between == and === in JavaScript?", a: "== compares values with type coercion, while === compares both value and type without coercion (strict equality)." },
    ],
    research: [
        { q: "What is the scientific method?", a: "The scientific method is a systematic approach to research involving observation, hypothesis formation, experimentation, analysis, and conclusion." },
        { q: "How does photosynthesis work?", a: "Photosynthesis converts light energy into chemical energy, using CO2 and water to produce glucose and oxygen in chloroplasts." },
        { q: "What is quantum entanglement?", a: "Quantum entanglement is a phenomenon where particles become correlated such that the quantum state of one particle cannot be described independently of others." },
    ],
};

/**
 * Generate Q&A pairs dataset
 */
export async function generateQAPairs(config) {
    const {
        domain,
        count,
        complexity,
        includeMetadata,
        groundingData,
    } = config;

    const qaPairs = [];
    const templates = QA_TEMPLATES[domain] || QA_TEMPLATES.customer_support;

    for (let i = 0; i < count; i++) {
        const qa = await generateSingleQAPair({
            templates,
            complexity,
            groundingData,
            index: i,
        });

        if (includeMetadata) {
            qa.metadata = {
                id: `qa_${i.toString().padStart(6, '0')}`,
                domain,
                complexity,
                question_length: qa.question.length,
                answer_length: qa.answer.length,
                generated_at: new Date().toISOString(),
                grounded: !!groundingData,
            };
        }

        qaPairs.push(qa);
    }

    return qaPairs;
}

/**
 * Generate a single Q&A pair
 */
async function generateSingleQAPair({ templates, complexity, groundingData, index }) {
    const baseTemplate = templates[index % templates.length];

    let question = varyQuestion(baseTemplate.q, complexity);
    let answer = varyAnswer(baseTemplate.a, complexity);

    // Use grounding data if available
    if (groundingData && groundingData.length > 0) {
        const groundingSample = groundingData[index % groundingData.length];
        question = enhanceWithGroundingData(question, groundingSample);
    }

    return {
        type: 'qa_pair',
        question,
        answer,
        complexity,
    };
}

/**
 * Vary question based on complexity
 */
function varyQuestion(question, complexity) {
    switch (complexity) {
        case 'simple':
            return question;
        case 'moderate':
            return `Could you explain ${question.toLowerCase()}`;
        case 'complex':
            return `I'm trying to understand ${question.toLowerCase()} Can you provide a detailed explanation with examples?`;
        case 'expert':
            return `From a technical perspective, ${question.toLowerCase()} I'm particularly interested in edge cases and best practices.`;
        default:
            return question;
    }
}

/**
 * Vary answer based on complexity
 */
function varyAnswer(answer, complexity) {
    switch (complexity) {
        case 'simple':
            return answer;
        case 'moderate':
            return `${answer} This approach is commonly used in many applications.`;
        case 'complex':
            return `${answer} For more advanced use cases, you should also consider error handling, performance implications, and scalability. Here are some best practices to keep in mind...`;
        case 'expert':
            return `${answer} From an architectural standpoint, this involves several key considerations: 1) Performance optimization, 2) Memory management, 3) Error handling strategies, 4) Testing approaches, and 5) Production deployment considerations. Let's dive deeper into each aspect...`;
        default:
            return answer;
    }
}

/**
 * Enhance question with grounding data
 */
function enhanceWithGroundingData(question, groundingData) {
    if (!groundingData || typeof groundingData !== 'object') {
        return question;
    }

    // Extract relevant context from grounding data
    const context = extractContext(groundingData);
    if (context) {
        return `${question} (Context: ${context})`;
    }

    return question;
}

/**
 * Extract context from grounding data object
 */
function extractContext(data) {
    // Try to find text content in various common fields
    const textFields = ['text', 'content', 'description', 'title', 'body'];

    for (const field of textFields) {
        if (data[field] && typeof data[field] === 'string') {
            return data[field].substring(0, 100);
        }
    }

    return null;
}
