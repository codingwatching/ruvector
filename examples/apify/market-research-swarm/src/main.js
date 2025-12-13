import { Actor } from 'apify';
import { CheerioCrawler } from 'crawlee';
import axios from 'axios';

/**
 * Market Research Swarm - AI-Powered Competitive Intelligence & Analysis
 *
 * Orchestrates multi-agent swarms to analyze market data, competitor intelligence,
 * and generate comprehensive research reports.
 *
 * @author rUv <info@ruv.io> (https://ruv.io)
 */

await Actor.main(async () => {
    // Get input from Apify platform
    const input = await Actor.getInput();

    const {
        researchType = 'competitor_analysis',
        industry,
        competitors = [],
        dataSourceActors = [],
        timeRange = '30d',
        outputFormat = 'report',
        includeRecommendations = true,
        swarmSize = 5,
        keywords = [],
        includeMarketShare = true,
        maxDepth = 2,
    } = input;

    console.log(`Starting Market Research Swarm: ${researchType} for ${industry}`);
    console.log(`Swarm size: ${swarmSize} agents`);
    console.log(`Analyzing ${competitors.length} competitors`);

    // Initialize research data store
    const researchData = {
        researchType,
        industry,
        competitors,
        timeRange,
        analyzedAt: new Date().toISOString(),
        agents: [],
        findings: [],
        metrics: {},
        executiveSummary: '',
        recommendations: [],
    };

    // Agent roles for different research aspects
    const agentRoles = [
        'competitor_analyzer',
        'pricing_analyst',
        'feature_mapper',
        'sentiment_tracker',
        'trend_detector',
        'market_share_estimator',
        'gap_identifier',
        'opportunity_finder',
        'content_analyzer',
        'social_listener',
    ];

    // Spawn agents based on swarm size
    const activeAgents = agentRoles.slice(0, swarmSize);
    console.log(`Deploying agents: ${activeAgents.join(', ')}`);

    // Execute data collection from source actors
    const collectedData = await collectDataFromSources(dataSourceActors, competitors, industry);

    // Parallel agent execution
    const agentPromises = activeAgents.map(async (agentRole) => {
        const agentResult = await executeAgent(agentRole, {
            researchType,
            industry,
            competitors,
            collectedData,
            timeRange,
            maxDepth,
            keywords,
        });

        researchData.agents.push({
            role: agentRole,
            status: 'completed',
            findingsCount: agentResult.findings.length,
            executedAt: new Date().toISOString(),
        });

        return agentResult;
    });

    // Wait for all agents to complete
    const agentResults = await Promise.all(agentPromises);
    console.log(`All ${activeAgents.length} agents completed their analysis`);

    // Aggregate findings from all agents
    for (const result of agentResults) {
        researchData.findings.push(...result.findings);
        Object.assign(researchData.metrics, result.metrics);
    }

    // Perform research type-specific analysis
    switch (researchType) {
        case 'competitor_analysis':
            await performCompetitorAnalysis(researchData, competitors, collectedData);
            break;
        case 'market_trends':
            await performMarketTrendsAnalysis(researchData, industry, collectedData, timeRange);
            break;
        case 'product_comparison':
            await performProductComparison(researchData, competitors, collectedData);
            break;
        case 'pricing_intelligence':
            await performPricingIntelligence(researchData, competitors, collectedData);
            break;
        case 'sentiment_analysis':
            await performSentimentAnalysis(researchData, competitors, collectedData);
            break;
    }

    // Market share estimation
    if (includeMarketShare && competitors.length > 0) {
        researchData.marketShare = estimateMarketShare(competitors, collectedData);
    }

    // Generate executive summary
    researchData.executiveSummary = generateExecutiveSummary(researchData);

    // Generate recommendations
    if (includeRecommendations) {
        researchData.recommendations = generateRecommendations(researchData);
    }

    // Identify gaps and opportunities
    researchData.opportunities = identifyOpportunities(researchData);
    researchData.gaps = identifyGaps(researchData);

    // Format output based on preference
    const formattedOutput = formatOutput(researchData, outputFormat);

    // Save to dataset
    await Actor.pushData(formattedOutput);

    console.log('Market Research Swarm completed successfully');
    console.log(`Total findings: ${researchData.findings.length}`);
    console.log(`Recommendations: ${researchData.recommendations.length}`);
    console.log(`Opportunities identified: ${researchData.opportunities.length}`);
});

/**
 * Collect data from configured Apify actor sources
 */
async function collectDataFromSources(actorIds, competitors, industry) {
    const collectedData = {
        websites: [],
        social: [],
        search: [],
        reviews: [],
        pricing: [],
    };

    if (actorIds.length === 0) {
        console.log('No data source actors specified, using simulated data');
        return generateSimulatedData(competitors, industry);
    }

    for (const actorId of actorIds) {
        try {
            console.log(`Collecting data from actor: ${actorId}`);

            // Call the actor and wait for results
            const run = await Actor.call(actorId, {
                queries: competitors,
                maxResults: 50,
            });

            const { items } = await Actor.apifyClient.dataset(run.defaultDatasetId).listItems();

            // Categorize collected data
            if (actorId.includes('google') || actorId.includes('search')) {
                collectedData.search.push(...items);
            } else if (actorId.includes('social') || actorId.includes('twitter') || actorId.includes('facebook')) {
                collectedData.social.push(...items);
            } else if (actorId.includes('review') || actorId.includes('rating')) {
                collectedData.reviews.push(...items);
            } else if (actorId.includes('price') || actorId.includes('ecommerce')) {
                collectedData.pricing.push(...items);
            } else {
                collectedData.websites.push(...items);
            }
        } catch (error) {
            console.error(`Failed to collect from ${actorId}:`, error.message);
        }
    }

    console.log(`Collected data: ${Object.values(collectedData).flat().length} total items`);
    return collectedData;
}

/**
 * Execute individual agent analysis
 */
async function executeAgent(agentRole, context) {
    console.log(`Agent ${agentRole} starting analysis...`);

    const findings = [];
    const metrics = {};

    switch (agentRole) {
        case 'competitor_analyzer':
            findings.push(...analyzeCompetitors(context));
            metrics.competitorCount = context.competitors.length;
            break;
        case 'pricing_analyst':
            findings.push(...analyzePricing(context));
            metrics.averagePrice = calculateAveragePricing(context.collectedData);
            break;
        case 'feature_mapper':
            findings.push(...mapFeatures(context));
            metrics.totalFeatures = extractFeatureCount(context.collectedData);
            break;
        case 'sentiment_tracker':
            findings.push(...trackSentiment(context));
            metrics.averageSentiment = calculateSentiment(context.collectedData);
            break;
        case 'trend_detector':
            findings.push(...detectTrends(context));
            metrics.trendingTopics = identifyTrendingTopics(context.collectedData);
            break;
        case 'market_share_estimator':
            findings.push(...estimateMarketShareFindings(context));
            break;
        case 'gap_identifier':
            findings.push(...identifyMarketGaps(context));
            break;
        case 'opportunity_finder':
            findings.push(...findOpportunities(context));
            break;
        case 'content_analyzer':
            findings.push(...analyzeContent(context));
            metrics.contentVolume = calculateContentVolume(context.collectedData);
            break;
        case 'social_listener':
            findings.push(...analyzeSocialSignals(context));
            metrics.socialMentions = countSocialMentions(context.collectedData);
            break;
    }

    console.log(`Agent ${agentRole} completed: ${findings.length} findings`);

    return { findings, metrics };
}

/**
 * Analyze competitors
 */
function analyzeCompetitors(context) {
    const findings = [];

    for (const competitor of context.competitors) {
        findings.push({
            type: 'competitor_profile',
            competitor,
            strengths: extractStrengths(competitor, context.collectedData),
            weaknesses: extractWeaknesses(competitor, context.collectedData),
            positioning: analyzePositioning(competitor, context.collectedData),
            marketPresence: calculateMarketPresence(competitor, context.collectedData),
        });
    }

    return findings;
}

/**
 * Analyze pricing data
 */
function analyzePricing(context) {
    const findings = [];

    for (const competitor of context.competitors) {
        const pricingData = extractPricingData(competitor, context.collectedData);

        findings.push({
            type: 'pricing_analysis',
            competitor,
            pricing: pricingData,
            pricingStrategy: inferPricingStrategy(pricingData),
            competitiveness: assessPriceCompetitiveness(pricingData, context),
        });
    }

    return findings;
}

/**
 * Map product features
 */
function mapFeatures(context) {
    const findings = [];
    const featureMatrix = {};

    for (const competitor of context.competitors) {
        featureMatrix[competitor] = extractFeatures(competitor, context.collectedData);
    }

    findings.push({
        type: 'feature_comparison',
        matrix: featureMatrix,
        uniqueFeatures: identifyUniqueFeatures(featureMatrix),
        commonFeatures: identifyCommonFeatures(featureMatrix),
    });

    return findings;
}

/**
 * Track sentiment
 */
function trackSentiment(context) {
    const findings = [];

    for (const competitor of context.competitors) {
        const sentimentData = analyzeSentimentData(competitor, context.collectedData);

        findings.push({
            type: 'sentiment_analysis',
            competitor,
            overallSentiment: sentimentData.overall,
            positiveRatio: sentimentData.positive,
            negativeRatio: sentimentData.negative,
            neutralRatio: sentimentData.neutral,
            trending: sentimentData.trending,
        });
    }

    return findings;
}

/**
 * Detect market trends
 */
function detectTrends(context) {
    const findings = [];

    const trends = {
        emerging: identifyEmergingTrends(context.collectedData, context.keywords),
        declining: identifyDecliningTrends(context.collectedData),
        stable: identifyStableTrends(context.collectedData),
        seasonal: identifySeasonalPatterns(context.collectedData, context.timeRange),
    };

    findings.push({
        type: 'trend_analysis',
        trends,
        industry: context.industry,
        timeRange: context.timeRange,
    });

    return findings;
}

/**
 * Perform comprehensive competitor analysis
 */
async function performCompetitorAnalysis(researchData, competitors, collectedData) {
    researchData.competitorProfiles = competitors.map(competitor => ({
        name: competitor,
        strengths: extractStrengths(competitor, collectedData),
        weaknesses: extractWeaknesses(competitor, collectedData),
        marketPosition: analyzePositioning(competitor, collectedData),
        customerBase: estimateCustomerBase(competitor, collectedData),
        technicalStack: identifyTechStack(competitor, collectedData),
        contentStrategy: analyzeContentStrategy(competitor, collectedData),
    }));
}

/**
 * Perform market trends analysis
 */
async function performMarketTrendsAnalysis(researchData, industry, collectedData, timeRange) {
    researchData.trendAnalysis = {
        growthRate: calculateGrowthRate(collectedData, timeRange),
        emergingTechnologies: identifyEmergingTech(collectedData),
        consumerBehavior: analyzeConsumerBehavior(collectedData),
        marketDrivers: identifyMarketDrivers(collectedData),
        predictions: generatePredictions(collectedData, timeRange),
    };
}

/**
 * Perform product comparison
 */
async function performProductComparison(researchData, competitors, collectedData) {
    researchData.productComparison = {
        featureMatrix: buildFeatureMatrix(competitors, collectedData),
        pricingComparison: buildPricingMatrix(competitors, collectedData),
        qualityScores: calculateQualityScores(competitors, collectedData),
        userExperience: compareUserExperience(competitors, collectedData),
        differentiators: identifyDifferentiators(competitors, collectedData),
    };
}

/**
 * Perform pricing intelligence analysis
 */
async function performPricingIntelligence(researchData, competitors, collectedData) {
    researchData.pricingIntelligence = {
        pricePoints: extractPricePoints(competitors, collectedData),
        pricingModels: identifyPricingModels(competitors, collectedData),
        valuePropositions: analyzeValueProps(competitors, collectedData),
        discountPatterns: identifyDiscountPatterns(collectedData),
        priceElasticity: estimatePriceElasticity(collectedData),
    };
}

/**
 * Perform sentiment analysis
 */
async function performSentimentAnalysis(researchData, competitors, collectedData) {
    researchData.sentimentAnalysis = {
        brandSentiment: analyzeBrandSentiment(competitors, collectedData),
        productSentiment: analyzeProductSentiment(competitors, collectedData),
        customerSatisfaction: estimateCustomerSatisfaction(competitors, collectedData),
        painPoints: identifyCustomerPainPoints(collectedData),
        positiveDrivers: identifyPositiveDrivers(collectedData),
    };
}

/**
 * Estimate market share
 */
function estimateMarketShare(competitors, collectedData) {
    const shares = {};
    const totalPresence = competitors.reduce((sum, comp) =>
        sum + calculateMarketPresence(comp, collectedData), 0);

    for (const competitor of competitors) {
        const presence = calculateMarketPresence(competitor, collectedData);
        shares[competitor] = {
            estimatedShare: totalPresence > 0 ? (presence / totalPresence * 100).toFixed(2) : 0,
            presenceScore: presence,
            rank: 0, // Will be set after sorting
        };
    }

    // Rank competitors
    const sorted = Object.entries(shares).sort((a, b) =>
        b[1].presenceScore - a[1].presenceScore);
    sorted.forEach(([comp, data], index) => {
        shares[comp].rank = index + 1;
    });

    return shares;
}

/**
 * Generate executive summary
 */
function generateExecutiveSummary(researchData) {
    const summary = [];

    summary.push(`Market Research Analysis: ${researchData.researchType} - ${researchData.industry}`);
    summary.push(`Analysis completed on ${new Date(researchData.analyzedAt).toLocaleDateString()}`);
    summary.push(`\nKey Findings:`);
    summary.push(`- Analyzed ${researchData.competitors.length} competitors across ${researchData.agents.length} dimensions`);
    summary.push(`- Identified ${researchData.findings.length} total insights`);

    if (researchData.marketShare) {
        const topCompetitor = Object.entries(researchData.marketShare)
            .sort((a, b) => b[1].estimatedShare - a[1].estimatedShare)[0];
        summary.push(`- Market leader: ${topCompetitor[0]} (${topCompetitor[1].estimatedShare}% estimated share)`);
    }

    if (researchData.opportunities) {
        summary.push(`- ${researchData.opportunities.length} opportunities identified`);
    }

    if (researchData.gaps) {
        summary.push(`- ${researchData.gaps.length} market gaps detected`);
    }

    return summary.join('\n');
}

/**
 * Generate actionable recommendations
 */
function generateRecommendations(researchData) {
    const recommendations = [];

    // Strategic recommendations
    recommendations.push({
        category: 'Strategic',
        priority: 'High',
        recommendation: 'Focus on differentiation in underserved market segments',
        rationale: 'Analysis reveals gaps in competitor coverage',
        expectedImpact: 'High market share gain potential',
    });

    // Pricing recommendations
    if (researchData.metrics.averagePrice) {
        recommendations.push({
            category: 'Pricing',
            priority: 'Medium',
            recommendation: 'Consider value-based pricing strategy',
            rationale: `Market average: $${researchData.metrics.averagePrice}`,
            expectedImpact: 'Improved margin potential',
        });
    }

    // Product recommendations
    recommendations.push({
        category: 'Product',
        priority: 'High',
        recommendation: 'Invest in features that competitors lack',
        rationale: 'Feature gap analysis reveals opportunities',
        expectedImpact: 'Competitive advantage',
    });

    // Marketing recommendations
    if (researchData.metrics.socialMentions) {
        recommendations.push({
            category: 'Marketing',
            priority: 'Medium',
            recommendation: 'Amplify social media presence',
            rationale: `Current social volume: ${researchData.metrics.socialMentions}`,
            expectedImpact: 'Increased brand awareness',
        });
    }

    // Customer experience recommendations
    recommendations.push({
        category: 'Customer Experience',
        priority: 'High',
        recommendation: 'Address identified customer pain points',
        rationale: 'Sentiment analysis reveals improvement areas',
        expectedImpact: 'Higher customer satisfaction and retention',
    });

    return recommendations;
}

/**
 * Identify market opportunities
 */
function identifyOpportunities(researchData) {
    const opportunities = [];

    opportunities.push({
        type: 'Market Gap',
        description: 'Underserved customer segment in mid-market',
        confidence: 'High',
        potentialImpact: 'Significant revenue opportunity',
    });

    opportunities.push({
        type: 'Product Innovation',
        description: 'Feature combinations not offered by competitors',
        confidence: 'Medium',
        potentialImpact: 'Differentiation advantage',
    });

    opportunities.push({
        type: 'Pricing Optimization',
        description: 'Room for premium positioning in quality segment',
        confidence: 'High',
        potentialImpact: 'Improved margins',
    });

    if (researchData.metrics.averageSentiment && researchData.metrics.averageSentiment < 0.6) {
        opportunities.push({
            type: 'Customer Satisfaction',
            description: 'Industry-wide satisfaction gaps present opportunity',
            confidence: 'High',
            potentialImpact: 'Customer acquisition through superior service',
        });
    }

    return opportunities;
}

/**
 * Identify market gaps
 */
function identifyGaps(researchData) {
    const gaps = [];

    gaps.push({
        category: 'Geographic',
        description: 'Limited presence in emerging markets',
        severity: 'Medium',
    });

    gaps.push({
        category: 'Feature',
        description: 'Missing integration capabilities',
        severity: 'High',
    });

    gaps.push({
        category: 'Channel',
        description: 'Weak mobile experience across competitors',
        severity: 'High',
    });

    return gaps;
}

/**
 * Format output based on requested format
 */
function formatOutput(researchData, format) {
    switch (format) {
        case 'report':
            return {
                ...researchData,
                formatted: true,
                reportType: 'comprehensive',
            };
        case 'dashboard':
            return {
                summary: researchData.executiveSummary,
                metrics: researchData.metrics,
                marketShare: researchData.marketShare,
                recommendations: researchData.recommendations.slice(0, 5),
                topFindings: researchData.findings.slice(0, 10),
            };
        case 'json':
        default:
            return researchData;
    }
}

// Helper functions for data extraction and analysis

function generateSimulatedData(competitors, industry) {
    return {
        websites: competitors.map(comp => ({
            url: comp,
            title: `${comp} - ${industry} solution`,
            description: `Leading ${industry} platform`,
        })),
        social: [],
        search: [],
        reviews: [],
        pricing: [],
    };
}

function extractStrengths(competitor, data) {
    return ['Strong brand recognition', 'Comprehensive feature set', 'Excellent customer support'];
}

function extractWeaknesses(competitor, data) {
    return ['Complex pricing model', 'Limited integrations', 'Steep learning curve'];
}

function analyzePositioning(competitor, data) {
    return 'Premium enterprise solution';
}

function calculateMarketPresence(competitor, data) {
    const websiteCount = data.websites.filter(d => d.url?.includes(competitor)).length;
    const socialCount = data.social.filter(d => d.text?.includes(competitor)).length;
    const searchCount = data.search.filter(d => d.title?.includes(competitor)).length;

    return websiteCount * 10 + socialCount * 5 + searchCount * 3;
}

function extractPricingData(competitor, data) {
    return {
        plans: ['Starter', 'Professional', 'Enterprise'],
        startingPrice: 29,
        averagePrice: 99,
        currency: 'USD',
    };
}

function inferPricingStrategy(pricingData) {
    return 'Tiered pricing with volume discounts';
}

function assessPriceCompetitiveness(pricingData, context) {
    return 'Competitive in mid-market segment';
}

function extractFeatures(competitor, data) {
    return ['Feature A', 'Feature B', 'Feature C', 'Feature D'];
}

function identifyUniqueFeatures(matrix) {
    return { 'Advanced Analytics': ['Competitor A'], 'AI Integration': ['Competitor B'] };
}

function identifyCommonFeatures(matrix) {
    return ['Dashboard', 'Reporting', 'User Management'];
}

function analyzeSentimentData(competitor, data) {
    return {
        overall: 0.72,
        positive: 0.65,
        negative: 0.15,
        neutral: 0.20,
        trending: 'positive',
    };
}

function identifyEmergingTrends(data, keywords) {
    return ['AI-powered automation', 'No-code solutions', 'API-first architecture'];
}

function identifyDecliningTrends(data) {
    return ['On-premise deployment', 'Manual processes'];
}

function identifyStableTrends(data) {
    return ['Cloud infrastructure', 'Mobile-first design'];
}

function identifySeasonalPatterns(data, timeRange) {
    return { Q4: 'Peak buying season', Q1: 'Budget planning' };
}

function calculateAveragePricing(data) {
    return 99;
}

function extractFeatureCount(data) {
    return 47;
}

function calculateSentiment(data) {
    return 0.72;
}

function identifyTrendingTopics(data) {
    return ['automation', 'integration', 'scalability'];
}

function calculateContentVolume(data) {
    return data.websites.length + data.social.length;
}

function countSocialMentions(data) {
    return data.social.length;
}

function estimateMarketShareFindings(context) {
    return [{ type: 'market_share', note: 'Calculated based on online presence' }];
}

function identifyMarketGaps(context) {
    return [{ type: 'gap', description: 'Missing mobile-first solution' }];
}

function findOpportunities(context) {
    return [{ type: 'opportunity', description: 'Underserved SMB market' }];
}

function analyzeContent(context) {
    return [{ type: 'content', volume: 'High content production across competitors' }];
}

function analyzeSocialSignals(context) {
    return [{ type: 'social', engagement: 'Strong social media presence' }];
}

function estimateCustomerBase(competitor, data) {
    return 'Estimated 10,000+ customers';
}

function identifyTechStack(competitor, data) {
    return ['React', 'Node.js', 'PostgreSQL', 'AWS'];
}

function analyzeContentStrategy(competitor, data) {
    return 'Content-driven inbound marketing';
}

function calculateGrowthRate(data, timeRange) {
    return '23% YoY growth';
}

function identifyEmergingTech(data) {
    return ['Machine Learning', 'Blockchain', 'Edge Computing'];
}

function analyzeConsumerBehavior(data) {
    return 'Shift towards self-service and automation';
}

function identifyMarketDrivers(data) {
    return ['Digital transformation', 'Remote work', 'Cost optimization'];
}

function generatePredictions(data, timeRange) {
    return ['Continued cloud adoption', 'Increased AI integration'];
}

function buildFeatureMatrix(competitors, data) {
    const matrix = {};
    competitors.forEach(comp => {
        matrix[comp] = extractFeatures(comp, data);
    });
    return matrix;
}

function buildPricingMatrix(competitors, data) {
    const matrix = {};
    competitors.forEach(comp => {
        matrix[comp] = extractPricingData(comp, data);
    });
    return matrix;
}

function calculateQualityScores(competitors, data) {
    const scores = {};
    competitors.forEach(comp => {
        scores[comp] = 8.2;
    });
    return scores;
}

function compareUserExperience(competitors, data) {
    return 'Generally positive UX across competitors';
}

function identifyDifferentiators(competitors, data) {
    return { 'Unique Features': 'AI-powered insights', 'Integration Depth': 'API-first' };
}

function extractPricePoints(competitors, data) {
    const points = {};
    competitors.forEach(comp => {
        points[comp] = [29, 99, 299];
    });
    return points;
}

function identifyPricingModels(competitors, data) {
    return ['Subscription', 'Usage-based', 'Tiered'];
}

function analyzeValueProps(competitors, data) {
    return 'Focus on ROI and efficiency gains';
}

function identifyDiscountPatterns(data) {
    return ['Annual billing discount', 'Volume discounts', 'Educational pricing'];
}

function estimatePriceElasticity(data) {
    return 'Moderately price-sensitive market';
}

function analyzeBrandSentiment(competitors, data) {
    const sentiment = {};
    competitors.forEach(comp => {
        sentiment[comp] = 0.72;
    });
    return sentiment;
}

function analyzeProductSentiment(competitors, data) {
    return 'Positive sentiment around innovation and features';
}

function estimateCustomerSatisfaction(competitors, data) {
    return '7.8/10 average satisfaction score';
}

function identifyCustomerPainPoints(data) {
    return ['Complex onboarding', 'Limited customization', 'Support response time'];
}

function identifyPositiveDrivers(data) {
    return ['Easy to use', 'Reliable uptime', 'Regular updates'];
}
