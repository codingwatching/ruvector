import { Actor, log } from 'apify';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createRequire } from 'module';
import { integrateActorData, SUPPORTED_ACTORS, USE_CASE_TEMPLATES, getTemplate, listSupportedActors, listTemplates } from './integrations.js';
import { addEmbeddingsToRecords, generateRandomEmbedding, EMBEDDING_MODELS } from './embeddings.js';

// CJS import workaround for RuvLLM native extension
const require = createRequire(import.meta.url);
let ruvllm = null;
let sonaCoordinator = null;
let trajectoryBuilder = null;

try {
  ruvllm = require('@ruvector/ruvllm');
  log.info('RuvLLM loaded successfully - TRM/SONA self-learning enabled');
} catch (e) {
  log.warning(`RuvLLM not available: ${e.message}. Using standard generation.`);
}

// Initialize Actor
await Actor.init();

try {
  // Get input
  const input = await Actor.getInput() || {};

  const {
    // Mode selection
    mode = 'generate',
    // Integration parameters
    integrateActorId,
    integrateRunId = 'latest',
    integrateDatasetId,
    memorizeFields = [],
    useTemplate,
    // Output options
    webhookUrl,
    generateEmbeddings = false,
    // Core parameters
    dataType = 'ecommerce',
    count = 100,
    schema = {},
    timeSeriesConfig = {},
    eventTypes = ['page_view', 'click', 'scroll', 'form_submit'],
    embeddingDimensions = 384,
    provider = 'gemini',
    apiKey,
    model = 'gemini-2.0-flash-exp',
    outputFormat = 'json',
    seed,
    quality = 0.8,
    // Web scraping specific options
    websiteType = 'ecommerce',
    apiEndpoint = '/api/products',
    simulationMode = false,
    batchSize = 100,
    delayBetweenBatches = 0,
    // SONA/TRM parameters
    sonaEnabled = true,
    ewcLambda = 2000,
    patternThreshold = 0.7,
    sonaLearningTiers = ['instant', 'background'],
    // ONNX Embedding parameters
    useOnnxEmbeddings = true,
    embeddingModel = 'all-MiniLM-L6-v2'
  } = input;

  log.info('AI Synthetic Data Generator v2.5 with ONNX Embeddings & TRM/SONA', { mode, dataType, count, provider, model, sonaEnabled, useOnnxEmbeddings, embeddingModel });

  // Initialize SONA if available and enabled
  if (ruvllm && sonaEnabled) {
    try {
      if (ruvllm.SonaCoordinator) {
        sonaCoordinator = new ruvllm.SonaCoordinator({
          tiers: sonaLearningTiers,
          ewcLambda,
          patternThreshold
        });
        log.info('SONA Coordinator initialized', { tiers: sonaLearningTiers, ewcLambda });
      }
      if (ruvllm.TrajectoryBuilder) {
        trajectoryBuilder = new ruvllm.TrajectoryBuilder({
          maxSteps: 100
        });
        log.info('Trajectory Builder initialized');
      }
      // Charge for SONA learning session
      await Actor.charge({ eventName: 'sona-learning-session', count: 1 });
    } catch (e) {
      log.warning(`SONA initialization failed: ${e.message}`);
    }
  }

  // Check for API key based on provider
  const geminiKey = provider === 'gemini' ? (apiKey || process.env.GEMINI_API_KEY) : null;
  const openRouterKey = provider === 'openrouter' ? (apiKey || process.env.OPENROUTER_API_KEY) : null;

  if (provider === 'gemini' && !geminiKey) {
    log.warning('No Gemini API key provided. Using algorithmic generation (still produces great data!)');
  }
  if (provider === 'openrouter' && !openRouterKey) {
    log.warning('No OpenRouter API key provided. Using algorithmic generation.');
  }

  let generatedData = [];
  const startTime = Date.now();

  // ============================================
  // MODE HANDLING: generate, integrate, template
  // ============================================

  if (mode === 'integrate' || mode === 'template') {
    // Integration mode - transform data from other Apify actors
    log.info(`Running in ${mode} mode`, { integrateActorId, useTemplate });

    // Get template config if using template mode
    let templateConfig = null;
    let effectiveActorId = integrateActorId;
    let effectiveMemorizeFields = memorizeFields;

    if (mode === 'template' && useTemplate) {
      templateConfig = getTemplate(useTemplate);
      log.info(`Using template: ${templateConfig.name}`, { suggestedActors: templateConfig.suggestedActors });

      // Use template defaults if not overridden
      if (!effectiveActorId && templateConfig.suggestedActors.length > 0) {
        effectiveActorId = templateConfig.suggestedActors[0];
        log.info(`Using template's suggested actor: ${effectiveActorId}`);
      }
      if (effectiveMemorizeFields.length === 0) {
        effectiveMemorizeFields = templateConfig.memorizeFields || [];
      }

      // Charge for template execution
      await Actor.charge({ eventName: 'template-execution', count: 1 });
    }

    // Fetch data from the actor's dataset
    let sourceData = [];

    if (integrateDatasetId) {
      // Direct dataset access
      log.info(`Fetching from dataset: ${integrateDatasetId}`);
      const dataset = await Actor.openDataset(integrateDatasetId, { forceCloud: true });
      const { items } = await dataset.getData({ limit: count });
      sourceData = items;
    } else if (effectiveActorId) {
      // Fetch from actor run
      log.info(`Fetching from actor: ${effectiveActorId}, run: ${integrateRunId}`);

      try {
        // Use Apify client to fetch last run's dataset
        const client = Actor.newClient();

        let runInfo;
        if (integrateRunId === 'latest') {
          const runs = await client.actor(effectiveActorId).runs().list({ limit: 1 });
          if (runs.items.length === 0) {
            throw new Error(`No runs found for actor ${effectiveActorId}`);
          }
          runInfo = runs.items[0];
        } else {
          runInfo = await client.run(integrateRunId).get();
        }

        if (runInfo && runInfo.defaultDatasetId) {
          const dataset = await client.dataset(runInfo.defaultDatasetId).listItems({ limit: count });
          sourceData = dataset.items;
          log.info(`Fetched ${sourceData.length} items from ${effectiveActorId}`);
        }
      } catch (e) {
        log.error(`Failed to fetch data from ${effectiveActorId}: ${e.message}`);
        log.info('Generating synthetic data as fallback...');
        // Fall back to synthetic data generation
        sourceData = [];
      }
    }

    if (sourceData.length > 0) {
      // Transform the data
      const result = await integrateActorData({
        actorId: effectiveActorId,
        data: sourceData,
        memorizeFields: effectiveMemorizeFields,
        template: useTemplate,
        maxItems: count
      });

      generatedData = result.data;

      // Charge for integration
      await Actor.charge({ eventName: 'actor-integration', count: 1 });
      await Actor.charge({ eventName: 'integrated-record', count: generatedData.length });

      log.info(`Transformed ${generatedData.length} records from ${effectiveActorId}`);
    } else if (mode === 'template' && templateConfig) {
      // Generate synthetic data based on template output format
      log.info('No source data available, generating synthetic data based on template schema...');

      const random = createSeededRandom(seed);
      generatedData = [];

      for (let i = 0; i < count; i++) {
        const record = generateFromTemplateSchema(templateConfig.outputFormat, random, i);
        generatedData.push(record);
      }
    } else {
      throw new Error('No data source specified. Provide integrateActorId or integrateDatasetId.');
    }

  } else {
    // Generate mode - create synthetic data
    // Generate data based on type - optimized for web scraping use cases
    switch (dataType) {
    case 'demo':
      generatedData = await generateDemoData(count, geminiKey, model);
      break;
    case 'ecommerce':
      generatedData = await generateEcommerceData(count, seed);
      break;
    case 'social':
      generatedData = await generateSocialMediaData(count, seed);
      break;
    case 'api_response':
      generatedData = await generateApiResponseData(count, apiEndpoint, seed);
      break;
    case 'search_results':
      generatedData = await generateSearchResultsData(count, seed);
      break;
    case 'real_estate':
      generatedData = await generateRealEstateData(count, seed);
      break;
    case 'jobs':
      generatedData = await generateJobListingsData(count, seed);
      break;
    case 'news':
      generatedData = await generateNewsData(count, seed);
      break;
    case 'structured':
      generatedData = await generateStructuredData(count, schema, geminiKey || openRouterKey, model, seed, provider);
      break;
    case 'timeseries':
      generatedData = await generateTimeSeriesData(count, timeSeriesConfig, seed);
      break;
    case 'events':
      generatedData = await generateEventData(count, eventTypes, seed);
      break;
    case 'embeddings':
      generatedData = await generateEmbeddingData(count, embeddingDimensions, seed);
      break;
    // Enterprise/Company Simulators
    case 'stock_trading':
      generatedData = await generateStockTradingData(count, seed);
      break;
    case 'medical':
      generatedData = await generateMedicalData(count, seed);
      break;
    case 'company':
      generatedData = await generateCompanyData(count, seed);
      break;
    case 'supply_chain':
      generatedData = await generateSupplyChainData(count, seed);
      break;
    case 'financial':
      generatedData = await generateFinancialData(count, seed);
      break;
    case 'bloomberg':
      generatedData = await generateBloombergData(count, seed);
      break;
    default:
      throw new Error(`Unknown data type: ${dataType}. Available: ecommerce, social, api_response, search_results, real_estate, jobs, news, structured, timeseries, events, embeddings, stock_trading, medical, company, supply_chain, financial, bloomberg, demo`);
    }
  } // End of generate mode else block

  const generationTime = Date.now() - startTime;

  // ============================================
  // EMBEDDING GENERATION (optional)
  // ============================================
  if (generateEmbeddings && generatedData.length > 0) {
    const modelConfig = EMBEDDING_MODELS[embeddingModel] || EMBEDDING_MODELS['all-MiniLM-L6-v2'];
    const effectiveDimensions = useOnnxEmbeddings ? modelConfig.dimensions : embeddingDimensions;

    log.info(`Generating embeddings with ${effectiveDimensions} dimensions...`, {
      useOnnx: useOnnxEmbeddings,
      model: useOnnxEmbeddings ? embeddingModel : 'random'
    });

    if (useOnnxEmbeddings) {
      // Use ONNX-powered semantic embeddings
      try {
        generatedData = await addEmbeddingsToRecords(generatedData, { modelName: embeddingModel });
        log.info(`Added ONNX embeddings using ${embeddingModel} model`);
        await Actor.charge({ eventName: 'onnx-embedding-generation', count: generatedData.length });
      } catch (e) {
        log.warning(`ONNX embedding failed: ${e.message}. Falling back to random embeddings.`);
        // Fall back to random embeddings
        const random = createSeededRandom(seed);
        generatedData = generatedData.map((item) => ({
          ...item,
          embedding: generateRandomEmbedding(effectiveDimensions, random),
          embeddingModel: 'random',
          embeddingDimensions: effectiveDimensions
        }));
      }
    } else {
      // Use random embeddings (faster, for testing)
      const random = createSeededRandom(seed);
      generatedData = generatedData.map((item) => ({
        ...item,
        embedding: generateRandomEmbedding(effectiveDimensions, random),
        embeddingModel: 'random',
        embeddingDimensions: effectiveDimensions
      }));
    }

    // Charge for embedding generation
    await Actor.charge({ eventName: 'embedding-generation', count: generatedData.length });
    log.info(`Added embeddings to ${generatedData.length} records`);
  }

  // Track generation trajectory for SONA learning
  if (trajectoryBuilder && sonaEnabled) {
    try {
      // Use correct TrajectoryBuilder API: startStep -> endStep -> complete
      const stepId = trajectoryBuilder.startStep('generate', {
        dataType,
        count: generatedData.length,
        quality,
        seed: seed || 'random'
      });
      trajectoryBuilder.endStep(stepId, {
        duration: generationTime,
        success: true,
        recordsGenerated: generatedData.length
      });
      log.info('Generation trajectory tracked for SONA learning');
    } catch (e) {
      log.warning(`Trajectory tracking failed: ${e.message}`);
    }
  }

  // SONA pattern learning from generated data
  if (sonaCoordinator && sonaEnabled && generatedData.length > 0) {
    try {
      const sampleSize = Math.min(10, generatedData.length);
      const sample = generatedData.slice(0, sampleSize);

      // Use correct SonaCoordinator API: recordSignal for instant learning
      sonaCoordinator.recordSignal({
        type: 'generation_complete',
        dataType,
        samples: sample,
        quality,
        generationTime,
        count: generatedData.length
      });

      // Process instant learning tier
      if (sonaLearningTiers.includes('instant')) {
        await sonaCoordinator.processInstantLearning();
      }

      log.info(`SONA recorded signal from ${sampleSize} samples`, {
        stats: sonaCoordinator.stats()
      });
    } catch (e) {
      log.warning(`SONA pattern learning failed: ${e.message}`);
    }
  }

  log.info(`Generated ${generatedData.length} records in ${generationTime}ms`);

  // Charge custom events based on data type
  const eventMap = {
    'ecommerce': 'ecommerce-product',
    'social': 'social-media-post',
    'jobs': 'job-listing',
    'real_estate': 'real-estate-listing',
    'search_results': 'search-result',
    'api_response': 'api-mock-response',
    'news': 'news-article',
    // Enterprise data types
    'stock_trading': 'stock-trading-record',
    'medical': 'medical-record',
    'company': 'company-record',
    'supply_chain': 'supply-chain-record',
    'financial': 'financial-record',
    'bloomberg': 'bloomberg-terminal-record'
  };

  // Simulation mode - push in batches with delays
  if (simulationMode && delayBetweenBatches > 0) {
    log.info(`Simulation mode: pushing ${batchSize} records every ${delayBetweenBatches}ms`);

    // Charge for simulation session
    await Actor.charge({ eventName: 'simulation-session', count: 1 });

    const totalBatches = Math.ceil(generatedData.length / batchSize);

    for (let i = 0; i < generatedData.length; i += batchSize) {
      const batch = generatedData.slice(i, i + batchSize);
      const batchNum = Math.floor(i / batchSize) + 1;

      await Actor.pushData(batch.map((item, idx) => ({
        id: i + idx + 1,
        type: dataType,
        data: item,
        metadata: {
          generatedAt: new Date().toISOString(),
          provider,
          model,
          quality,
          seed: seed || 'random',
          batch: batchNum,
          totalBatches,
          simulationMode: true
        }
      })));

      // Charge for simulation batch
      await Actor.charge({ eventName: 'simulation-batch', count: 1 });

      log.info(`Pushed batch ${batchNum}/${totalBatches}`);

      if (i + batchSize < generatedData.length) {
        await new Promise(resolve => setTimeout(resolve, delayBetweenBatches));
      }
    }
  } else {
    // Push all results at once
    await Actor.pushData(generatedData.map((item, index) => ({
      id: index + 1,
      type: mode === 'generate' ? dataType : mode,
      data: item,
      metadata: {
        generatedAt: new Date().toISOString(),
        mode,
        dataType: mode === 'generate' ? dataType : null,
        actorId: integrateActorId || null,
        template: useTemplate || null,
        provider,
        model,
        quality,
        seed: seed || 'random',
        hasEmbedding: generateEmbeddings
      }
    })));

    // Charge for data type specific events
    const eventName = eventMap[dataType];
    if (eventName && mode === 'generate') {
      await Actor.charge({ eventName, count: generatedData.length });
      log.info(`Charged ${generatedData.length} ${eventName} events`);
    }

    // Charge for AI-enhanced records if using AI
    if ((geminiKey || openRouterKey) && dataType === 'structured') {
      await Actor.charge({ eventName: 'ai-enhanced-record', count: generatedData.length });
      log.info(`Charged ${generatedData.length} AI-enhanced events`);
    }
  }

  log.info(`Pushed ${generatedData.length} records to dataset`);

  // ============================================
  // WEBHOOK NOTIFICATION (optional)
  // ============================================
  if (webhookUrl) {
    log.info(`Sending webhook to: ${webhookUrl}`);

    try {
      const webhookPayload = {
        actorId: 'ruv/ai-synthetic-data-generator',
        runId: process.env.APIFY_ACTOR_RUN_ID,
        status: 'success',
        mode,
        dataType: mode === 'generate' ? dataType : null,
        template: useTemplate || null,
        integrateActorId: integrateActorId || null,
        totalRecords: generatedData.length,
        generationTime,
        hasEmbeddings: generateEmbeddings,
        datasetId: process.env.APIFY_DEFAULT_DATASET_ID,
        timestamp: new Date().toISOString()
      };

      const response = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'Apify-AI-Synthetic-Data-Generator/2.5'
        },
        body: JSON.stringify(webhookPayload)
      });

      if (response.ok) {
        log.info('Webhook notification sent successfully');
        await Actor.charge({ eventName: 'webhook-notification', count: 1 });
      } else {
        log.warning(`Webhook failed with status: ${response.status}`);
      }
    } catch (e) {
      log.warning(`Webhook notification failed: ${e.message}`);
    }
  }

} catch (error) {
  log.error('Actor failed', { error: error.message });
  throw error;
} finally {
  await Actor.exit();
}

// ============================================
// WEB SCRAPING FOCUSED GENERATORS
// ============================================

async function generateEcommerceData(count, seed) {
  log.info('Generating e-commerce product data...');
  const random = createSeededRandom(seed);
  const results = [];

  const categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys', 'Beauty', 'Automotive'];
  const brands = ['TechPro', 'StyleMax', 'HomeEssentials', 'SportZone', 'BookWorld', 'KidsFun', 'GlowUp', 'AutoParts'];
  const conditions = ['New', 'Used - Like New', 'Used - Good', 'Refurbished'];

  for (let i = 0; i < count; i++) {
    const category = categories[Math.floor(random() * categories.length)];
    const brand = brands[Math.floor(random() * brands.length)];
    const basePrice = 10 + random() * 990;
    const hasDiscount = random() > 0.6;

    results.push({
      url: `https://example-store.com/products/${generateSlug(random)}-${i}`,
      title: `${brand} ${generateProductName(category, random)}`,
      price: Math.round(basePrice * 100) / 100,
      originalPrice: hasDiscount ? Math.round(basePrice * (1.1 + random() * 0.4) * 100) / 100 : null,
      currency: 'USD',
      category,
      brand,
      rating: Math.round((3 + random() * 2) * 10) / 10,
      reviewCount: Math.floor(random() * 5000),
      inStock: random() > 0.15,
      stockCount: Math.floor(random() * 500),
      condition: conditions[Math.floor(random() * conditions.length)],
      seller: {
        name: `Seller${Math.floor(random() * 1000)}`,
        rating: Math.round((3.5 + random() * 1.5) * 10) / 10,
        totalSales: Math.floor(random() * 50000)
      },
      shipping: {
        free: random() > 0.4,
        estimatedDays: Math.floor(2 + random() * 8),
        price: random() > 0.4 ? 0 : Math.round(random() * 15 * 100) / 100
      },
      images: Array.from({ length: Math.floor(1 + random() * 5) }, (_, j) =>
        `https://example-store.com/images/product-${i}-${j}.jpg`
      ),
      specifications: generateSpecs(category, random),
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateSocialMediaData(count, seed) {
  log.info('Generating social media data...');
  const random = createSeededRandom(seed);
  const results = [];

  const platforms = ['twitter', 'instagram', 'facebook', 'linkedin', 'tiktok'];
  const postTypes = ['text', 'image', 'video', 'link', 'poll'];

  for (let i = 0; i < count; i++) {
    const platform = platforms[Math.floor(random() * platforms.length)];
    const postType = postTypes[Math.floor(random() * postTypes.length)];
    const timestamp = new Date(Date.now() - random() * 30 * 24 * 60 * 60 * 1000);

    results.push({
      url: `https://${platform}.com/post/${generateId(random)}`,
      platform,
      postType,
      author: {
        username: `user_${generateId(random)}`,
        displayName: generateName(random),
        verified: random() > 0.85,
        followers: Math.floor(random() * 1000000),
        following: Math.floor(random() * 5000),
        profileUrl: `https://${platform}.com/user_${generateId(random)}`
      },
      content: {
        text: generateSocialText(random),
        hashtags: Array.from({ length: Math.floor(random() * 6) }, () => `#${generateHashtag(random)}`),
        mentions: Array.from({ length: Math.floor(random() * 3) }, () => `@user_${generateId(random)}`),
        mediaUrls: postType !== 'text' ? [`https://${platform}.com/media/${generateId(random)}.jpg`] : []
      },
      engagement: {
        likes: Math.floor(random() * 100000),
        comments: Math.floor(random() * 5000),
        shares: Math.floor(random() * 10000),
        views: Math.floor(random() * 1000000)
      },
      timestamp: timestamp.toISOString(),
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateApiResponseData(count, endpoint, seed) {
  log.info('Generating API response data...', { endpoint });
  const random = createSeededRandom(seed);
  const results = [];

  for (let i = 0; i < count; i++) {
    const statusCodes = [200, 200, 200, 200, 201, 400, 401, 404, 500];
    const statusCode = statusCodes[Math.floor(random() * statusCodes.length)];

    results.push({
      endpoint: `${endpoint}/${i}`,
      method: 'GET',
      statusCode,
      headers: {
        'content-type': 'application/json',
        'x-request-id': generateId(random),
        'x-rate-limit-remaining': Math.floor(random() * 1000),
        'cache-control': random() > 0.5 ? 'max-age=3600' : 'no-cache'
      },
      responseTime: Math.floor(50 + random() * 500),
      body: statusCode < 400 ? {
        id: generateId(random),
        data: generateRandomObject(random),
        pagination: {
          page: 1,
          perPage: 20,
          total: Math.floor(random() * 10000),
          hasMore: random() > 0.3
        }
      } : {
        error: {
          code: `ERR_${statusCode}`,
          message: getErrorMessage(statusCode)
        }
      },
      timestamp: new Date().toISOString()
    });
  }

  return results;
}

async function generateSearchResultsData(count, seed) {
  log.info('Generating search results data...');
  const random = createSeededRandom(seed);
  const results = [];

  const domains = ['example.com', 'blog.example.org', 'news.example.net', 'shop.example.io', 'docs.example.dev'];

  for (let i = 0; i < count; i++) {
    const domain = domains[Math.floor(random() * domains.length)];

    results.push({
      position: i + 1,
      url: `https://${domain}/${generateSlug(random)}`,
      title: generateSearchTitle(random),
      snippet: generateSnippet(random),
      domain,
      displayUrl: `${domain} > ${generateBreadcrumb(random)}`,
      type: random() > 0.8 ? 'featured' : 'organic',
      sitelinks: random() > 0.7 ? Array.from({ length: Math.floor(2 + random() * 4) }, () => ({
        title: generateSearchTitle(random),
        url: `https://${domain}/${generateSlug(random)}`
      })) : null,
      rich_snippet: random() > 0.6 ? {
        rating: Math.round((3 + random() * 2) * 10) / 10,
        reviewCount: Math.floor(random() * 10000),
        price: random() > 0.5 ? `$${Math.floor(10 + random() * 500)}` : null
      } : null,
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateRealEstateData(count, seed) {
  log.info('Generating real estate listing data...');
  const random = createSeededRandom(seed);
  const results = [];

  const propertyTypes = ['House', 'Apartment', 'Condo', 'Townhouse', 'Land', 'Commercial'];
  const cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'San Diego', 'Dallas', 'Austin'];
  const listingTypes = ['For Sale', 'For Rent', 'Auction'];

  for (let i = 0; i < count; i++) {
    const propertyType = propertyTypes[Math.floor(random() * propertyTypes.length)];
    const city = cities[Math.floor(random() * cities.length)];
    const listingType = listingTypes[Math.floor(random() * listingTypes.length)];
    const bedrooms = Math.floor(1 + random() * 6);
    const sqft = Math.floor(500 + random() * 4500);

    results.push({
      url: `https://realestate-example.com/listing/${generateId(random)}`,
      listingId: generateId(random),
      title: `${bedrooms} Bed ${propertyType} in ${city}`,
      price: Math.floor(100000 + random() * 2000000),
      listingType,
      propertyType,
      address: {
        street: `${Math.floor(100 + random() * 9900)} ${generateStreetName(random)}`,
        city,
        state: getState(city),
        zipCode: String(Math.floor(10000 + random() * 90000)),
        country: 'USA'
      },
      details: {
        bedrooms,
        bathrooms: Math.floor(1 + random() * 4),
        sqft,
        lotSize: Math.floor(sqft * (1.5 + random() * 3)),
        yearBuilt: Math.floor(1950 + random() * 74),
        parking: Math.floor(random() * 4),
        stories: Math.floor(1 + random() * 3)
      },
      features: generateRealEstateFeatures(random),
      agent: {
        name: generateName(random),
        phone: generatePhone(random),
        email: `agent${Math.floor(random() * 1000)}@realestate.com`,
        company: `${generateName(random)} Realty`
      },
      images: Array.from({ length: Math.floor(5 + random() * 20) }, (_, j) =>
        `https://realestate-example.com/images/listing-${i}-${j}.jpg`
      ),
      daysOnMarket: Math.floor(random() * 180),
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateJobListingsData(count, seed) {
  log.info('Generating job listings data...');
  const random = createSeededRandom(seed);
  const results = [];

  const titles = ['Software Engineer', 'Product Manager', 'Data Scientist', 'UX Designer', 'DevOps Engineer', 'Marketing Manager', 'Sales Representative', 'Customer Success Manager'];
  const companies = ['TechCorp', 'InnovateLabs', 'DataDriven Inc', 'CloudScale', 'StartupXYZ', 'Enterprise Solutions', 'Digital Agency', 'Growth Partners'];
  const locations = ['Remote', 'New York, NY', 'San Francisco, CA', 'Austin, TX', 'Seattle, WA', 'Boston, MA', 'Chicago, IL', 'Los Angeles, CA'];
  const types = ['Full-time', 'Part-time', 'Contract', 'Internship'];

  for (let i = 0; i < count; i++) {
    const title = titles[Math.floor(random() * titles.length)];
    const company = companies[Math.floor(random() * companies.length)];
    const location = locations[Math.floor(random() * locations.length)];
    const salaryMin = Math.floor(50000 + random() * 100000);

    results.push({
      url: `https://jobs-example.com/job/${generateId(random)}`,
      jobId: generateId(random),
      title,
      company: {
        name: company,
        logo: `https://jobs-example.com/logos/${company.toLowerCase().replace(/\s/g, '-')}.png`,
        rating: Math.round((3 + random() * 2) * 10) / 10,
        reviewCount: Math.floor(random() * 5000),
        size: ['1-50', '51-200', '201-500', '501-1000', '1000+'][Math.floor(random() * 5)]
      },
      location,
      remote: location === 'Remote' || random() > 0.7,
      type: types[Math.floor(random() * types.length)],
      salary: {
        min: salaryMin,
        max: salaryMin + Math.floor(random() * 50000),
        currency: 'USD',
        period: 'yearly'
      },
      description: generateJobDescription(random),
      requirements: Array.from({ length: Math.floor(3 + random() * 5) }, () => generateRequirement(random)),
      benefits: generateBenefits(random),
      postedDate: new Date(Date.now() - random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      applicants: Math.floor(random() * 500),
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateNewsData(count, seed) {
  log.info('Generating news article data...');
  const random = createSeededRandom(seed);
  const results = [];

  const sources = ['TechNews', 'BusinessDaily', 'WorldReport', 'ScienceToday', 'HealthWatch', 'SportsCentral'];
  const categories = ['Technology', 'Business', 'Politics', 'Science', 'Health', 'Sports', 'Entertainment'];
  const authors = ['John Smith', 'Sarah Johnson', 'Mike Williams', 'Emily Brown', 'David Lee', 'Lisa Chen'];

  for (let i = 0; i < count; i++) {
    const source = sources[Math.floor(random() * sources.length)];
    const category = categories[Math.floor(random() * categories.length)];
    const publishDate = new Date(Date.now() - random() * 7 * 24 * 60 * 60 * 1000);

    results.push({
      url: `https://${source.toLowerCase()}.com/article/${generateSlug(random)}`,
      title: generateNewsTitle(category, random),
      subtitle: generateSubtitle(random),
      source,
      category,
      author: {
        name: authors[Math.floor(random() * authors.length)],
        url: `https://${source.toLowerCase()}.com/author/${generateSlug(random)}`
      },
      publishedAt: publishDate.toISOString(),
      updatedAt: random() > 0.7 ? new Date(publishDate.getTime() + random() * 24 * 60 * 60 * 1000).toISOString() : null,
      content: {
        text: generateArticleContent(random),
        wordCount: Math.floor(300 + random() * 1500),
        readingTime: Math.floor(2 + random() * 10)
      },
      images: [{
        url: `https://${source.toLowerCase()}.com/images/article-${i}.jpg`,
        caption: generateCaption(random)
      }],
      tags: Array.from({ length: Math.floor(2 + random() * 5) }, () => generateTag(random)),
      engagement: {
        views: Math.floor(random() * 100000),
        comments: Math.floor(random() * 500),
        shares: Math.floor(random() * 2000)
      },
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

// ============================================
// ORIGINAL GENERATORS (kept for compatibility)
// ============================================

async function generateDemoData(count, apiKey, model) {
  log.info('Generating demo data with web scraping examples...');

  const results = [];
  const perType = Math.ceil(count / 5);

  // E-commerce products
  const ecommerce = await generateEcommerceData(perType);
  results.push(...ecommerce.map(d => ({ ...d, _type: 'ecommerce' })));

  // Social media posts
  const social = await generateSocialMediaData(perType);
  results.push(...social.map(d => ({ ...d, _type: 'social' })));

  // Search results
  const search = await generateSearchResultsData(perType);
  results.push(...search.map(d => ({ ...d, _type: 'search_results' })));

  // Job listings
  const jobs = await generateJobListingsData(perType);
  results.push(...jobs.map(d => ({ ...d, _type: 'jobs' })));

  // News articles
  const news = await generateNewsData(perType);
  results.push(...news.map(d => ({ ...d, _type: 'news' })));

  return results.slice(0, count);
}

async function generateStructuredData(count, schema, apiKey, model, seed, provider = 'gemini') {
  log.info('Generating structured data...', { count, schema, provider, model });

  const results = [];
  const random = createSeededRandom(seed);

  if (apiKey && Object.keys(schema).length > 0) {
    try {
      const prompt = `Generate ${Math.min(count, 20)} unique records matching this schema:
${JSON.stringify(schema, null, 2)}

Return ONLY a valid JSON array with no additional text. Each record should be realistic and diverse.`;

      let text;

      if (provider === 'openrouter') {
        // Use OpenRouter API (supports GPT, Claude, etc.)
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://apify.com',
            'X-Title': 'AI Synthetic Data Generator'
          },
          body: JSON.stringify({
            model: model || 'openai/gpt-3.5-turbo',
            messages: [{ role: 'user', content: prompt }],
            temperature: 0.7
          })
        });
        const data = await response.json();
        text = data.choices?.[0]?.message?.content || '';
        log.info('OpenRouter response received', { model });
      } else {
        // Use Gemini
        const genAI = new GoogleGenerativeAI(apiKey);
        const gemini = genAI.getGenerativeModel({ model: model || 'gemini-2.0-flash-exp' });
        const result = await gemini.generateContent(prompt);
        text = result.response.text();
        log.info('Gemini response received', { model });
      }

      const jsonMatch = text.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        results.push(...parsed);
        log.info(`AI generated ${parsed.length} records`);
      }

      while (results.length < count) {
        results.push(generateFallbackStructured(schema, random));
      }

    } catch (e) {
      log.warning(`AI generation failed: ${e.message}. Using fallback.`);
      for (let i = 0; i < count; i++) {
        results.push(generateFallbackStructured(schema, random));
      }
    }
  } else {
    for (let i = 0; i < count; i++) {
      results.push(generateFallbackStructured(schema, random));
    }
  }

  return results.slice(0, count);
}

function generateFallbackStructured(schema, random) {
  const record = {};

  for (const [key, type] of Object.entries(schema)) {
    if (typeof type === 'string') {
      if (type.includes('url')) {
        record[key] = `https://example.com/${generateSlug(random)}`;
      } else if (type.includes('email')) {
        record[key] = `user${Math.floor(random() * 10000)}@example.com`;
      } else if (type.includes('fullName') || type.includes('name')) {
        record[key] = generateName(random);
      } else if (type.includes('number')) {
        const match = type.match(/\((\d+)-(\d+)\)/);
        if (match) {
          const min = parseInt(match[1]);
          const max = parseInt(match[2]);
          record[key] = min + Math.floor(random() * (max - min + 1));
        } else {
          record[key] = Math.floor(random() * 100);
        }
      } else if (type.includes('boolean')) {
        record[key] = random() > 0.5;
      } else if (type.includes('(') && type.includes(',')) {
        const options = type.match(/\(([^)]+)\)/)?.[1].split(',').map(s => s.trim()) || ['Option1', 'Option2'];
        record[key] = options[Math.floor(random() * options.length)];
      } else {
        record[key] = `value_${Math.floor(random() * 1000)}`;
      }
    }
  }

  return record;
}

async function generateTimeSeriesData(count, config, seed) {
  log.info('Generating time-series data...', { count, config });

  const {
    interval = '1h',
    trend = 'flat',
    seasonality = false,
    noise = 0.1,
    startDate = '2024-01-01'
  } = config;

  const random = createSeededRandom(seed);
  const results = [];

  const start = new Date(startDate);
  const intervalMs = parseInterval(interval);

  let value = 100;
  const trendFactor = trend === 'upward' ? 0.01 : trend === 'downward' ? -0.01 : 0;

  for (let i = 0; i < count; i++) {
    const timestamp = new Date(start.getTime() + i * intervalMs);

    value *= (1 + trendFactor);

    let seasonalValue = value;
    if (seasonality) {
      const hour = timestamp.getHours();
      const seasonalFactor = 1 + 0.2 * Math.sin((hour / 24) * 2 * Math.PI);
      seasonalValue = value * seasonalFactor;
    }

    const noiseValue = seasonalValue * (1 + (random() - 0.5) * 2 * noise);

    results.push({
      timestamp: timestamp.toISOString(),
      value: Math.round(noiseValue * 100) / 100,
      open: Math.round(noiseValue * (1 - random() * 0.02) * 100) / 100,
      high: Math.round(noiseValue * (1 + random() * 0.03) * 100) / 100,
      low: Math.round(noiseValue * (1 - random() * 0.03) * 100) / 100,
      close: Math.round(noiseValue * (1 + (random() - 0.5) * 0.02) * 100) / 100,
      volume: Math.floor(random() * 1000000)
    });
  }

  return results;
}

async function generateEventData(count, eventTypes, seed) {
  log.info('Generating web event data...', { count, eventTypes });

  const random = createSeededRandom(seed);
  const results = [];

  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;

  for (let i = 0; i < count; i++) {
    const eventType = eventTypes[Math.floor(random() * eventTypes.length)];
    const timestamp = new Date(now - random() * 30 * dayMs);

    const event = {
      eventId: `evt_${Date.now()}_${i}`,
      type: eventType,
      timestamp: timestamp.toISOString(),
      userId: `user_${Math.floor(random() * 1000)}`,
      sessionId: `sess_${Math.floor(random() * 10000)}`,
      page: {
        url: `https://example.com/${generateSlug(random)}`,
        title: generateSearchTitle(random),
        referrer: random() > 0.3 ? 'https://google.com' : 'direct'
      },
      device: {
        type: random() > 0.6 ? 'mobile' : 'desktop',
        browser: ['Chrome', 'Firefox', 'Safari', 'Edge'][Math.floor(random() * 4)],
        os: ['Windows', 'macOS', 'iOS', 'Android', 'Linux'][Math.floor(random() * 5)]
      },
      properties: generateEventProperties(eventType, random)
    };

    results.push(event);
  }

  results.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  return results;
}

function generateEventProperties(eventType, random) {
  switch (eventType) {
    case 'page_view':
      return {
        loadTime: Math.floor(100 + random() * 3000),
        scrollDepth: Math.floor(random() * 100)
      };
    case 'click':
      return {
        element: ['button', 'link', 'image', 'card'][Math.floor(random() * 4)],
        elementId: `el_${Math.floor(random() * 1000)}`,
        x: Math.floor(random() * 1920),
        y: Math.floor(random() * 1080)
      };
    case 'scroll':
      return {
        direction: random() > 0.8 ? 'up' : 'down',
        depth: Math.floor(random() * 100),
        velocity: Math.floor(random() * 500)
      };
    case 'form_submit':
      return {
        formId: `form_${Math.floor(random() * 100)}`,
        formName: ['contact', 'signup', 'checkout', 'search'][Math.floor(random() * 4)],
        success: random() > 0.1,
        fieldCount: Math.floor(2 + random() * 10)
      };
    case 'api_call':
      return {
        endpoint: `/api/${['users', 'products', 'orders', 'search'][Math.floor(random() * 4)]}`,
        method: ['GET', 'POST', 'PUT', 'DELETE'][Math.floor(random() * 4)],
        statusCode: random() > 0.9 ? 500 : random() > 0.1 ? 200 : 400,
        responseTime: Math.floor(50 + random() * 500)
      };
    default:
      return { value: Math.floor(random() * 100) };
  }
}

async function generateEmbeddingData(count, dimensions, seed) {
  log.info('Generating embedding data...', { count, dimensions });

  const random = createSeededRandom(seed);
  const results = [];

  const topics = [
    'Product search optimization',
    'Customer sentiment analysis',
    'Price comparison algorithms',
    'Inventory management',
    'User behavior tracking',
    'Market trend analysis',
    'Competitor monitoring',
    'Review aggregation',
    'Category classification',
    'Recommendation engines'
  ];

  for (let i = 0; i < count; i++) {
    const embedding = [];
    let norm = 0;

    for (let j = 0; j < dimensions; j++) {
      const val = random() * 2 - 1;
      embedding.push(val);
      norm += val * val;
    }

    norm = Math.sqrt(norm);
    for (let j = 0; j < dimensions; j++) {
      embedding[j] = Math.round((embedding[j] / norm) * 1000000) / 1000000;
    }

    results.push({
      id: `emb_${i}`,
      text: topics[i % topics.length] + ` - variant ${Math.floor(i / topics.length)}`,
      embedding,
      dimensions,
      model: 'synthetic'
    });
  }

  return results;
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function createSeededRandom(seed) {
  if (!seed) return Math.random;

  let s = hashCode(String(seed));
  return function() {
    s = Math.sin(s) * 10000;
    return s - Math.floor(s);
  };
}

function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

/**
 * Generate synthetic data based on a template output schema
 */
function generateFromTemplateSchema(outputFormat, random, index) {
  const record = {};

  for (const [key, type] of Object.entries(outputFormat)) {
    if (type === 'string') {
      record[key] = generateTemplateString(key, random);
    } else if (type.startsWith('number')) {
      const match = type.match(/\((\d+)-(\d+)\)/);
      if (match) {
        const min = parseInt(match[1]);
        const max = parseInt(match[2]);
        record[key] = min + Math.floor(random() * (max - min + 1));
      } else {
        record[key] = Math.floor(random() * 100);
      }
    } else if (type === 'object') {
      record[key] = { id: generateId(random), value: Math.floor(random() * 1000) };
    } else if (type.startsWith('array')) {
      const itemType = type.match(/<(\w+)>/)?.[1] || 'string';
      const count = Math.floor(2 + random() * 4);
      record[key] = Array.from({ length: count }, () =>
        itemType === 'string' ? generateTemplateString(key, random) :
        itemType === 'object' ? { id: generateId(random), value: Math.floor(random() * 100) } :
        Math.floor(random() * 1000)
      );
    } else {
      record[key] = `value_${index}_${Math.floor(random() * 1000)}`;
    }
  }

  record._templateId = `tpl_${index}`;
  record._generatedAt = new Date().toISOString();

  return record;
}

/**
 * Generate context-aware string values based on field name
 */
function generateTemplateString(fieldName, random) {
  const lowerField = fieldName.toLowerCase();

  if (lowerField.includes('id') || lowerField.includes('Id')) {
    return `id_${Math.floor(random() * 100000)}`;
  }
  if (lowerField.includes('name') || lowerField.includes('title')) {
    return generateName(random);
  }
  if (lowerField.includes('email')) {
    return `user${Math.floor(random() * 10000)}@example.com`;
  }
  if (lowerField.includes('phone')) {
    return generatePhone(random);
  }
  if (lowerField.includes('url') || lowerField.includes('website')) {
    return `https://example.com/${generateSlug(random)}`;
  }
  if (lowerField.includes('description') || lowerField.includes('content') || lowerField.includes('summary')) {
    return generateSnippet(random);
  }
  if (lowerField.includes('approach') || lowerField.includes('strategy')) {
    const approaches = ['Direct outreach', 'Email campaign', 'Social engagement', 'Referral network', 'Content marketing'];
    return approaches[Math.floor(random() * approaches.length)];
  }
  if (lowerField.includes('insight') || lowerField.includes('finding')) {
    const insights = ['High growth potential', 'Active buyer signals', 'Recent funding round', 'Expanding market', 'Technology adoption'];
    return insights[Math.floor(random() * insights.length)];
  }
  if (lowerField.includes('style') || lowerField.includes('type')) {
    const styles = ['Professional', 'Casual', 'Educational', 'Entertaining', 'Promotional'];
    return styles[Math.floor(random() * styles.length)];
  }

  return `value_${Math.floor(random() * 1000)}`;
}

function parseInterval(interval) {
  const match = interval.match(/(\d+)([mhd])/);
  if (!match) return 3600000;

  const value = parseInt(match[1]);
  const unit = match[2];

  switch (unit) {
    case 'm': return value * 60 * 1000;
    case 'h': return value * 60 * 60 * 1000;
    case 'd': return value * 24 * 60 * 60 * 1000;
    default: return 3600000;
  }
}

function generateId(random) {
  return Math.random().toString(36).substring(2, 15);
}

function generateSlug(random) {
  const words = ['best', 'top', 'new', 'amazing', 'premium', 'ultra', 'pro', 'max', 'elite', 'smart'];
  const nouns = ['product', 'item', 'deal', 'offer', 'guide', 'review', 'article', 'post'];
  return `${words[Math.floor(random() * words.length)]}-${nouns[Math.floor(random() * nouns.length)]}-${Math.floor(random() * 10000)}`;
}

function generateName(random) {
  const firstNames = ['John', 'Jane', 'Alex', 'Sarah', 'Mike', 'Emma', 'Chris', 'Lisa', 'David', 'Amy'];
  const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Wilson'];
  return `${firstNames[Math.floor(random() * firstNames.length)]} ${lastNames[Math.floor(random() * lastNames.length)]}`;
}

function generateProductName(category, random) {
  const adjectives = ['Premium', 'Ultra', 'Pro', 'Classic', 'Smart', 'Portable', 'Wireless', 'Advanced'];
  const products = {
    'Electronics': ['Headphones', 'Speaker', 'Charger', 'Cable', 'Adapter', 'Mouse', 'Keyboard'],
    'Clothing': ['T-Shirt', 'Jacket', 'Jeans', 'Sneakers', 'Hat', 'Sweater', 'Dress'],
    'Home & Garden': ['Lamp', 'Planter', 'Organizer', 'Tool Set', 'Decoration', 'Rug'],
    'Sports': ['Ball', 'Gloves', 'Bag', 'Mat', 'Weights', 'Bottle', 'Band'],
    'Books': ['Guide', 'Novel', 'Textbook', 'Cookbook', 'Biography', 'Manual'],
    'Toys': ['Figure', 'Game', 'Puzzle', 'Set', 'Doll', 'Car'],
    'Beauty': ['Cream', 'Serum', 'Mask', 'Oil', 'Brush', 'Palette'],
    'Automotive': ['Cover', 'Mat', 'Charger', 'Holder', 'Cleaner', 'Light']
  };
  const items = products[category] || products['Electronics'];
  return `${adjectives[Math.floor(random() * adjectives.length)]} ${items[Math.floor(random() * items.length)]}`;
}

function generateSpecs(category, random) {
  const specs = {
    'Electronics': { battery: `${Math.floor(1000 + random() * 4000)}mAh`, connectivity: 'Bluetooth 5.0', warranty: '1 year' },
    'Clothing': { material: random() > 0.5 ? 'Cotton' : 'Polyester', size: ['S', 'M', 'L', 'XL'][Math.floor(random() * 4)] },
    'Home & Garden': { dimensions: `${Math.floor(10 + random() * 50)}x${Math.floor(10 + random() * 50)}cm`, weight: `${Math.floor(random() * 10)}kg` }
  };
  return specs[category] || { general: 'Standard specifications' };
}

function generateSocialText(random) {
  const texts = [
    'Just discovered this amazing product! Highly recommend',
    'Working on something exciting today',
    'Can\'t believe how good this turned out',
    'Who else is enjoying this beautiful day?',
    'Sharing my latest project with you all',
    'This is a game changer for productivity',
    'Thoughts on the latest industry trends?'
  ];
  return texts[Math.floor(random() * texts.length)];
}

function generateHashtag(random) {
  const tags = ['tech', 'innovation', 'business', 'startup', 'coding', 'design', 'marketing', 'growth', 'success', 'tips'];
  return tags[Math.floor(random() * tags.length)];
}

function generateRandomObject(random) {
  return {
    name: generateName(random),
    value: Math.floor(random() * 1000),
    active: random() > 0.3,
    tags: ['tag1', 'tag2', 'tag3'].slice(0, Math.floor(1 + random() * 3))
  };
}

function getErrorMessage(code) {
  const messages = {
    400: 'Bad Request - Invalid parameters',
    401: 'Unauthorized - Invalid API key',
    403: 'Forbidden - Access denied',
    404: 'Not Found - Resource does not exist',
    500: 'Internal Server Error'
  };
  return messages[code] || 'Unknown error';
}

function generateSearchTitle(random) {
  const templates = [
    'How to Get Started with {topic}',
    'The Complete Guide to {topic}',
    'Top 10 {topic} Tips for Beginners',
    'Best {topic} Practices in 2024',
    '{topic}: Everything You Need to Know'
  ];
  const topics = ['Web Scraping', 'Data Analysis', 'API Integration', 'Automation', 'Machine Learning'];
  const template = templates[Math.floor(random() * templates.length)];
  const topic = topics[Math.floor(random() * topics.length)];
  return template.replace('{topic}', topic);
}

function generateSnippet(random) {
  const snippets = [
    'Learn how to effectively implement solutions with our comprehensive guide. Discover best practices and expert tips.',
    'This detailed tutorial walks you through step-by-step instructions for achieving optimal results.',
    'Get started quickly with our beginner-friendly approach. No prior experience required.',
    'Explore advanced techniques used by industry professionals to maximize efficiency.',
    'Find out why thousands of users trust our methods for reliable, consistent outcomes.'
  ];
  return snippets[Math.floor(random() * snippets.length)];
}

function generateBreadcrumb(random) {
  const paths = ['guides', 'tutorials', 'blog', 'docs', 'resources'];
  return paths[Math.floor(random() * paths.length)];
}

function generateStreetName(random) {
  const types = ['St', 'Ave', 'Blvd', 'Dr', 'Ln', 'Way', 'Ct'];
  const names = ['Oak', 'Main', 'Park', 'Cedar', 'Elm', 'Washington', 'Lake', 'Hill'];
  return `${names[Math.floor(random() * names.length)]} ${types[Math.floor(random() * types.length)]}`;
}

function getState(city) {
  const states = {
    'New York': 'NY', 'Los Angeles': 'CA', 'Chicago': 'IL', 'Houston': 'TX',
    'Phoenix': 'AZ', 'San Diego': 'CA', 'Dallas': 'TX', 'Austin': 'TX'
  };
  return states[city] || 'CA';
}

function generateRealEstateFeatures(random) {
  const allFeatures = ['Pool', 'Garage', 'Garden', 'Fireplace', 'Central AC', 'Hardwood Floors', 'Updated Kitchen', 'Smart Home', 'Solar Panels', 'Home Office'];
  const count = Math.floor(2 + random() * 5);
  return allFeatures.sort(() => random() - 0.5).slice(0, count);
}

function generatePhone(random) {
  return `(${Math.floor(200 + random() * 800)}) ${Math.floor(100 + random() * 900)}-${Math.floor(1000 + random() * 9000)}`;
}

function generateJobDescription(random) {
  return 'We are looking for a talented professional to join our growing team. You will work on challenging projects and collaborate with cross-functional teams to deliver exceptional results.';
}

function generateRequirement(random) {
  const reqs = [
    '3+ years of relevant experience',
    'Strong communication skills',
    'Bachelor\'s degree or equivalent',
    'Experience with modern tools',
    'Ability to work independently',
    'Team collaboration experience',
    'Problem-solving mindset'
  ];
  return reqs[Math.floor(random() * reqs.length)];
}

function generateBenefits(random) {
  const allBenefits = ['Health Insurance', '401k Match', 'Remote Work', 'Unlimited PTO', 'Stock Options', 'Learning Budget', 'Gym Membership', 'Free Lunch'];
  return allBenefits.sort(() => random() - 0.5).slice(0, Math.floor(3 + random() * 4));
}

function generateNewsTitle(category, random) {
  const templates = {
    'Technology': ['New AI Breakthrough Transforms {x}', 'Tech Giants Announce {x} Initiative', 'The Future of {x} is Here'],
    'Business': ['Market Sees Record {x}', 'Company Reports {x} Growth', 'Industry Leaders Discuss {x}'],
    'Politics': ['Government Announces {x} Policy', 'Leaders Meet to Discuss {x}', 'New {x} Legislation Proposed'],
    'Science': ['Scientists Discover {x}', 'New Research Reveals {x}', 'Breakthrough in {x} Studies'],
    'Health': ['Health Experts Recommend {x}', 'New Study Links {x} to Wellness', 'Medical Advances in {x}'],
    'Sports': ['Team Wins {x} Championship', 'Athletes Break {x} Record', 'Sports World Reacts to {x}'],
    'Entertainment': ['Celebrity Announces {x}', 'New {x} Series Premieres', 'Entertainment Industry Embraces {x}']
  };
  const words = ['Major', 'Surprising', 'Historic', 'Unprecedented', 'Exciting'];
  const catTemplates = templates[category] || templates['Technology'];
  const template = catTemplates[Math.floor(random() * catTemplates.length)];
  const word = words[Math.floor(random() * words.length)];
  return template.replace('{x}', word);
}

function generateSubtitle(random) {
  return 'Industry experts weigh in on the implications and what it means for the future.';
}

function generateArticleContent(random) {
  return 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.';
}

function generateCaption(random) {
  return 'Image: Illustration of the main topic covered in this article.';
}

function generateTag(random) {
  const tags = ['trending', 'breaking', 'exclusive', 'analysis', 'opinion', 'featured', 'popular'];
  return tags[Math.floor(random() * tags.length)];
}

// ============================================
// ENTERPRISE/COMPANY SIMULATORS
// ============================================

async function generateStockTradingData(count, seed) {
  log.info('Generating stock trading data (Bloomberg-style)...');
  const random = createSeededRandom(seed);
  const results = [];

  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT', 'UNH', 'JNJ', 'PG', 'HD', 'BAC'];
  const exchanges = ['NYSE', 'NASDAQ', 'LSE', 'TSE', 'HKEX'];
  const orderTypes = ['market', 'limit', 'stop', 'stop_limit', 'trailing_stop'];
  const sides = ['buy', 'sell'];

  for (let i = 0; i < count; i++) {
    const symbol = symbols[Math.floor(random() * symbols.length)];
    const basePrice = 50 + random() * 500;
    const timestamp = new Date(Date.now() - random() * 24 * 60 * 60 * 1000);
    const volume = Math.floor(100 + random() * 100000);

    results.push({
      tradeId: `TRD${Date.now()}${i}`,
      symbol,
      exchange: exchanges[Math.floor(random() * exchanges.length)],
      timestamp: timestamp.toISOString(),
      ohlcv: {
        open: Math.round(basePrice * (1 - random() * 0.02) * 100) / 100,
        high: Math.round(basePrice * (1 + random() * 0.03) * 100) / 100,
        low: Math.round(basePrice * (1 - random() * 0.03) * 100) / 100,
        close: Math.round(basePrice * 100) / 100,
        volume,
        vwap: Math.round(basePrice * (1 + (random() - 0.5) * 0.01) * 100) / 100
      },
      quote: {
        bid: Math.round(basePrice * 0.999 * 100) / 100,
        ask: Math.round(basePrice * 1.001 * 100) / 100,
        bidSize: Math.floor(100 + random() * 10000),
        askSize: Math.floor(100 + random() * 10000),
        spread: Math.round(basePrice * 0.002 * 100) / 100
      },
      order: {
        type: orderTypes[Math.floor(random() * orderTypes.length)],
        side: sides[Math.floor(random() * sides.length)],
        quantity: Math.floor(10 + random() * 1000),
        filledQuantity: Math.floor(10 + random() * 1000),
        status: random() > 0.1 ? 'filled' : random() > 0.5 ? 'partial' : 'pending'
      },
      marketData: {
        marketCap: Math.floor(random() * 3000) + 'B',
        peRatio: Math.round((10 + random() * 40) * 10) / 10,
        dividendYield: Math.round(random() * 5 * 100) / 100,
        beta: Math.round((0.5 + random() * 1.5) * 100) / 100,
        fiftyTwoWeekHigh: Math.round(basePrice * 1.3 * 100) / 100,
        fiftyTwoWeekLow: Math.round(basePrice * 0.7 * 100) / 100
      },
      analytics: {
        rsi: Math.round((20 + random() * 60) * 10) / 10,
        macd: Math.round((random() - 0.5) * 10 * 100) / 100,
        movingAvg50: Math.round(basePrice * (1 + (random() - 0.5) * 0.1) * 100) / 100,
        movingAvg200: Math.round(basePrice * (1 + (random() - 0.5) * 0.15) * 100) / 100
      },
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateMedicalData(count, seed) {
  log.info('Generating medical/healthcare data...');
  const random = createSeededRandom(seed);
  const results = [];

  const departments = ['Cardiology', 'Neurology', 'Orthopedics', 'Oncology', 'Pediatrics', 'Emergency', 'Radiology', 'Surgery'];
  const diagnoses = ['Hypertension', 'Type 2 Diabetes', 'Chronic Pain', 'Respiratory Infection', 'Anxiety Disorder', 'Cardiac Arrhythmia', 'Migraine', 'Osteoarthritis'];
  const procedures = ['Blood Test', 'MRI Scan', 'X-Ray', 'CT Scan', 'Ultrasound', 'ECG', 'Endoscopy', 'Biopsy'];
  const insurers = ['Blue Cross', 'Aetna', 'UnitedHealth', 'Cigna', 'Humana', 'Kaiser', 'Medicare', 'Medicaid'];
  const statuses = ['admitted', 'discharged', 'outpatient', 'emergency', 'scheduled'];

  for (let i = 0; i < count; i++) {
    const admitDate = new Date(Date.now() - random() * 365 * 24 * 60 * 60 * 1000);
    const age = Math.floor(18 + random() * 70);

    results.push({
      recordId: `MED${Date.now()}${i}`,
      patient: {
        id: `PAT${Math.floor(random() * 1000000)}`,
        age,
        gender: random() > 0.5 ? 'M' : 'F',
        bloodType: ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'][Math.floor(random() * 8)],
        allergies: random() > 0.7 ? ['Penicillin', 'Sulfa', 'Latex'][Math.floor(random() * 3)] : null
      },
      encounter: {
        type: statuses[Math.floor(random() * statuses.length)],
        department: departments[Math.floor(random() * departments.length)],
        admitDate: admitDate.toISOString(),
        dischargeDate: random() > 0.3 ? new Date(admitDate.getTime() + random() * 7 * 24 * 60 * 60 * 1000).toISOString() : null,
        lengthOfStay: Math.floor(1 + random() * 14)
      },
      diagnosis: {
        primary: diagnoses[Math.floor(random() * diagnoses.length)],
        secondary: random() > 0.5 ? diagnoses[Math.floor(random() * diagnoses.length)] : null,
        icdCode: `I${Math.floor(10 + random() * 90)}.${Math.floor(random() * 10)}`,
        severity: ['mild', 'moderate', 'severe', 'critical'][Math.floor(random() * 4)]
      },
      procedures: Array.from({ length: Math.floor(1 + random() * 3) }, () => ({
        name: procedures[Math.floor(random() * procedures.length)],
        cptCode: `${Math.floor(10000 + random() * 90000)}`,
        date: new Date(admitDate.getTime() + random() * 3 * 24 * 60 * 60 * 1000).toISOString(),
        result: random() > 0.1 ? 'normal' : 'abnormal'
      })),
      vitals: {
        bloodPressure: `${Math.floor(100 + random() * 60)}/${Math.floor(60 + random() * 40)}`,
        heartRate: Math.floor(60 + random() * 40),
        temperature: Math.round((97 + random() * 4) * 10) / 10,
        oxygenSaturation: Math.floor(94 + random() * 6),
        weight: Math.floor(120 + random() * 150),
        height: Math.floor(60 + random() * 20)
      },
      billing: {
        insurer: insurers[Math.floor(random() * insurers.length)],
        policyNumber: `POL${Math.floor(random() * 10000000)}`,
        totalCharges: Math.floor(1000 + random() * 50000),
        covered: Math.floor(800 + random() * 40000),
        patientResponsibility: Math.floor(100 + random() * 5000),
        claimStatus: random() > 0.2 ? 'approved' : random() > 0.5 ? 'pending' : 'denied'
      },
      provider: {
        physician: generateName(random),
        npi: `${Math.floor(1000000000 + random() * 9000000000)}`,
        facility: `${['Metro', 'Central', 'Regional', 'University'][Math.floor(random() * 4)]} Medical Center`
      },
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateCompanyData(count, seed) {
  log.info('Generating company/corporate data...');
  const random = createSeededRandom(seed);
  const results = [];

  const industries = ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail', 'Energy', 'Telecommunications', 'Transportation'];
  const companyTypes = ['Corporation', 'LLC', 'Partnership', 'Sole Proprietorship', 'S-Corp', 'Non-Profit'];
  const departments = ['Engineering', 'Sales', 'Marketing', 'Finance', 'HR', 'Operations', 'Legal', 'R&D'];

  for (let i = 0; i < count; i++) {
    const founded = Math.floor(1950 + random() * 74);
    const employees = Math.floor(10 + random() * 100000);
    const revenue = Math.floor(100000 + random() * 50000000000);

    results.push({
      companyId: `COM${Date.now()}${i}`,
      profile: {
        name: `${generateName(random).split(' ')[1]} ${['Industries', 'Corp', 'Inc', 'Holdings', 'Group', 'Technologies', 'Solutions'][Math.floor(random() * 7)]}`,
        ticker: random() > 0.5 ? `${String.fromCharCode(65 + Math.floor(random() * 26))}${String.fromCharCode(65 + Math.floor(random() * 26))}${String.fromCharCode(65 + Math.floor(random() * 26))}${String.fromCharCode(65 + Math.floor(random() * 26))}` : null,
        type: companyTypes[Math.floor(random() * companyTypes.length)],
        industry: industries[Math.floor(random() * industries.length)],
        founded,
        website: `https://example-company-${i}.com`,
        description: 'Leading provider of innovative solutions for modern enterprises.'
      },
      headquarters: {
        address: `${Math.floor(100 + random() * 9900)} Corporate Blvd`,
        city: ['New York', 'San Francisco', 'Chicago', 'Boston', 'Austin', 'Seattle'][Math.floor(random() * 6)],
        state: ['NY', 'CA', 'IL', 'MA', 'TX', 'WA'][Math.floor(random() * 6)],
        country: 'USA',
        timezone: 'America/New_York'
      },
      financials: {
        revenue,
        revenueGrowth: Math.round((random() * 40 - 10) * 10) / 10,
        netIncome: Math.floor(revenue * (0.05 + random() * 0.15)),
        grossMargin: Math.round((30 + random() * 40) * 10) / 10,
        operatingMargin: Math.round((10 + random() * 25) * 10) / 10,
        debtToEquity: Math.round(random() * 2 * 100) / 100,
        currentRatio: Math.round((1 + random() * 2) * 100) / 100,
        fiscalYearEnd: ['December', 'March', 'June', 'September'][Math.floor(random() * 4)]
      },
      workforce: {
        totalEmployees: employees,
        fullTime: Math.floor(employees * 0.85),
        partTime: Math.floor(employees * 0.1),
        contractors: Math.floor(employees * 0.05),
        departments: departments.slice(0, Math.floor(3 + random() * 5)).map(dept => ({
          name: dept,
          headcount: Math.floor(employees * (0.05 + random() * 0.2)),
          budget: Math.floor(revenue * (0.01 + random() * 0.1))
        })),
        avgTenure: Math.round((2 + random() * 8) * 10) / 10,
        turnoverRate: Math.round((5 + random() * 20) * 10) / 10
      },
      leadership: Array.from({ length: Math.floor(3 + random() * 5) }, () => ({
        name: generateName(random),
        title: ['CEO', 'CFO', 'CTO', 'COO', 'CMO', 'CHRO', 'CLO', 'CIO'][Math.floor(random() * 8)],
        since: Math.floor(2010 + random() * 14),
        compensation: Math.floor(500000 + random() * 10000000)
      })),
      metrics: {
        customerCount: Math.floor(100 + random() * 1000000),
        nps: Math.floor(-20 + random() * 100),
        marketShare: Math.round(random() * 30 * 10) / 10,
        brandValue: Math.floor(random() * 50) + 'B'
      },
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateSupplyChainData(count, seed) {
  log.info('Generating supply chain data...');
  const random = createSeededRandom(seed);
  const results = [];

  const productCategories = ['Electronics', 'Raw Materials', 'Components', 'Finished Goods', 'Packaging', 'Chemicals', 'Textiles', 'Machinery'];
  const statuses = ['in_transit', 'delivered', 'pending', 'delayed', 'customs_hold', 'processing', 'shipped', 'cancelled'];
  const transportModes = ['air', 'sea', 'rail', 'truck', 'multimodal'];
  const warehouses = ['WH-NYC-01', 'WH-LAX-02', 'WH-CHI-03', 'WH-HOU-04', 'WH-SEA-05', 'WH-MIA-06'];
  const countries = ['USA', 'China', 'Germany', 'Japan', 'Mexico', 'Vietnam', 'India', 'South Korea'];

  for (let i = 0; i < count; i++) {
    const orderDate = new Date(Date.now() - random() * 90 * 24 * 60 * 60 * 1000);
    const quantity = Math.floor(10 + random() * 10000);
    const unitPrice = Math.round((1 + random() * 500) * 100) / 100;

    results.push({
      shipmentId: `SHP${Date.now()}${i}`,
      order: {
        orderId: `ORD${Math.floor(random() * 10000000)}`,
        orderDate: orderDate.toISOString(),
        priority: ['standard', 'express', 'critical'][Math.floor(random() * 3)],
        status: statuses[Math.floor(random() * statuses.length)]
      },
      product: {
        sku: `SKU-${Math.floor(100000 + random() * 900000)}`,
        name: `${productCategories[Math.floor(random() * productCategories.length)]} Item ${Math.floor(random() * 1000)}`,
        category: productCategories[Math.floor(random() * productCategories.length)],
        quantity,
        unitPrice,
        totalValue: Math.round(quantity * unitPrice * 100) / 100,
        weight: Math.round((0.1 + random() * 100) * 10) / 10,
        dimensions: {
          length: Math.floor(10 + random() * 100),
          width: Math.floor(10 + random() * 100),
          height: Math.floor(10 + random() * 50)
        }
      },
      supplier: {
        id: `SUP${Math.floor(random() * 10000)}`,
        name: `${generateName(random).split(' ')[1]} Supply Co`,
        country: countries[Math.floor(random() * countries.length)],
        leadTime: Math.floor(7 + random() * 60),
        rating: Math.round((3 + random() * 2) * 10) / 10,
        onTimeDelivery: Math.round((70 + random() * 30) * 10) / 10
      },
      logistics: {
        carrier: ['FedEx', 'UPS', 'DHL', 'Maersk', 'Expeditors', 'DB Schenker'][Math.floor(random() * 6)],
        mode: transportModes[Math.floor(random() * transportModes.length)],
        trackingNumber: `TRK${Math.floor(random() * 1000000000000)}`,
        origin: {
          facility: warehouses[Math.floor(random() * warehouses.length)],
          country: countries[Math.floor(random() * countries.length)],
          departureDate: orderDate.toISOString()
        },
        destination: {
          facility: warehouses[Math.floor(random() * warehouses.length)],
          country: countries[Math.floor(random() * countries.length)],
          eta: new Date(orderDate.getTime() + (7 + random() * 30) * 24 * 60 * 60 * 1000).toISOString()
        },
        currentLocation: {
          lat: 25 + random() * 25,
          lng: -120 + random() * 60,
          lastUpdate: new Date(orderDate.getTime() + random() * 7 * 24 * 60 * 60 * 1000).toISOString()
        }
      },
      inventory: {
        warehouse: warehouses[Math.floor(random() * warehouses.length)],
        stockLevel: Math.floor(random() * 5000),
        reorderPoint: Math.floor(100 + random() * 500),
        safetyStock: Math.floor(50 + random() * 200),
        daysOfSupply: Math.floor(10 + random() * 90)
      },
      costs: {
        productCost: Math.round(quantity * unitPrice * 100) / 100,
        shippingCost: Math.round(quantity * unitPrice * (0.05 + random() * 0.15) * 100) / 100,
        tariffs: Math.round(quantity * unitPrice * random() * 0.1 * 100) / 100,
        insurance: Math.round(quantity * unitPrice * 0.02 * 100) / 100,
        totalLandedCost: Math.round(quantity * unitPrice * (1.1 + random() * 0.2) * 100) / 100
      },
      compliance: {
        hsCode: `${Math.floor(1000 + random() * 9000)}.${Math.floor(10 + random() * 90)}`,
        countryOfOrigin: countries[Math.floor(random() * countries.length)],
        certificates: random() > 0.5 ? ['ISO 9001', 'CE', 'RoHS'][Math.floor(random() * 3)] : null,
        customsCleared: random() > 0.3
      },
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateFinancialData(count, seed) {
  log.info('Generating financial services data...');
  const random = createSeededRandom(seed);
  const results = [];

  const accountTypes = ['checking', 'savings', 'investment', 'retirement', 'credit', 'loan', 'mortgage'];
  const transactionTypes = ['debit', 'credit', 'transfer', 'payment', 'withdrawal', 'deposit', 'fee', 'interest'];
  const categories = ['groceries', 'utilities', 'entertainment', 'dining', 'travel', 'shopping', 'healthcare', 'insurance', 'investment'];
  const institutions = ['Chase', 'Bank of America', 'Wells Fargo', 'Citi', 'Capital One', 'Goldman Sachs', 'Morgan Stanley', 'Fidelity'];

  for (let i = 0; i < count; i++) {
    const transactionDate = new Date(Date.now() - random() * 365 * 24 * 60 * 60 * 1000);
    const amount = Math.round((1 + random() * 10000) * 100) / 100;

    results.push({
      transactionId: `TXN${Date.now()}${i}`,
      account: {
        accountId: `ACC${Math.floor(random() * 100000000)}`,
        type: accountTypes[Math.floor(random() * accountTypes.length)],
        institution: institutions[Math.floor(random() * institutions.length)],
        balance: Math.round((1000 + random() * 500000) * 100) / 100,
        availableCredit: random() > 0.5 ? Math.round((5000 + random() * 50000) * 100) / 100 : null,
        interestRate: Math.round((random() * 25) * 100) / 100
      },
      transaction: {
        type: transactionTypes[Math.floor(random() * transactionTypes.length)],
        amount,
        currency: 'USD',
        date: transactionDate.toISOString(),
        description: `${categories[Math.floor(random() * categories.length)].toUpperCase()} - ${generateName(random).split(' ')[1]} Store`,
        category: categories[Math.floor(random() * categories.length)],
        status: random() > 0.05 ? 'completed' : random() > 0.5 ? 'pending' : 'failed',
        merchant: {
          name: `${generateName(random).split(' ')[1]} ${['Store', 'Shop', 'Market', 'Services'][Math.floor(random() * 4)]}`,
          category: categories[Math.floor(random() * categories.length)],
          mcc: `${Math.floor(1000 + random() * 9000)}`
        }
      },
      card: random() > 0.3 ? {
        last4: `${Math.floor(1000 + random() * 9000)}`,
        brand: ['Visa', 'Mastercard', 'Amex', 'Discover'][Math.floor(random() * 4)],
        expiryMonth: Math.floor(1 + random() * 12),
        expiryYear: Math.floor(2025 + random() * 5)
      } : null,
      fraud: {
        score: Math.round(random() * 100),
        flagged: random() > 0.95,
        rules: random() > 0.9 ? ['unusual_location', 'high_amount', 'velocity_check'][Math.floor(random() * 3)] : null
      },
      analytics: {
        dayOfWeek: transactionDate.getDay(),
        hourOfDay: transactionDate.getHours(),
        isRecurring: random() > 0.7,
        monthlyAverage: Math.round((100 + random() * 2000) * 100) / 100
      },
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}

async function generateBloombergData(count, seed) {
  log.info('Generating Bloomberg terminal-style data...');
  const random = createSeededRandom(seed);
  const results = [];

  const assetClasses = ['equity', 'fixed_income', 'commodity', 'fx', 'derivative', 'crypto'];
  const sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Materials', 'Utilities'];
  const ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'B', 'CCC'];
  const newsCategories = ['earnings', 'merger', 'regulatory', 'analyst_upgrade', 'analyst_downgrade', 'dividend', 'lawsuit', 'executive'];

  for (let i = 0; i < count; i++) {
    const timestamp = new Date(Date.now() - random() * 24 * 60 * 60 * 1000);
    const basePrice = 10 + random() * 500;

    results.push({
      terminalId: `BBG${Date.now()}${i}`,
      security: {
        ticker: `${String.fromCharCode(65 + Math.floor(random() * 26))}${String.fromCharCode(65 + Math.floor(random() * 26))}${String.fromCharCode(65 + Math.floor(random() * 26))}${String.fromCharCode(65 + Math.floor(random() * 26))}`,
        name: `${generateName(random).split(' ')[1]} ${['Corp', 'Inc', 'Ltd', 'Holdings', 'Group'][Math.floor(random() * 5)]}`,
        assetClass: assetClasses[Math.floor(random() * assetClasses.length)],
        sector: sectors[Math.floor(random() * sectors.length)],
        country: ['US', 'GB', 'JP', 'DE', 'CN', 'FR', 'CA', 'AU'][Math.floor(random() * 8)],
        currency: ['USD', 'EUR', 'GBP', 'JPY', 'CNY'][Math.floor(random() * 5)],
        isin: `US${Math.floor(1000000000 + random() * 9000000000)}`,
        cusip: `${Math.floor(100000000 + random() * 900000000)}`
      },
      pricing: {
        last: Math.round(basePrice * 100) / 100,
        bid: Math.round(basePrice * 0.999 * 100) / 100,
        ask: Math.round(basePrice * 1.001 * 100) / 100,
        open: Math.round(basePrice * (1 - random() * 0.02) * 100) / 100,
        high: Math.round(basePrice * (1 + random() * 0.03) * 100) / 100,
        low: Math.round(basePrice * (1 - random() * 0.03) * 100) / 100,
        close: Math.round(basePrice * (1 + (random() - 0.5) * 0.02) * 100) / 100,
        change: Math.round((random() - 0.5) * 10 * 100) / 100,
        changePercent: Math.round((random() - 0.5) * 5 * 100) / 100,
        volume: Math.floor(random() * 50000000),
        avgVolume: Math.floor(random() * 30000000)
      },
      fundamentals: {
        marketCap: Math.floor(random() * 3000) + 'B',
        enterpriseValue: Math.floor(random() * 3500) + 'B',
        peRatio: Math.round((5 + random() * 50) * 10) / 10,
        forwardPe: Math.round((5 + random() * 40) * 10) / 10,
        pbRatio: Math.round((0.5 + random() * 10) * 10) / 10,
        evEbitda: Math.round((5 + random() * 30) * 10) / 10,
        debtToEquity: Math.round(random() * 3 * 100) / 100,
        roe: Math.round((5 + random() * 30) * 10) / 10,
        eps: Math.round((random() * 20) * 100) / 100,
        dividend: Math.round(random() * 5 * 100) / 100,
        payoutRatio: Math.round((20 + random() * 60) * 10) / 10
      },
      credit: {
        rating: ratings[Math.floor(random() * ratings.length)],
        outlook: ['positive', 'stable', 'negative'][Math.floor(random() * 3)],
        agency: ['S&P', 'Moody\'s', 'Fitch'][Math.floor(random() * 3)],
        spread: Math.round((50 + random() * 500)),
        cds: Math.round((20 + random() * 300))
      },
      analytics: {
        beta: Math.round((0.5 + random() * 1.5) * 100) / 100,
        sharpeRatio: Math.round((random() * 3) * 100) / 100,
        volatility: Math.round((10 + random() * 40) * 10) / 10,
        correlation: Math.round((random() * 2 - 1) * 100) / 100,
        var95: Math.round((random() * 10) * 100) / 100,
        maxDrawdown: Math.round((5 + random() * 30) * 10) / 10
      },
      consensus: {
        recommendation: ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell'][Math.floor(random() * 5)],
        targetPrice: Math.round(basePrice * (1 + (random() - 0.3) * 0.5) * 100) / 100,
        numAnalysts: Math.floor(5 + random() * 40),
        buyRatings: Math.floor(random() * 30),
        holdRatings: Math.floor(random() * 15),
        sellRatings: Math.floor(random() * 10)
      },
      news: {
        headline: `${generateName(random).split(' ')[1]} Corp ${newsCategories[Math.floor(random() * newsCategories.length)].replace('_', ' ')} update`,
        source: ['Reuters', 'Bloomberg', 'WSJ', 'FT', 'CNBC'][Math.floor(random() * 5)],
        timestamp: timestamp.toISOString(),
        sentiment: ['positive', 'neutral', 'negative'][Math.floor(random() * 3)],
        relevance: Math.round(random() * 100)
      },
      events: {
        nextEarnings: new Date(Date.now() + random() * 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        exDividendDate: random() > 0.5 ? new Date(Date.now() + random() * 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0] : null,
        annualMeeting: new Date(Date.now() + random() * 180 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
      },
      scrapedAt: new Date().toISOString()
    });
  }

  return results;
}
