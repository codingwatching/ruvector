/**
 * Apify Actor Integrations
 *
 * One-click integration with popular Apify actors for RAG/memory use cases.
 * Enables instant data transformation from scrapers to AI-ready formats.
 */

// ============================================
// SUPPORTED APIFY ACTORS
// ============================================

export const SUPPORTED_ACTORS = {
  // Google/Maps
  'apify/google-maps-scraper': {
    name: 'Google Maps Scraper',
    category: 'local-business',
    defaultFields: ['title', 'description', 'address', 'phone', 'website', 'rating', 'reviews', 'categories'],
    transform: transformGoogleMaps
  },
  'apify/google-search-scraper': {
    name: 'Google Search Scraper',
    category: 'search',
    defaultFields: ['title', 'description', 'url', 'position'],
    transform: transformGoogleSearch
  },

  // Social Media
  'apify/instagram-scraper': {
    name: 'Instagram Scraper',
    category: 'social-media',
    defaultFields: ['caption', 'hashtags', 'likes', 'comments', 'ownerUsername', 'timestamp'],
    transform: transformInstagram
  },
  'apify/tiktok-scraper': {
    name: 'TikTok Scraper',
    category: 'social-media',
    defaultFields: ['text', 'hashtags', 'likes', 'comments', 'shares', 'authorMeta', 'createTime'],
    transform: transformTikTok
  },
  'apify/youtube-scraper': {
    name: 'YouTube Scraper',
    category: 'video',
    defaultFields: ['title', 'description', 'viewCount', 'likes', 'channelName', 'publishedAt'],
    transform: transformYouTube
  },
  'apify/twitter-scraper': {
    name: 'Twitter/X Scraper',
    category: 'social-media',
    defaultFields: ['text', 'retweets', 'likes', 'replies', 'author', 'createdAt'],
    transform: transformTwitter
  },

  // E-commerce
  'apify/amazon-scraper': {
    name: 'Amazon Scraper',
    category: 'ecommerce',
    defaultFields: ['title', 'price', 'rating', 'reviewCount', 'description', 'brand', 'category'],
    transform: transformAmazon
  },
  'apify/shopify-scraper': {
    name: 'Shopify Scraper',
    category: 'ecommerce',
    defaultFields: ['title', 'price', 'description', 'vendor', 'productType', 'images'],
    transform: transformShopify
  },

  // General Web
  'apify/web-scraper': {
    name: 'Web Scraper',
    category: 'general',
    defaultFields: ['url', 'title', 'text', 'html'],
    transform: transformWebScraper
  },
  'apify/website-content-crawler': {
    name: 'Website Content Crawler',
    category: 'content',
    defaultFields: ['url', 'title', 'text', 'markdown'],
    transform: transformContentCrawler
  },
  'apify/cheerio-scraper': {
    name: 'Cheerio Scraper',
    category: 'general',
    defaultFields: ['url', 'pageTitle', 'data'],
    transform: transformCheerio
  },

  // News & Content
  'apify/news-scraper': {
    name: 'News Scraper',
    category: 'news',
    defaultFields: ['title', 'text', 'author', 'publishedAt', 'source'],
    transform: transformNews
  },

  // Job & Business
  'apify/linkedin-scraper': {
    name: 'LinkedIn Scraper',
    category: 'professional',
    defaultFields: ['title', 'company', 'location', 'description', 'salary'],
    transform: transformLinkedIn
  },

  // Reviews & Local
  'trudax/tripadvisor-scraper': {
    name: 'TripAdvisor Scraper',
    category: 'reviews',
    defaultFields: ['name', 'rating', 'reviewCount', 'address', 'priceLevel', 'cuisine'],
    transform: transformTripAdvisor
  },
  'maxcopell/yelp-scraper': {
    name: 'Yelp Scraper',
    category: 'reviews',
    defaultFields: ['name', 'rating', 'reviewCount', 'address', 'categories', 'phone'],
    transform: transformYelp
  },
  'trudax/booking-scraper': {
    name: 'Booking.com Scraper',
    category: 'travel',
    defaultFields: ['name', 'rating', 'price', 'location', 'amenities', 'reviewScore'],
    transform: transformBooking
  },

  // Real Estate
  'petr_cermak/zillow-scraper': {
    name: 'Zillow Scraper',
    category: 'real-estate',
    defaultFields: ['address', 'price', 'bedrooms', 'bathrooms', 'sqft', 'propertyType'],
    transform: transformZillow
  },
  'epctex/craigslist-scraper': {
    name: 'Craigslist Scraper',
    category: 'classifieds',
    defaultFields: ['title', 'price', 'location', 'description', 'category', 'postedAt'],
    transform: transformCraigslist
  },

  // Social Platforms
  'apify/reddit-scraper': {
    name: 'Reddit Scraper',
    category: 'social-media',
    defaultFields: ['title', 'text', 'subreddit', 'score', 'comments', 'author'],
    transform: transformReddit
  },
  'apify/facebook-posts-scraper': {
    name: 'Facebook Posts Scraper',
    category: 'social-media',
    defaultFields: ['text', 'likes', 'comments', 'shares', 'author', 'timestamp'],
    transform: transformFacebook
  },

  // Places & Maps
  'compass/google-places-api': {
    name: 'Google Places API',
    category: 'local-business',
    defaultFields: ['name', 'rating', 'address', 'phone', 'website', 'types', 'priceLevel'],
    transform: transformGooglePlaces
  }
};

// ============================================
// USE CASE TEMPLATES
// ============================================

export const USE_CASE_TEMPLATES = {
  'lead-intelligence': {
    name: 'Lead Intelligence',
    description: 'Sales teams memorizing prospect data for personalized outreach',
    targetUsers: ['Sales', 'BD', 'Account Executives'],
    suggestedActors: ['apify/google-maps-scraper', 'apify/linkedin-scraper'],
    memorizeFields: ['title', 'description', 'address', 'phone', 'website', 'rating'],
    enrichWith: ['company_size', 'industry', 'decision_makers'],
    outputFormat: {
      prospectId: 'string',
      companyName: 'string',
      contactInfo: 'object',
      businessProfile: 'string',
      leadScore: 'number (1-100)',
      keyInsights: 'array<string>',
      suggestedApproach: 'string'
    }
  },

  'competitor-monitor': {
    name: 'Competitor Monitor',
    description: 'Track competitor mentions, pricing changes, and market positioning',
    targetUsers: ['Marketing', 'Strategy', 'Product'],
    suggestedActors: ['apify/google-search-scraper', 'apify/twitter-scraper', 'apify/news-scraper'],
    memorizeFields: ['title', 'description', 'url', 'timestamp', 'sentiment'],
    enrichWith: ['competitor_name', 'mention_type', 'market_impact'],
    outputFormat: {
      competitorId: 'string',
      competitorName: 'string',
      mentions: 'array<object>',
      pricingChanges: 'array<object>',
      marketSentiment: 'number (-1 to 1)',
      alerts: 'array<string>',
      trendAnalysis: 'object'
    }
  },

  'support-knowledge': {
    name: 'Support Knowledge Base',
    description: 'Customer support RAG system for instant answers',
    targetUsers: ['Support Teams', 'Customer Success', 'Help Desk'],
    suggestedActors: ['apify/website-content-crawler', 'apify/web-scraper'],
    memorizeFields: ['url', 'title', 'text', 'category', 'lastUpdated'],
    enrichWith: ['topic', 'difficulty_level', 'related_articles'],
    outputFormat: {
      articleId: 'string',
      title: 'string',
      content: 'string',
      summary: 'string',
      topics: 'array<string>',
      relatedQuestions: 'array<string>',
      embedding: 'array<number>'
    }
  },

  'research-assistant': {
    name: 'Research Assistant',
    description: 'Academic and market research with comprehensive sourcing',
    targetUsers: ['Researchers', 'Analysts', 'Consultants'],
    suggestedActors: ['apify/google-search-scraper', 'apify/news-scraper', 'apify/website-content-crawler'],
    memorizeFields: ['title', 'text', 'url', 'publishedAt', 'author', 'source'],
    enrichWith: ['credibility_score', 'citations', 'key_findings'],
    outputFormat: {
      sourceId: 'string',
      title: 'string',
      content: 'string',
      summary: 'string',
      keyFindings: 'array<string>',
      credibilityScore: 'number (1-100)',
      citations: 'array<object>',
      relatedSources: 'array<string>'
    }
  },

  'content-library': {
    name: 'Content Library',
    description: 'Content creators reference library for inspiration and research',
    targetUsers: ['Content Creators', 'Marketers', 'Social Media Managers'],
    suggestedActors: ['apify/instagram-scraper', 'apify/tiktok-scraper', 'apify/youtube-scraper'],
    memorizeFields: ['caption', 'hashtags', 'likes', 'comments', 'shares', 'engagement'],
    enrichWith: ['content_type', 'performance_tier', 'trending_score'],
    outputFormat: {
      contentId: 'string',
      platform: 'string',
      content: 'string',
      hashtags: 'array<string>',
      performance: 'object',
      trendingScore: 'number (1-100)',
      contentStyle: 'string',
      suggestedVariations: 'array<string>'
    }
  },

  'product-catalog': {
    name: 'Product Catalog Memory',
    description: 'E-commerce product memory for comparison and recommendations',
    targetUsers: ['E-commerce', 'Retail', 'Buyers'],
    suggestedActors: ['apify/amazon-scraper', 'apify/shopify-scraper', 'apify/google-maps-scraper'],
    memorizeFields: ['title', 'price', 'rating', 'reviewCount', 'description', 'category', 'brand'],
    enrichWith: ['price_history', 'competitor_prices', 'stock_status'],
    outputFormat: {
      productId: 'string',
      title: 'string',
      price: 'number',
      priceHistory: 'array<object>',
      rating: 'number',
      category: 'string',
      brand: 'string',
      competitorPrices: 'array<object>',
      recommendedAlternatives: 'array<string>',
      embedding: 'array<number>'
    }
  },

  'review-aggregator': {
    name: 'Review Aggregator',
    description: 'Aggregate and analyze reviews from multiple platforms',
    targetUsers: ['Product Managers', 'Brand Managers', 'Customer Experience'],
    suggestedActors: ['trudax/tripadvisor-scraper', 'maxcopell/yelp-scraper', 'apify/google-maps-scraper'],
    memorizeFields: ['name', 'rating', 'reviewCount', 'text', 'sentiment', 'categories'],
    enrichWith: ['sentiment_score', 'common_themes', 'rating_trend'],
    outputFormat: {
      entityId: 'string',
      name: 'string',
      averageRating: 'number',
      totalReviews: 'number',
      platforms: 'array<object>',
      sentimentAnalysis: 'object',
      topPraises: 'array<string>',
      topComplaints: 'array<string>',
      embedding: 'array<number>'
    }
  },

  'price-tracker': {
    name: 'Price Tracker',
    description: 'Monitor prices across platforms for competitive intelligence',
    targetUsers: ['Pricing Teams', 'Buyers', 'Resellers'],
    suggestedActors: ['apify/amazon-scraper', 'petr_cermak/zillow-scraper', 'trudax/booking-scraper'],
    memorizeFields: ['title', 'price', 'currency', 'availability', 'seller', 'timestamp'],
    enrichWith: ['price_history', 'price_alerts', 'competitor_comparison'],
    outputFormat: {
      productId: 'string',
      title: 'string',
      currentPrice: 'number',
      priceHistory: 'array<object>',
      lowestPrice: 'number',
      highestPrice: 'number',
      priceChange: 'number',
      competitors: 'array<object>',
      embedding: 'array<number>'
    }
  },

  'social-listening': {
    name: 'Social Listening',
    description: 'Monitor social conversations about brands, topics, and trends',
    targetUsers: ['Social Media Managers', 'PR Teams', 'Brand Managers'],
    suggestedActors: ['apify/reddit-scraper', 'apify/twitter-scraper', 'apify/facebook-posts-scraper'],
    memorizeFields: ['text', 'author', 'engagement', 'sentiment', 'platform', 'timestamp'],
    enrichWith: ['sentiment_analysis', 'influencer_score', 'viral_potential'],
    outputFormat: {
      postId: 'string',
      platform: 'string',
      content: 'string',
      author: 'object',
      engagement: 'object',
      sentiment: 'number (-1 to 1)',
      mentions: 'array<string>',
      hashtags: 'array<string>',
      viralScore: 'number (1-100)',
      embedding: 'array<number>'
    }
  },

  'talent-sourcing': {
    name: 'Talent Sourcing',
    description: 'Recruit and source candidates from job platforms',
    targetUsers: ['Recruiters', 'HR Teams', 'Talent Acquisition'],
    suggestedActors: ['apify/linkedin-scraper', 'epctex/craigslist-scraper'],
    memorizeFields: ['title', 'company', 'location', 'skills', 'experience', 'salary'],
    enrichWith: ['skill_match', 'culture_fit', 'availability'],
    outputFormat: {
      candidateId: 'string',
      name: 'string',
      currentRole: 'string',
      company: 'string',
      skills: 'array<string>',
      experience: 'number',
      location: 'string',
      matchScore: 'number (1-100)',
      embedding: 'array<number>'
    }
  },

  'real-estate-intel': {
    name: 'Real Estate Intelligence',
    description: 'Market analysis and property intelligence for real estate',
    targetUsers: ['Real Estate Agents', 'Investors', 'Property Managers'],
    suggestedActors: ['petr_cermak/zillow-scraper', 'apify/google-maps-scraper', 'epctex/craigslist-scraper'],
    memorizeFields: ['address', 'price', 'sqft', 'bedrooms', 'bathrooms', 'propertyType'],
    enrichWith: ['market_trends', 'comparable_sales', 'neighborhood_score'],
    outputFormat: {
      propertyId: 'string',
      address: 'string',
      price: 'number',
      pricePerSqft: 'number',
      propertyType: 'string',
      specs: 'object',
      marketAnalysis: 'object',
      comparables: 'array<object>',
      investmentScore: 'number (1-100)',
      embedding: 'array<number>'
    }
  },

  'travel-planner': {
    name: 'Travel Planner',
    description: 'Plan trips with aggregated hotel, restaurant, and activity data',
    targetUsers: ['Travel Agents', 'Travelers', 'Tourism Boards'],
    suggestedActors: ['trudax/tripadvisor-scraper', 'trudax/booking-scraper', 'apify/google-maps-scraper'],
    memorizeFields: ['name', 'rating', 'price', 'location', 'amenities', 'reviews'],
    enrichWith: ['booking_availability', 'best_time_to_visit', 'local_tips'],
    outputFormat: {
      placeId: 'string',
      name: 'string',
      type: 'string (hotel, restaurant, attraction)',
      rating: 'number',
      priceRange: 'string',
      location: 'object',
      highlights: 'array<string>',
      reviews: 'array<object>',
      embedding: 'array<number>'
    }
  }
};

// ============================================
// TRANSFORM FUNCTIONS
// ============================================

function transformGoogleMaps(item) {
  return {
    id: item.placeId || generateId(),
    source: 'google-maps',
    businessName: item.title || item.name,
    description: item.description || item.categoryName,
    address: {
      full: item.address,
      street: item.street,
      city: item.city,
      state: item.state,
      country: item.countryCode,
      postalCode: item.postalCode
    },
    contact: {
      phone: item.phone,
      website: item.website,
      email: item.email
    },
    metrics: {
      rating: item.rating || item.totalScore,
      reviewCount: item.reviewsCount,
      priceLevel: item.priceLevel
    },
    categories: item.categories || [item.categoryName],
    location: {
      lat: item.location?.lat,
      lng: item.location?.lng
    },
    hours: item.openingHours,
    reviews: (item.reviews || []).slice(0, 5).map(r => ({
      text: r.text,
      rating: r.rating || r.stars,
      author: r.name || r.author
    })),
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformGoogleSearch(item) {
  return {
    id: generateId(),
    source: 'google-search',
    position: item.position,
    title: item.title,
    description: item.description || item.snippet,
    url: item.url || item.link,
    domain: extractDomain(item.url || item.link),
    type: item.type || 'organic',
    richSnippet: item.richSnippet || null,
    sitelinks: item.sitelinks || [],
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformInstagram(item) {
  return {
    id: item.id || item.shortCode || generateId(),
    source: 'instagram',
    type: item.type || 'post',
    caption: item.caption || item.text,
    hashtags: extractHashtags(item.caption || item.text),
    mentions: extractMentions(item.caption || item.text),
    author: {
      username: item.ownerUsername || item.owner?.username,
      id: item.ownerId || item.owner?.id,
      verified: item.ownerVerified || item.owner?.isVerified
    },
    engagement: {
      likes: item.likesCount || item.likes,
      comments: item.commentsCount || item.comments,
      views: item.videoViewCount || item.views
    },
    media: {
      type: item.type,
      url: item.displayUrl || item.imageUrl,
      dimensions: item.dimensions
    },
    timestamp: item.timestamp || item.createdAt,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformTikTok(item) {
  return {
    id: item.id || generateId(),
    source: 'tiktok',
    text: item.text || item.desc,
    hashtags: item.hashtags || extractHashtags(item.text || item.desc),
    mentions: item.mentions || extractMentions(item.text || item.desc),
    author: {
      username: item.authorMeta?.name || item.author?.uniqueId,
      nickname: item.authorMeta?.nickName || item.author?.nickname,
      verified: item.authorMeta?.verified || item.author?.verified,
      followers: item.authorMeta?.fans || item.author?.stats?.followerCount
    },
    engagement: {
      likes: item.diggCount || item.likes,
      comments: item.commentCount || item.comments,
      shares: item.shareCount || item.shares,
      views: item.playCount || item.views
    },
    music: {
      title: item.musicMeta?.musicName || item.music?.title,
      author: item.musicMeta?.musicAuthor || item.music?.authorName
    },
    duration: item.videoMeta?.duration || item.video?.duration,
    timestamp: item.createTime || item.createdAt,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformYouTube(item) {
  return {
    id: item.id || item.videoId || generateId(),
    source: 'youtube',
    title: item.title,
    description: item.description?.substring(0, 500),
    channel: {
      name: item.channelName || item.channel?.name,
      id: item.channelId || item.channel?.id,
      subscribers: item.channel?.subscribers
    },
    engagement: {
      views: item.viewCount || item.views,
      likes: item.likeCount || item.likes,
      comments: item.commentCount || item.comments
    },
    duration: item.duration,
    publishedAt: item.publishedAt || item.date,
    thumbnail: item.thumbnailUrl || item.thumbnail,
    tags: item.tags || [],
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformTwitter(item) {
  return {
    id: item.id || item.tweetId || generateId(),
    source: 'twitter',
    text: item.text || item.fullText,
    hashtags: item.hashtags || extractHashtags(item.text || item.fullText),
    mentions: item.mentions || extractMentions(item.text || item.fullText),
    author: {
      username: item.author?.username || item.user?.screenName,
      name: item.author?.name || item.user?.name,
      verified: item.author?.isVerified || item.user?.verified,
      followers: item.author?.followers || item.user?.followersCount
    },
    engagement: {
      retweets: item.retweetCount || item.retweets,
      likes: item.likeCount || item.likes || item.favoriteCount,
      replies: item.replyCount || item.replies,
      quotes: item.quoteCount || item.quotes
    },
    media: item.media || [],
    isRetweet: item.isRetweet || false,
    isReply: item.isReply || !!item.inReplyToStatusId,
    timestamp: item.createdAt || item.timestamp,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformAmazon(item) {
  return {
    id: item.asin || item.productId || generateId(),
    source: 'amazon',
    title: item.title,
    price: item.price || item.currentPrice,
    originalPrice: item.originalPrice || item.wasPrice,
    currency: item.currency || 'USD',
    rating: item.rating || item.stars,
    reviewCount: item.reviewCount || item.reviewsCount,
    description: item.description,
    brand: item.brand || item.manufacturer,
    category: item.category || item.categoryBreadcrumb,
    features: item.features || item.bulletPoints,
    images: item.images || [item.imageUrl],
    inStock: item.inStock ?? true,
    seller: item.seller || item.sellerName,
    url: item.url || item.productUrl,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformShopify(item) {
  return {
    id: item.id || item.productId || generateId(),
    source: 'shopify',
    title: item.title,
    description: item.description || item.bodyHtml,
    price: item.price || item.variants?.[0]?.price,
    compareAtPrice: item.compareAtPrice || item.variants?.[0]?.compareAtPrice,
    vendor: item.vendor,
    productType: item.productType,
    tags: item.tags || [],
    images: (item.images || []).map(i => i.src || i),
    variants: item.variants || [],
    handle: item.handle,
    url: item.url,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformWebScraper(item) {
  return {
    id: generateId(),
    source: 'web-scraper',
    url: item.url || item.pageUrl,
    title: item.title || item.pageTitle,
    text: item.text || item.content,
    html: item.html,
    metadata: item.metadata || {},
    links: item.links || [],
    images: item.images || [],
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformContentCrawler(item) {
  return {
    id: generateId(),
    source: 'content-crawler',
    url: item.url,
    title: item.title || item.metadata?.title,
    text: item.text || item.content,
    markdown: item.markdown,
    metadata: {
      description: item.metadata?.description,
      keywords: item.metadata?.keywords,
      author: item.metadata?.author
    },
    headers: item.headers || [],
    links: item.links || [],
    wordCount: (item.text || item.content || '').split(/\s+/).length,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformCheerio(item) {
  return {
    id: generateId(),
    source: 'cheerio-scraper',
    url: item.url,
    title: item.pageTitle || item.title,
    data: item.data || {},
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformNews(item) {
  return {
    id: item.id || generateId(),
    source: 'news',
    title: item.title || item.headline,
    text: item.text || item.content || item.body,
    summary: item.summary || item.description,
    author: item.author || item.byline,
    source: item.source || item.publisher,
    category: item.category || item.section,
    publishedAt: item.publishedAt || item.date,
    url: item.url || item.link,
    images: item.images || (item.image ? [item.image] : []),
    tags: item.tags || item.keywords || [],
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformLinkedIn(item) {
  return {
    id: item.id || item.jobId || generateId(),
    source: 'linkedin',
    title: item.title || item.jobTitle,
    company: {
      name: item.company || item.companyName,
      logo: item.companyLogo,
      url: item.companyUrl
    },
    location: item.location,
    remote: item.remote || item.workplaceType === 'Remote',
    description: item.description || item.jobDescription,
    salary: item.salary || item.salaryRange,
    employmentType: item.employmentType || item.type,
    experienceLevel: item.experienceLevel || item.seniorityLevel,
    postedAt: item.postedAt || item.listDate,
    applicants: item.applicants || item.applicantCount,
    url: item.url || item.jobUrl,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformTripAdvisor(item) {
  return {
    id: item.id || item.locationId || generateId(),
    source: 'tripadvisor',
    name: item.name || item.title,
    type: item.type || item.category,
    rating: item.rating || item.overallRating,
    reviewCount: item.reviewCount || item.numberOfReviews,
    priceLevel: item.priceLevel || item.priceRange,
    address: {
      full: item.address || item.addressObj?.street1,
      city: item.city || item.addressObj?.city,
      country: item.country || item.addressObj?.country
    },
    cuisine: item.cuisine || item.cuisines || [],
    features: item.features || item.amenities || [],
    awards: item.awards || [],
    photos: (item.photos || []).slice(0, 5).map(p => p.url || p),
    url: item.url || item.webUrl,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformYelp(item) {
  return {
    id: item.id || item.businessId || generateId(),
    source: 'yelp',
    name: item.name || item.businessName,
    rating: item.rating,
    reviewCount: item.reviewCount || item.review_count,
    priceLevel: item.price || item.priceRange,
    address: {
      full: item.address || item.location?.display_address?.join(', '),
      street: item.location?.address1,
      city: item.location?.city,
      state: item.location?.state,
      zip: item.location?.zip_code
    },
    phone: item.phone || item.display_phone,
    categories: (item.categories || []).map(c => c.title || c),
    hours: item.hours || item.businessHours,
    photos: (item.photos || []).slice(0, 5),
    isClaimed: item.is_claimed,
    url: item.url,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformBooking(item) {
  return {
    id: item.id || item.hotelId || generateId(),
    source: 'booking',
    name: item.name || item.hotelName,
    type: item.type || item.accommodationType || 'hotel',
    rating: item.rating || item.reviewScore,
    reviewScore: item.reviewScore || item.score,
    reviewCount: item.reviewCount || item.numberOfReviews,
    stars: item.stars || item.starRating,
    price: {
      amount: item.price || item.priceAmount,
      currency: item.currency || 'USD',
      perNight: item.pricePerNight || item.price
    },
    location: {
      address: item.address,
      city: item.city,
      country: item.country,
      lat: item.latitude || item.location?.lat,
      lng: item.longitude || item.location?.lng
    },
    amenities: item.amenities || item.facilities || [],
    photos: (item.photos || []).slice(0, 5).map(p => p.url || p),
    url: item.url,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformZillow(item) {
  return {
    id: item.zpid || item.id || generateId(),
    source: 'zillow',
    address: {
      full: item.address || item.streetAddress,
      street: item.streetAddress,
      city: item.city,
      state: item.state,
      zip: item.zipcode
    },
    price: item.price || item.zestimate,
    zestimate: item.zestimate,
    rentZestimate: item.rentZestimate,
    propertyType: item.homeType || item.propertyType,
    status: item.homeStatus || item.status,
    specs: {
      bedrooms: item.bedrooms || item.beds,
      bathrooms: item.bathrooms || item.baths,
      sqft: item.livingArea || item.sqft,
      lotSize: item.lotSize || item.lotAreaValue,
      yearBuilt: item.yearBuilt
    },
    features: item.resoFacts?.atAGlanceFacts || [],
    priceHistory: item.priceHistory || [],
    taxHistory: item.taxHistory || [],
    photos: (item.photos || item.hiResImageLink || []).slice(0, 5),
    url: item.url || item.hdpUrl,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformCraigslist(item) {
  return {
    id: item.id || item.postId || generateId(),
    source: 'craigslist',
    title: item.title || item.postTitle,
    price: item.price,
    category: item.category || item.section,
    subcategory: item.subcategory,
    location: {
      area: item.location || item.hood,
      city: item.city,
      region: item.region
    },
    description: item.description || item.body,
    attributes: item.attributes || {},
    images: (item.images || item.pics || []).slice(0, 5),
    postedAt: item.datetime || item.postedAt,
    updatedAt: item.updated,
    url: item.url || item.postUrl,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformReddit(item) {
  return {
    id: item.id || item.postId || generateId(),
    source: 'reddit',
    type: item.type || (item.isSelf ? 'text' : 'link'),
    title: item.title,
    text: item.selftext || item.body || item.text,
    subreddit: item.subreddit || item.subredditName,
    author: {
      username: item.author || item.authorName,
      id: item.authorId
    },
    engagement: {
      score: item.score || item.ups - (item.downs || 0),
      upvotes: item.ups,
      downvotes: item.downs,
      comments: item.numComments || item.num_comments,
      awards: item.totalAwards || item.total_awards_received
    },
    flair: item.linkFlair || item.link_flair_text,
    nsfw: item.over18 || item.over_18 || false,
    spoiler: item.spoiler || false,
    url: item.url || `https://reddit.com${item.permalink}`,
    mediaUrl: item.mediaUrl || item.url_overridden_by_dest,
    createdAt: item.created || item.createdUtc,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformFacebook(item) {
  return {
    id: item.id || item.postId || generateId(),
    source: 'facebook',
    type: item.type || 'post',
    text: item.text || item.message || item.content,
    author: {
      name: item.authorName || item.user?.name,
      id: item.authorId || item.user?.id,
      url: item.authorUrl || item.user?.url
    },
    engagement: {
      likes: item.likes || item.likesCount,
      comments: item.comments || item.commentsCount,
      shares: item.shares || item.sharesCount,
      reactions: item.reactions || {}
    },
    media: {
      images: item.images || [],
      videos: item.videos || [],
      links: item.links || []
    },
    hashtags: extractHashtags(item.text || item.message),
    mentions: extractMentions(item.text || item.message),
    timestamp: item.time || item.timestamp || item.createdAt,
    url: item.url || item.postUrl,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

function transformGooglePlaces(item) {
  return {
    id: item.place_id || item.placeId || generateId(),
    source: 'google-places',
    name: item.name,
    rating: item.rating,
    reviewCount: item.user_ratings_total || item.reviewCount,
    priceLevel: item.price_level || item.priceLevel,
    address: item.formatted_address || item.address,
    phone: item.formatted_phone_number || item.phone,
    website: item.website,
    types: item.types || [],
    location: {
      lat: item.geometry?.location?.lat || item.lat,
      lng: item.geometry?.location?.lng || item.lng
    },
    hours: item.opening_hours || item.hours,
    photos: (item.photos || []).slice(0, 5).map(p => p.photo_reference || p),
    reviews: (item.reviews || []).slice(0, 5).map(r => ({
      text: r.text,
      rating: r.rating,
      author: r.author_name
    })),
    url: item.url,
    scrapedAt: item.scrapedAt || new Date().toISOString()
  };
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function generateId() {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function extractDomain(url) {
  if (!url) return null;
  try {
    return new URL(url).hostname;
  } catch {
    return null;
  }
}

function extractHashtags(text) {
  if (!text) return [];
  const matches = text.match(/#[\w]+/g) || [];
  return matches.map(tag => tag.toLowerCase());
}

function extractMentions(text) {
  if (!text) return [];
  const matches = text.match(/@[\w]+/g) || [];
  return matches.map(mention => mention.toLowerCase());
}

// ============================================
// MAIN INTEGRATION FUNCTIONS
// ============================================

/**
 * Integrate data from an Apify actor run
 * @param {Object} options - Integration options
 * @returns {Object} Transformed data ready for RAG/memory
 */
export async function integrateActorData(options) {
  const {
    actorId,
    runId = 'latest',
    datasetId,
    data,
    memorizeFields,
    template,
    maxItems = 1000
  } = options;

  // Get actor config
  const actorConfig = SUPPORTED_ACTORS[actorId];
  if (!actorConfig && !data) {
    throw new Error(`Unsupported actor: ${actorId}. Supported: ${Object.keys(SUPPORTED_ACTORS).join(', ')}`);
  }

  // Use template defaults if provided
  const templateConfig = template ? USE_CASE_TEMPLATES[template] : null;
  const fields = memorizeFields || templateConfig?.memorizeFields || actorConfig?.defaultFields || [];

  // Transform data
  const transformFn = actorConfig?.transform || ((item) => item);
  const transformedData = (data || []).slice(0, maxItems).map(item => {
    const transformed = transformFn(item);

    // Filter to memorize fields if specified
    if (fields.length > 0) {
      const filtered = {};
      for (const field of fields) {
        if (transformed[field] !== undefined) {
          filtered[field] = transformed[field];
        }
      }
      return {
        ...filtered,
        _meta: {
          originalId: transformed.id,
          source: transformed.source,
          scrapedAt: transformed.scrapedAt
        }
      };
    }

    return transformed;
  });

  return {
    success: true,
    actorId,
    template: template || null,
    totalItems: transformedData.length,
    fields: fields.length > 0 ? fields : Object.keys(transformedData[0] || {}),
    data: transformedData
  };
}

/**
 * Get template configuration
 * @param {string} templateId - Template ID
 * @returns {Object} Template configuration
 */
export function getTemplate(templateId) {
  const template = USE_CASE_TEMPLATES[templateId];
  if (!template) {
    throw new Error(`Unknown template: ${templateId}. Available: ${Object.keys(USE_CASE_TEMPLATES).join(', ')}`);
  }
  return template;
}

/**
 * List all supported actors
 * @returns {Array} List of supported actors with metadata
 */
export function listSupportedActors() {
  return Object.entries(SUPPORTED_ACTORS).map(([id, config]) => ({
    actorId: id,
    name: config.name,
    category: config.category,
    defaultFields: config.defaultFields
  }));
}

/**
 * List all use case templates
 * @returns {Array} List of templates with metadata
 */
export function listTemplates() {
  return Object.entries(USE_CASE_TEMPLATES).map(([id, config]) => ({
    templateId: id,
    name: config.name,
    description: config.description,
    targetUsers: config.targetUsers,
    suggestedActors: config.suggestedActors
  }));
}
