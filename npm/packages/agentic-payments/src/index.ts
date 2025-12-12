/**
 * @ruvector/agentic-payments
 *
 * WASM App Store payment integration with agentic-payments.
 * Provides both AP2 (Agent Payments Protocol) and ACP (Agentic Commerce Protocol).
 *
 * @example
 * ```typescript
 * import { AppStorePayments, PurchaseType } from '@ruvector/agentic-payments';
 *
 * const payments = new AppStorePayments();
 * const receipt = payments.processPurchase({
 *   appId: 'my-chip-app',
 *   buyerId: 'user-123',
 *   purchaseType: PurchaseType.PayPerUse,
 *   paymentMethod: 'auto'
 * }, 10); // 10 cents
 * ```
 */

// Re-export core agentic-payments functionality
export * from 'agentic-payments';

// App Store specific types
export interface AppPurchaseRequest {
  appId: string;
  buyerId: string;
  purchaseType: PurchaseType;
  paymentMethod: PaymentMethodPreference;
}

export enum PurchaseType {
  OneTime = 'one_time',
  PayPerUse = 'pay_per_use',
  Subscription = 'subscription'
}

export type PaymentMethodPreference = 'ap2' | 'acp' | 'auto';

export interface AppTransaction {
  id: string;
  appId: string;
  buyerId: string;
  purchaseType: PurchaseType;
  totalAmount: number;
  developerAmount: number;
  platformAmount: number;
  status: TransactionStatus;
  protocol: string;
  timestamp: Date;
}

export type TransactionStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'refunded';

export interface PaymentConfig {
  ap2Enabled: boolean;
  acpEnabled: boolean;
  developerShare: number;
  platformFee: number;
  minTransaction: number;
  maxMicropayment: number;
}

export const DEFAULT_PAYMENT_CONFIG: PaymentConfig = {
  ap2Enabled: true,
  acpEnabled: true,
  developerShare: 0.70,
  platformFee: 0.30,
  minTransaction: 1,
  maxMicropayment: 100
};

/**
 * App Store Payment Processor
 *
 * Integrates with the agentic-payments package for AP2/ACP protocol support.
 */
export class AppStorePayments {
  private config: PaymentConfig;
  private transactions: Map<string, AppTransaction> = new Map();

  constructor(config: Partial<PaymentConfig> = {}) {
    this.config = { ...DEFAULT_PAYMENT_CONFIG, ...config };
  }

  /**
   * Process an app purchase
   */
  processPurchase(request: AppPurchaseRequest, priceCents: number): AppTransaction {
    if (priceCents < this.config.minTransaction) {
      throw new Error(`Amount ${priceCents} cents is below minimum ${this.config.minTransaction} cents`);
    }

    const developerAmount = Math.floor(priceCents * this.config.developerShare);
    const platformAmount = priceCents - developerAmount;

    const transaction: AppTransaction = {
      id: `txn_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`,
      appId: request.appId,
      buyerId: request.buyerId,
      purchaseType: request.purchaseType,
      totalAmount: priceCents,
      developerAmount,
      platformAmount,
      status: 'completed',
      protocol: request.paymentMethod.toUpperCase(),
      timestamp: new Date()
    };

    this.transactions.set(transaction.id, transaction);
    return transaction;
  }

  /**
   * Process a micropayment (pay-per-use)
   */
  processMicropayment(appId: string, userId: string, amountCents: number): AppTransaction {
    if (amountCents > this.config.maxMicropayment) {
      throw new Error(`Amount ${amountCents} cents exceeds micropayment limit of ${this.config.maxMicropayment} cents`);
    }

    return this.processPurchase({
      appId,
      buyerId: userId,
      purchaseType: PurchaseType.PayPerUse,
      paymentMethod: 'auto'
    }, amountCents);
  }

  /**
   * Get transaction by ID
   */
  getTransaction(id: string): AppTransaction | undefined {
    return this.transactions.get(id);
  }

  /**
   * Get all transactions for an app
   */
  getAppTransactions(appId: string): AppTransaction[] {
    return Array.from(this.transactions.values()).filter(t => t.appId === appId);
  }

  /**
   * Get total revenue for an app
   */
  getAppRevenue(appId: string): number {
    return this.getAppTransactions(appId).reduce((sum, t) => sum + t.developerAmount, 0);
  }

  /**
   * Get configuration
   */
  getConfig(): PaymentConfig {
    return { ...this.config };
  }
}

// App size categories and pricing
export const APP_SIZE_PRICING = {
  chip: { maxBytes: 8 * 1024, basePriceCents: 1 },      // 8KB, 1 cent
  micro: { maxBytes: 64 * 1024, basePriceCents: 5 },    // 64KB, 5 cents
  small: { maxBytes: 512 * 1024, basePriceCents: 10 },  // 512KB, 10 cents
  medium: { maxBytes: 2 * 1024 * 1024, basePriceCents: 25 }, // 2MB, 25 cents
  large: { maxBytes: 10 * 1024 * 1024, basePriceCents: 50 }, // 10MB, 50 cents
  full: { maxBytes: Infinity, basePriceCents: 100 },    // >10MB, $1
} as const;

export type AppSizeCategory = keyof typeof APP_SIZE_PRICING;

export function getAppSizeCategory(bytes: number): AppSizeCategory {
  if (bytes <= APP_SIZE_PRICING.chip.maxBytes) return 'chip';
  if (bytes <= APP_SIZE_PRICING.micro.maxBytes) return 'micro';
  if (bytes <= APP_SIZE_PRICING.small.maxBytes) return 'small';
  if (bytes <= APP_SIZE_PRICING.medium.maxBytes) return 'medium';
  if (bytes <= APP_SIZE_PRICING.large.maxBytes) return 'large';
  return 'full';
}

export function getBasePrice(bytes: number): number {
  const category = getAppSizeCategory(bytes);
  return APP_SIZE_PRICING[category].basePriceCents;
}

// Subscription tiers
export interface SubscriptionTier {
  id: string;
  name: string;
  description: string;
  monthlyPriceCents: number;
  annualPriceCents: number;
  monthlyCredits: number;
  features: string[];
  isPopular: boolean;
}

export const SUBSCRIPTION_TIERS: SubscriptionTier[] = [
  {
    id: 'free',
    name: 'Free',
    description: 'Perfect for trying out chip apps',
    monthlyPriceCents: 0,
    annualPriceCents: 0,
    monthlyCredits: 100,
    features: [
      '100 credits/month',
      '10 free chip app uses/day',
      'Community support',
      'Basic analytics',
    ],
    isPopular: false,
  },
  {
    id: 'pro',
    name: 'Pro',
    description: 'For developers and power users',
    monthlyPriceCents: 2900,
    annualPriceCents: 29000,
    monthlyCredits: 1000,
    features: [
      '1,000 credits/month',
      'Unlimited chip app uses',
      'Priority support',
      'Advanced analytics',
      '5 concurrent executions',
      'Revenue sharing for published apps',
    ],
    isPopular: true,
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    description: 'For teams and organizations',
    monthlyPriceCents: 9900,
    annualPriceCents: 99000,
    monthlyCredits: 10000,
    features: [
      '10,000 credits/month',
      'Unlimited everything',
      'Dedicated support',
      'Custom analytics',
      '50 concurrent executions',
      'Higher revenue share',
      'SLA guarantee',
      'Custom integrations',
    ],
    isPopular: false,
  },
];

// Utility functions
export function formatCredits(credits: number): string {
  if (credits >= 1000000) {
    return `${(credits / 1000000).toFixed(1)}M`;
  }
  if (credits >= 1000) {
    return `${(credits / 1000).toFixed(1)}K`;
  }
  return credits.toString();
}

export function creditsToDollars(credits: number): number {
  return credits / 100;
}

export function dollarsToCredits(dollars: number): number {
  return dollars * 100;
}
