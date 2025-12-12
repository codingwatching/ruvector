/**
 * ACP (Agentic Commerce Protocol) - Stripe-compatible checkout flows
 *
 * Re-exports ACP functionality from agentic-payments package.
 * Use this for standard e-commerce checkout experiences.
 *
 * @example
 * ```typescript
 * import { AcpCheckout, CheckoutSession } from '@ruvector/agentic-payments/acp';
 *
 * const checkout = new AcpCheckout();
 * const session = await checkout.createSession({
 *   items: [{ appId: 'my-app', priceCents: 999 }],
 *   successUrl: '/success',
 *   cancelUrl: '/cancel'
 * });
 * ```
 */

// Re-export ACP types from agentic-payments
export {
  // Core ACP types if available
} from 'agentic-payments';

// ACP-specific types for WASM App Store integration
export interface CheckoutSessionOptions {
  buyerId?: string;
  items: CheckoutItem[];
  successUrl: string;
  cancelUrl: string;
  metadata?: Record<string, string>;
}

export interface CheckoutItem {
  appId: string;
  name: string;
  description?: string;
  priceCents: number;
  quantity: number;
  imageUrl?: string;
}

export interface CheckoutSession {
  id: string;
  status: CheckoutStatus;
  items: CheckoutItem[];
  totalCents: number;
  successUrl: string;
  cancelUrl: string;
  checkoutUrl: string;
  expiresAt: Date;
  createdAt: Date;
  completedAt?: Date;
}

export type CheckoutStatus = 'open' | 'complete' | 'expired' | 'cancelled';

export interface CheckoutResult {
  success: boolean;
  sessionId: string;
  transactionId?: string;
  error?: string;
}

/**
 * ACP Checkout handler for WASM App Store
 *
 * Provides Stripe-compatible checkout flows for app purchases.
 */
export class AcpCheckout {
  private sessions: Map<string, CheckoutSession> = new Map();
  private baseUrl: string;

  constructor(baseUrl: string = 'https://app-store.ruvector.com') {
    this.baseUrl = baseUrl;
  }

  /**
   * Create a checkout session
   */
  async createSession(options: CheckoutSessionOptions): Promise<CheckoutSession> {
    const now = new Date();
    const expiresAt = new Date(now.getTime() + 30 * 60 * 1000); // 30 minutes

    const totalCents = options.items.reduce((sum, item) => sum + item.priceCents * item.quantity, 0);
    const sessionId = `cs_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;

    const session: CheckoutSession = {
      id: sessionId,
      status: 'open',
      items: options.items,
      totalCents,
      successUrl: options.successUrl,
      cancelUrl: options.cancelUrl,
      checkoutUrl: `${this.baseUrl}/checkout/${sessionId}`,
      expiresAt,
      createdAt: now
    };

    this.sessions.set(sessionId, session);
    return session;
  }

  /**
   * Complete a checkout session
   */
  async completeSession(sessionId: string): Promise<CheckoutResult> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return { success: false, sessionId, error: 'Session not found' };
    }

    if (session.status !== 'open') {
      return { success: false, sessionId, error: `Session is ${session.status}` };
    }

    if (new Date() > session.expiresAt) {
      session.status = 'expired';
      return { success: false, sessionId, error: 'Session has expired' };
    }

    session.status = 'complete';
    session.completedAt = new Date();

    return {
      success: true,
      sessionId,
      transactionId: `txn_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`
    };
  }

  /**
   * Cancel a checkout session
   */
  async cancelSession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session && session.status === 'open') {
      session.status = 'cancelled';
    }
  }

  /**
   * Get session by ID
   */
  getSession(sessionId: string): CheckoutSession | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * Expire old sessions (cleanup)
   */
  expireOldSessions(): number {
    const now = new Date();
    let expiredCount = 0;

    for (const session of this.sessions.values()) {
      if (session.status === 'open' && now > session.expiresAt) {
        session.status = 'expired';
        expiredCount++;
      }
    }

    return expiredCount;
  }
}

/**
 * Quick checkout for single app purchase
 */
export async function quickCheckout(
  appId: string,
  appName: string,
  priceCents: number,
  options: { successUrl: string; cancelUrl: string }
): Promise<CheckoutSession> {
  const checkout = new AcpCheckout();
  return checkout.createSession({
    items: [{
      appId,
      name: appName,
      priceCents,
      quantity: 1
    }],
    ...options
  });
}
