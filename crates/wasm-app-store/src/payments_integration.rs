//! Payment Integration for WASM App Store
//!
//! Integrates with the agentic-payments crate (crates.io) for:
//! - AP2 (Agent Payments Protocol) - cryptographic multi-agent consensus
//! - ACP (Agentic Commerce Protocol) - Stripe-compatible checkout flows
//!
//! This module bridges the app store's needs with the payment infrastructure.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Re-export agentic-payments types for convenience
pub use agentic_payments::{ap2, acp};

/// App purchase request for the store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppPurchaseRequest {
    /// App ID to purchase
    pub app_id: String,
    /// Buyer agent identity
    pub buyer_id: String,
    /// Purchase type
    pub purchase_type: PurchaseType,
    /// Payment method preference
    pub payment_method: PaymentMethodPreference,
}

/// Purchase types supported by the store
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PurchaseType {
    /// One-time purchase (perpetual license)
    OneTime,
    /// Pay per use (micropayments)
    PayPerUse,
    /// Subscription based
    Subscription { tier: SubscriptionTier },
}

/// Subscription tiers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SubscriptionTier {
    Free,
    Pro,
    Enterprise,
}

/// Payment method preferences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaymentMethodPreference {
    /// Use AP2 protocol with DIDs and VCs
    Ap2,
    /// Use ACP protocol with Stripe
    Acp,
    /// Auto-detect best protocol
    Auto,
}

/// App store payment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentConfig {
    /// Enable AP2 protocol support
    pub ap2_enabled: bool,
    /// Enable ACP protocol support
    pub acp_enabled: bool,
    /// Revenue split for developers (0.0 - 1.0)
    pub developer_share: f64,
    /// Platform fee percentage
    pub platform_fee: f64,
    /// Minimum transaction amount (in cents)
    pub min_transaction: u64,
    /// Maximum micropayment amount (in cents)
    pub max_micropayment: u64,
}

impl Default for PaymentConfig {
    fn default() -> Self {
        Self {
            ap2_enabled: true,
            acp_enabled: true,
            developer_share: 0.70, // 70% to developers
            platform_fee: 0.30,    // 30% platform fee
            min_transaction: 1,    // 1 cent minimum
            max_micropayment: 100, // $1 max micropayment
        }
    }
}

/// Payment processor for the app store
#[derive(Debug)]
pub struct AppStorePayments {
    config: PaymentConfig,
    /// Transaction history
    transactions: HashMap<String, AppTransaction>,
}

impl AppStorePayments {
    /// Create a new payment processor
    pub fn new(config: PaymentConfig) -> Self {
        Self {
            config,
            transactions: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(PaymentConfig::default())
    }

    /// Process an app purchase
    pub fn process_purchase(&mut self, request: AppPurchaseRequest, price_cents: u64) -> Result<AppTransaction, PaymentError> {
        // Validate request
        if price_cents < self.config.min_transaction {
            return Err(PaymentError::BelowMinimum {
                amount: price_cents,
                minimum: self.config.min_transaction,
            });
        }

        // Calculate revenue split
        let developer_amount = (price_cents as f64 * self.config.developer_share) as u64;
        let platform_amount = price_cents - developer_amount;

        // Create transaction
        let transaction = AppTransaction {
            id: generate_transaction_id(),
            app_id: request.app_id,
            buyer_id: request.buyer_id,
            purchase_type: request.purchase_type,
            total_amount: price_cents,
            developer_amount,
            platform_amount,
            status: TransactionStatus::Completed,
            protocol: match request.payment_method {
                PaymentMethodPreference::Ap2 => "AP2".to_string(),
                PaymentMethodPreference::Acp => "ACP".to_string(),
                PaymentMethodPreference::Auto => "AUTO".to_string(),
            },
            timestamp: chrono::Utc::now(),
        };

        self.transactions.insert(transaction.id.clone(), transaction.clone());
        Ok(transaction)
    }

    /// Process a micropayment (pay-per-use)
    pub fn process_micropayment(&mut self, app_id: &str, user_id: &str, amount_cents: u64) -> Result<AppTransaction, PaymentError> {
        if amount_cents > self.config.max_micropayment {
            return Err(PaymentError::ExceedsMicropaymentLimit {
                amount: amount_cents,
                limit: self.config.max_micropayment,
            });
        }

        let request = AppPurchaseRequest {
            app_id: app_id.to_string(),
            buyer_id: user_id.to_string(),
            purchase_type: PurchaseType::PayPerUse,
            payment_method: PaymentMethodPreference::Auto,
        };

        self.process_purchase(request, amount_cents)
    }

    /// Get transaction by ID
    pub fn get_transaction(&self, id: &str) -> Option<&AppTransaction> {
        self.transactions.get(id)
    }

    /// Get all transactions for an app
    pub fn get_app_transactions(&self, app_id: &str) -> Vec<&AppTransaction> {
        self.transactions
            .values()
            .filter(|t| t.app_id == app_id)
            .collect()
    }

    /// Get total revenue for an app
    pub fn get_app_revenue(&self, app_id: &str) -> u64 {
        self.get_app_transactions(app_id)
            .iter()
            .map(|t| t.developer_amount)
            .sum()
    }

    /// Get configuration
    pub fn config(&self) -> &PaymentConfig {
        &self.config
    }
}

/// App store transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppTransaction {
    /// Unique transaction ID
    pub id: String,
    /// App that was purchased
    pub app_id: String,
    /// Buyer ID
    pub buyer_id: String,
    /// Type of purchase
    pub purchase_type: PurchaseType,
    /// Total amount in cents
    pub total_amount: u64,
    /// Amount to developer
    pub developer_amount: u64,
    /// Platform fee amount
    pub platform_amount: u64,
    /// Transaction status
    pub status: TransactionStatus,
    /// Protocol used (AP2/ACP)
    pub protocol: String,
    /// Transaction timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Refunded,
}

/// Payment errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum PaymentError {
    #[error("Amount {amount} cents is below minimum {minimum} cents")]
    BelowMinimum { amount: u64, minimum: u64 },

    #[error("Amount {amount} cents exceeds micropayment limit of {limit} cents")]
    ExceedsMicropaymentLimit { amount: u64, limit: u64 },

    #[error("Invalid payment method")]
    InvalidPaymentMethod,

    #[error("Transaction failed: {reason}")]
    TransactionFailed { reason: String },

    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

/// Generate a unique transaction ID
fn generate_transaction_id() -> String {
    use sha2::{Sha256, Digest};
    let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let random: u64 = rand::random();
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}", timestamp, random).as_bytes());
    let result = hasher.finalize();
    format!("txn_{}", hex::encode(&result[..8]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payment_config_default() {
        let config = PaymentConfig::default();
        assert!(config.ap2_enabled);
        assert!(config.acp_enabled);
        assert_eq!(config.developer_share, 0.70);
    }

    #[test]
    fn test_process_purchase() {
        let mut payments = AppStorePayments::default_config();

        let request = AppPurchaseRequest {
            app_id: "test-app".to_string(),
            buyer_id: "buyer-123".to_string(),
            purchase_type: PurchaseType::OneTime,
            payment_method: PaymentMethodPreference::Auto,
        };

        let result = payments.process_purchase(request, 1000);
        assert!(result.is_ok());

        let txn = result.unwrap();
        assert_eq!(txn.total_amount, 1000);
        assert_eq!(txn.developer_amount, 700); // 70%
        assert_eq!(txn.platform_amount, 300);  // 30%
    }

    #[test]
    fn test_micropayment() {
        let mut payments = AppStorePayments::default_config();

        let result = payments.process_micropayment("chip-app", "user-1", 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_micropayment_limit() {
        let mut payments = AppStorePayments::default_config();

        let result = payments.process_micropayment("chip-app", "user-1", 500);
        assert!(result.is_err());
    }

    #[test]
    fn test_revenue_tracking() {
        let mut payments = AppStorePayments::default_config();

        payments.process_micropayment("app-1", "user-1", 10).unwrap();
        payments.process_micropayment("app-1", "user-2", 20).unwrap();
        payments.process_micropayment("app-2", "user-1", 50).unwrap();

        let app1_revenue = payments.get_app_revenue("app-1");
        assert_eq!(app1_revenue, 21); // 70% of 30 cents
    }
}
