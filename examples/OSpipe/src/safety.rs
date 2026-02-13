//! Safety gate for content filtering and PII redaction.
//!
//! The safety gate inspects captured content before it enters the
//! ingestion pipeline, detecting and optionally redacting sensitive
//! information such as credit card numbers, SSNs, and custom patterns.

use crate::config::SafetyConfig;

/// Decision made by the safety gate about a piece of content.
#[derive(Debug, Clone, PartialEq)]
pub enum SafetyDecision {
    /// Content is safe to store as-is.
    Allow,
    /// Content is safe after redaction; the redacted version is provided.
    AllowRedacted(String),
    /// Content must not be stored.
    Deny {
        /// Reason for denial.
        reason: String,
    },
}

/// Safety gate that checks content for sensitive information.
pub struct SafetyGate {
    config: SafetyConfig,
}

impl SafetyGate {
    /// Create a new safety gate with the given configuration.
    pub fn new(config: SafetyConfig) -> Self {
        Self { config }
    }

    /// Check content and return a safety decision.
    ///
    /// If PII is detected and redaction is enabled, the content is
    /// returned in redacted form. If custom patterns match and no
    /// redaction is possible, the content is denied.
    pub fn check(&self, content: &str) -> SafetyDecision {
        let mut redacted = content.to_string();
        let mut was_redacted = false;

        // Credit card redaction
        if self.config.credit_card_redaction {
            let (new_text, found) = redact_credit_cards(&redacted);
            if found {
                redacted = new_text;
                was_redacted = true;
            }
        }

        // SSN redaction
        if self.config.ssn_redaction {
            let (new_text, found) = redact_ssns(&redacted);
            if found {
                redacted = new_text;
                was_redacted = true;
            }
        }

        // PII detection (email addresses)
        if self.config.pii_detection {
            let (new_text, found) = redact_emails(&redacted);
            if found {
                redacted = new_text;
                was_redacted = true;
            }
        }

        // Custom patterns: deny if found (custom patterns indicate content
        // that should not be stored at all)
        for pattern in &self.config.custom_patterns {
            if content.contains(pattern.as_str()) {
                return SafetyDecision::Deny {
                    reason: format!("Custom pattern matched: {}", pattern),
                };
            }
        }

        if was_redacted {
            SafetyDecision::AllowRedacted(redacted)
        } else {
            SafetyDecision::Allow
        }
    }

    /// Redact all detected sensitive content and return the cleaned string.
    pub fn redact(&self, content: &str) -> String {
        match self.check(content) {
            SafetyDecision::Allow => content.to_string(),
            SafetyDecision::AllowRedacted(redacted) => redacted,
            SafetyDecision::Deny { .. } => "[REDACTED]".to_string(),
        }
    }
}

/// Detect and redact sequences of 13-16 digits that look like credit card numbers.
///
/// This uses a simple pattern: sequences of digits (with optional spaces or dashes)
/// totaling 13-16 digits are replaced with [CC_REDACTED].
fn redact_credit_cards(text: &str) -> (String, bool) {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut found = false;

    while i < chars.len() {
        // Check if we are at the start of a digit sequence
        if chars[i].is_ascii_digit() {
            let start = i;
            let mut digit_count = 0;

            // Consume digits, spaces, and dashes
            while i < chars.len()
                && (chars[i].is_ascii_digit() || chars[i] == ' ' || chars[i] == '-')
            {
                if chars[i].is_ascii_digit() {
                    digit_count += 1;
                }
                i += 1;
            }

            if (13..=16).contains(&digit_count) {
                result.push_str("[CC_REDACTED]");
                found = true;
            } else {
                // Not a credit card, keep original text
                for c in &chars[start..i] {
                    result.push(*c);
                }
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    (result, found)
}

/// Detect and redact SSN patterns (XXX-XX-XXXX).
fn redact_ssns(text: &str) -> (String, bool) {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut found = false;
    let mut i = 0;

    while i < chars.len() {
        // Check for SSN pattern: 3 digits, dash, 2 digits, dash, 4 digits
        if i + 10 < chars.len() && is_ssn_at(&chars, i) {
            result.push_str("[SSN_REDACTED]");
            found = true;
            i += 11; // Skip the SSN (XXX-XX-XXXX = 11 chars)
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    (result, found)
}

/// Check if an SSN pattern exists at the given position.
fn is_ssn_at(chars: &[char], pos: usize) -> bool {
    if pos + 10 >= chars.len() {
        return false;
    }
    // XXX-XX-XXXX
    chars[pos].is_ascii_digit()
        && chars[pos + 1].is_ascii_digit()
        && chars[pos + 2].is_ascii_digit()
        && chars[pos + 3] == '-'
        && chars[pos + 4].is_ascii_digit()
        && chars[pos + 5].is_ascii_digit()
        && chars[pos + 6] == '-'
        && chars[pos + 7].is_ascii_digit()
        && chars[pos + 8].is_ascii_digit()
        && chars[pos + 9].is_ascii_digit()
        && chars[pos + 10].is_ascii_digit()
}

/// Detect and redact email addresses.
fn redact_emails(text: &str) -> (String, bool) {
    let mut result = String::new();
    let mut found = false;

    // Simple email detection: look for word@word.word patterns
    let words: Vec<&str> = text.split_whitespace().collect();
    for (idx, word) in words.iter().enumerate() {
        if idx > 0 {
            result.push(' ');
        }

        // Strip trailing punctuation for matching but keep for non-email words
        let trimmed = word.trim_end_matches([',', '.', ';', ')']);
        let suffix = &word[trimmed.len()..];

        if is_email_like(trimmed) {
            result.push_str("[EMAIL_REDACTED]");
            result.push_str(suffix);
            found = true;
        } else {
            result.push_str(word);
        }
    }

    (result, found)
}

/// Simple heuristic check for email-like patterns.
fn is_email_like(word: &str) -> bool {
    if let Some(at_pos) = word.find('@') {
        let local = &word[..at_pos];
        let domain = &word[at_pos + 1..];
        !local.is_empty() && domain.contains('.') && domain.len() >= 3
    } else {
        false
    }
}
