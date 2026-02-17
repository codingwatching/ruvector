//! Compatibility report generation.
//!
//! Produces `compat_report.json` with ABI version, pubkey,
//! witness root, and per-test pass/fail results.

/// A single test result in the compatibility report.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub details: String,
}

/// Full compatibility report.
#[derive(Debug, Clone)]
pub struct CompatReport {
    pub abi_version: u32,
    pub pubkey_hex: String,
    pub witness_root_hex: String,
    pub tests: Vec<TestResult>,
}

impl CompatReport {
    pub fn new(abi_version: u32) -> Self {
        Self {
            abi_version,
            pubkey_hex: String::new(),
            witness_root_hex: String::new(),
            tests: Vec::new(),
        }
    }

    pub fn add_result(&mut self, name: &str, passed: bool, details: &str) {
        self.tests.push(TestResult {
            name: String::from(name),
            passed,
            details: String::from(details),
        });
    }

    pub fn all_passed(&self) -> bool {
        self.tests.iter().all(|t| t.passed)
    }

    /// Format as JSON string.
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str(&format!("  \"abi_version\": {},\n", self.abi_version));
        json.push_str(&format!("  \"pubkey\": \"{}\",\n", self.pubkey_hex));
        json.push_str(&format!(
            "  \"witness_root\": \"{}\",\n",
            self.witness_root_hex
        ));
        json.push_str("  \"tests\": [\n");
        for (i, t) in self.tests.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"name\": \"{}\", \"passed\": {}, \"details\": \"{}\"}}",
                t.name, t.passed, t.details
            ));
            if i < self.tests.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("  ]\n");
        json.push('}');
        json
    }
}

/// Convert a byte slice to hex string.
pub fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
