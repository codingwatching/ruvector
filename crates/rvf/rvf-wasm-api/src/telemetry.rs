//! RVF v1 Telemetry API: metric emission.
//!
//! `rvf.telemetry.emit(metric_name, value) -> Result<()>`
//!
//! Telemetry is optional and has no global state.
//! The host provides a TelemetrySink at init time.
//! Without a sink, metrics are silently dropped.

use crate::error::ApiResult;
use crate::host::TelemetrySink;

/// Telemetry emitter backed by a host-provided sink.
pub struct TelemetryApi<'a, S: TelemetrySink> {
    sink: &'a S,
    enabled: bool,
}

impl<'a, S: TelemetrySink> TelemetryApi<'a, S> {
    /// Create a new telemetry API.
    pub fn new(sink: &'a S, enabled: bool) -> Self {
        Self { sink, enabled }
    }

    /// Emit a metric. No-op if telemetry is disabled.
    pub fn emit(&self, metric_name: &str, value: f64) -> ApiResult<()> {
        if !self.enabled {
            return Ok(());
        }
        self.sink.emit(metric_name, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::NoTelemetry;

    #[test]
    fn disabled_telemetry_is_noop() {
        let sink = NoTelemetry;
        let api = TelemetryApi::new(&sink, false);
        assert!(api.emit("test.metric", 42.0).is_ok());
    }

    #[test]
    fn enabled_noop_sink_works() {
        let sink = NoTelemetry;
        let api = TelemetryApi::new(&sink, true);
        assert!(api.emit("test.metric", 42.0).is_ok());
    }
}
