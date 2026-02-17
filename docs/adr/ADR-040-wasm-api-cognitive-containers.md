# ADR-040: Minimal WASM API for Cognitive Containers, RuVector Migration, and Compatibility Tests

**Status**: Accepted
**Date**: 2026-02-17
**Deciders**: RuVector / RVF maintainers
**Supersedes**: None
**Related**: ADR-030 (RVF Cognitive Container), ADR-032 (RVF WASM Integration), ADR-039 (RVF Solver WASM AGI Integration)

---

## Context

RVF cognitive containers must run identically across browser, edge, and appliance targets. We need a small, stable WASM surface, a clear migration path for existing ruvector crates/packages, and repeatable tests for identity, audit, and deterministic boot. Public docs should stay vendor-neutral (no container-brand specifics).

Existing WASM crates (`rvf-wasm`, `ruvector-wasm`, `ruvector-router-wasm`) each define their own API surface. This ADR consolidates them behind a single v1 API that all hosts can target.

## 1) Minimal WASM API Surface (stable v1)

Goal: one tiny, capability-safe interface that works in browsers, WASI, and embedded hosts.

### Namespaces

#### `rvf.core`
- `init(manifest_ptr, len) -> Result<()>` — load RVF manifest & segments
- `tick(input_ptr, len, out_ptr) -> Result<u32>` — single step; returns bytes written
- `seal() -> Result<()>` — finalize epoch; closes current witness frame
- `version() -> u32` — semantic ABI version (e.g., 0x00010000)

#### `rvf.state`
- `get(key_ptr, len, out_ptr) -> u32`
- `put(key_ptr, len, val_ptr, len) -> Result<()>`
- `commit() -> Result<[u8; 32]>` — Merkle root after commit

#### `rvf.crypto`
- `ed25519_pubkey(out_ptr) -> u32`
- `ed25519_sign(msg_ptr, len, sig_out_ptr) -> Result<()>`
- `sha256(data_ptr, len, out_ptr) -> Result<()>`

#### `rvf.io` (capability-gated; optional)
- `read_cap(cap_id, off, len, out_ptr) -> u32`
- `write_cap(cap_id, off, buf_ptr, len) -> Result<()>`

#### `rvf.telemetry` (optional, no globals)
- `emit(metric_ptr, len, val_f64) -> Result<()>`

### Constraints
- No global mutable singletons; host passes capabilities at `init`.
- Deterministic: no wall-clock, no RNG without host-provided seed; all randomness must be DRBG from witness seed.
- Fixed-width types; little-endian; zero allocator assumptions (host may provide `realloc` if needed).
- Streaming friendly: `tick` is the only progress primitive; hosts can drive loops, budgets, and time.

## 2) Migration Steps for Existing RuVector Packages

### Scope
Rust crates and npm packages (Node/Edge/Browser) moving to RVF-WASM microkernel.

### Steps
1. **Crate refactor** — Extract core logic to `no_std` where possible. Introduce a thin `wasm` feature exposing the v1 API bindings. Replace ad-hoc hashing/signing with `rvf.crypto` calls.
2. **State & storage** — Migrate internal state to `rvf.state` (key-scoped, commit-based). Emit `commit()` after meaningful checkpoints to align witness frames.
3. **Determinism audit** — Remove system time, threads, atomics, and non-seeded RNG. Gate any I/O behind `rvf.io` capability checks.
4. **Embeddings/graphs** — Encode indices (HNSW, min-cut metadata) into RVF segments; load via `init`. Stream updates via `tick` with bounded buffers.
5. **TypeScript/Node/browser packages** — Consolidate to `@ruvector/rvf` facade. Node uses WASI host with the same v1 calls. Browser/Edge uses `@ruvector/rvf-wasm` build; feature-flag telemetry. Deprecate legacy entry points; add shims that forward to the v1 API.
6. **CI pipeline** — Build three targets per package: `wasm32-unknown-unknown`, `wasm32-wasi`, native (for tests). Run cross-host conformance (see §3).

### Rename/Docs
- Public docs: say "cognitive container runtime" / "host runtime," not brand-specific container terms.
- Mark old APIs as **Deprecated** with end-of-support date.

## 3) Compatibility Test Suite (must-pass)

Purpose: guarantee identity, auditability, and reproducibility across hosts.

1. **Ed25519 identity** — `ed25519_pubkey()` is stable across boots of the same artifact. Sign/verify known vectors; compare against host verifier.
2. **Witness logs** — Each `tick` appends a frame: inputs hash, state delta hash, output hash. Replay the same input sequence across hosts => identical witness chain root.
3. **Deterministic boot** — Cold boot `init` with fixed manifest & DRBG seed -> identical `commit()` root prior to any `tick`.
4. **State model** — Put/Get/Commit round-trips with randomized (but DRBG-seeded) keys/values. Snapshot/Restore via RVF segments yields the same Merkle root.
5. **Stress & streaming** — Variable-size `tick` inputs (1 KB to 8 MB), back-pressure on out buffer. Must not allocate unbounded memory or diverge.
6. **Capability fencing** — Without `rvf.io`, I/O calls must fail deterministically. With `rvf.io`, read/write must match golden files; hashes recorded in witness.
7. **Cross-target matrix** — Hosts: Browser, WASI, Appliance runtime. Builds: `wasm32-unknown-unknown`, `wasm32-wasi`. Artifacts: same RVF must pass all targets with identical roots.

### Outputs
- `compat_report.json`: ABI version, pubkey, witness root, per-test pass/fail.
- `witness.rvw`: compact witness stream for audit/replay.

## Decision

Adopt the v1 minimal WASM API, migrate ruvector packages to the RVF microkernel via the steps above, and enforce the must-pass compatibility suite in CI before release.

## Consequences

- Smaller, safer surface; easier browser/edge/appliance parity.
- Deterministic runs with portable audit trails.
- One documentation story (vendor-neutral), less environment drift.

## Implementation

- New crate: `rvf-wasm-api` under `crates/rvf/` implements the v1 API surface.
- New test crate: `rvf-compat-tests` under `crates/rvf/tests/` implements the compatibility test suite.
- Migration shim module inside `rvf-wasm-api` forwards legacy calls to v1 API.
- Workspace wiring in both `crates/rvf/Cargo.toml` and root `Cargo.toml`.

## Open Questions

- Should `rvf.io` expose mmap-like windows for zero-copy reads?
- Pluggable DRBGs (HMAC-DRBG vs. ChaCha20-DRBG) under the same witness contract?
- Optional `rvf.scheduler.yield()` for cooperative multitask within `tick`?
