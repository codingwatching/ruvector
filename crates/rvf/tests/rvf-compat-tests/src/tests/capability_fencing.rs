//! Test 6: Capability fencing.
//!
//! Without rvf.io, I/O calls must fail deterministically.
//! With rvf.io, read/write must match expected behavior.

use rvf_wasm_api::host::{HostCapabilities, IoCapability, NoIo};
use rvf_wasm_api::io::IoApi;
use rvf_wasm_api::error::ApiError;

#[test]
fn no_capabilities_all_io_denied() {
    let caps = HostCapabilities::minimal(b"fence-seed");
    let backend = NoIo;
    let api = IoApi::new(&caps, &backend);

    let mut buf = [0u8; 32];

    // Any cap_id should fail with InvalidCapability
    for cap_id in 0..10u32 {
        let result = api.read_cap(cap_id, 0, &mut buf);
        assert!(
            matches!(result, Err(ApiError::InvalidCapability)),
            "read on missing cap {} must return InvalidCapability",
            cap_id
        );
    }
}

#[test]
fn read_only_cap_blocks_write() {
    let mut caps = HostCapabilities::minimal(b"ro-seed");
    caps.io_caps.push(IoCapability {
        cap_id: 1,
        name: "read-only-store".into(),
        can_read: true,
        can_write: false,
        max_offset: 0,
    });
    let backend = NoIo;
    let api = IoApi::new(&caps, &backend);

    let result = api.write_cap(1, 0, b"data");
    assert!(matches!(result, Err(ApiError::CapabilityDenied)));
}

#[test]
fn write_only_cap_blocks_read() {
    let mut caps = HostCapabilities::minimal(b"wo-seed");
    caps.io_caps.push(IoCapability {
        cap_id: 2,
        name: "write-only-log".into(),
        can_read: false,
        can_write: true,
        max_offset: 0,
    });
    let backend = NoIo;
    let api = IoApi::new(&caps, &backend);

    let mut buf = [0u8; 32];
    let result = api.read_cap(2, 0, &mut buf);
    assert!(matches!(result, Err(ApiError::CapabilityDenied)));
}

#[test]
fn cap_offset_bounds_enforced() {
    let mut caps = HostCapabilities::minimal(b"bounds-seed");
    caps.io_caps.push(IoCapability {
        cap_id: 3,
        name: "bounded".into(),
        can_read: true,
        can_write: true,
        max_offset: 100,
    });
    let backend = NoIo;
    let api = IoApi::new(&caps, &backend);

    // Requesting beyond max_offset should fail
    let mut buf = [0u8; 32];
    let result = api.read_cap(3, 90, &mut buf);
    assert!(
        matches!(result, Err(ApiError::BufferTooSmall)),
        "read beyond max_offset must fail"
    );
}

#[test]
fn unknown_cap_id_fails() {
    let mut caps = HostCapabilities::minimal(b"unknown-seed");
    caps.io_caps.push(IoCapability {
        cap_id: 5,
        name: "known".into(),
        can_read: true,
        can_write: true,
        max_offset: 0,
    });
    let backend = NoIo;
    let api = IoApi::new(&caps, &backend);

    let mut buf = [0u8; 32];
    // Cap 99 doesn't exist
    assert!(matches!(
        api.read_cap(99, 0, &mut buf),
        Err(ApiError::InvalidCapability)
    ));
}
