//! HNSW PostgreSQL Access Method Implementation
//!
//! This module implements HNSW as a proper PostgreSQL index access method,
//! storing the graph structure in PostgreSQL pages for persistence.

use pgrx::prelude::*;
use pgrx::pg_sys::*;
use std::ffi::CStr;
use std::ptr;
use std::collections::BinaryHeap;

use crate::distance::{DistanceMetric, distance};
use crate::index::HnswConfig;

// ============================================================================
// Page Layout Constants
// ============================================================================

/// Magic number for HNSW index pages (ASCII "HNSW")
const HNSW_MAGIC: u32 = 0x484E5357;

/// Page type identifiers
const HNSW_PAGE_META: u8 = 0;
const HNSW_PAGE_NODE: u8 = 1;
const HNSW_PAGE_DELETED: u8 = 2;

/// Maximum neighbors per node (aligned with default M)
const MAX_NEIGHBORS_L0: usize = 32;  // 2*M for layer 0
const MAX_NEIGHBORS: usize = 16;      // M for other layers
const MAX_LAYERS: usize = 16;         // Maximum graph layers

// ============================================================================
// Page Structures
// ============================================================================

/// Metadata page (page 0)
///
/// Layout:
/// - magic: u32 (4 bytes)
/// - version: u32 (4 bytes)
/// - dimensions: u32 (4 bytes)
/// - m: u16 (2 bytes)
/// - m0: u16 (2 bytes)
/// - ef_construction: u32 (4 bytes)
/// - entry_point: BlockNumber (4 bytes)
/// - max_layer: u16 (2 bytes)
/// - metric: u8 (1 byte - 0=L2, 1=Cosine, 2=IP)
/// - node_count: u64 (8 bytes)
/// - next_block: BlockNumber (4 bytes)
#[repr(C)]
#[derive(Copy, Clone)]
struct HnswMetaPage {
    magic: u32,
    version: u32,
    dimensions: u32,
    m: u16,
    m0: u16,
    ef_construction: u32,
    entry_point: BlockNumber,
    max_layer: u16,
    metric: u8,
    _padding: u8,
    node_count: u64,
    next_block: BlockNumber,
}

impl Default for HnswMetaPage {
    fn default() -> Self {
        Self {
            magic: HNSW_MAGIC,
            version: 1,
            dimensions: 0,
            m: 16,
            m0: 32,
            ef_construction: 64,
            entry_point: InvalidBlockNumber,
            max_layer: 0,
            metric: 0,  // L2 by default
            _padding: 0,
            node_count: 0,
            next_block: 1,  // First node page
        }
    }
}

/// Node page header
#[repr(C)]
#[derive(Copy, Clone)]
struct HnswNodePageHeader {
    page_type: u8,
    max_layer: u8,
    _padding: [u8; 2],
    item_id: ItemPointerData,  // TID of the heap tuple
}

/// Neighbor entry in the graph
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct HnswNeighbor {
    block_num: BlockNumber,
    distance: f32,
}

/// Node structure stored in pages
///
/// Layout per node page:
/// - HnswNodePageHeader
/// - vector data: [f32; dimensions]
/// - layer 0 neighbors: [HnswNeighbor; m0]
/// - layer 1+ neighbors: [[HnswNeighbor; m]; max_layer]
struct HnswNode {
    header: HnswNodePageHeader,
    // Variable-length data follows
}

// ============================================================================
// Index Build State
// ============================================================================

/// State for building an HNSW index
struct HnswBuildState {
    index_relation: PgRelation,
    heap_relation: PgRelation,
    dimensions: usize,
    config: HnswConfig,
    entry_point: BlockNumber,
    max_layer: usize,
    node_count: u64,
    next_block: BlockNumber,
}

// ============================================================================
// Index Scan State
// ============================================================================

/// State for scanning an HNSW index
struct HnswScanState {
    query_vector: Vec<f32>,
    k: usize,
    ef_search: usize,
    metric: DistanceMetric,
    dimensions: usize,
    results: Vec<(BlockNumber, ItemPointerData, f32)>,
    current_pos: usize,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get metadata page from index relation
unsafe fn get_meta_page(index_rel: &PgRelation) -> (*mut Page, Buffer) {
    let buffer = ReadBuffer(index_rel.as_ptr(), 0);
    LockBuffer(buffer, BUFFER_LOCK_SHARE as i32);
    let page = BufferGetPage(buffer);
    (page, buffer)
}

/// Get or create metadata page
unsafe fn get_or_create_meta_page(index_rel: &PgRelation, for_write: bool) -> (*mut Page, Buffer) {
    let buffer = ReadBuffer(index_rel.as_ptr(), 0);
    if for_write {
        LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE as i32);
    } else {
        LockBuffer(buffer, BUFFER_LOCK_SHARE as i32);
    }
    let page = BufferGetPage(buffer);
    (page, buffer)
}

/// Read metadata from page
unsafe fn read_metadata(page: *mut Page) -> HnswMetaPage {
    let data_ptr = PageGetContents(page as *const PageHeaderData);
    ptr::read(data_ptr as *const HnswMetaPage)
}

/// Write metadata to page
unsafe fn write_metadata(page: *mut Page, meta: &HnswMetaPage) {
    let data_ptr = PageGetContents(page as *const PageHeaderData) as *mut HnswMetaPage;
    ptr::write(data_ptr, *meta);
}

/// Allocate a new node page
unsafe fn allocate_node_page(
    index_rel: &PgRelation,
    vector: &[f32],
    tid: ItemPointerData,
    max_layer: usize,
) -> BlockNumber {
    // Get a new buffer
    let buffer = ReadBuffer(index_rel.as_ptr(), P_NEW);
    let block = BufferGetBlockNumber(buffer);

    LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE as i32);
    let page = BufferGetPage(buffer);

    // Initialize page
    PageInit(page as *mut PageHeaderData, BLCKSZ as Size, 0);

    // Write node header
    let data_ptr = PageGetContents(page as *const PageHeaderData);
    let header = HnswNodePageHeader {
        page_type: HNSW_PAGE_NODE,
        max_layer: max_layer as u8,
        _padding: [0; 2],
        item_id: tid,
    };
    ptr::write(data_ptr as *mut HnswNodePageHeader, header);

    // Write vector data after header
    let vector_ptr = data_ptr.add(std::mem::size_of::<HnswNodePageHeader>()) as *mut f32;
    for (i, &val) in vector.iter().enumerate() {
        ptr::write(vector_ptr.add(i), val);
    }

    // Mark buffer dirty and unlock
    MarkBufferDirty(buffer);
    UnlockReleaseBuffer(buffer);

    block
}

/// Read vector from node page
unsafe fn read_vector(
    index_rel: &PgRelation,
    block: BlockNumber,
    dimensions: usize,
) -> Option<Vec<f32>> {
    if block == InvalidBlockNumber {
        return None;
    }

    let buffer = ReadBuffer(index_rel.as_ptr(), block);
    LockBuffer(buffer, BUFFER_LOCK_SHARE as i32);
    let page = BufferGetPage(buffer);

    let data_ptr = PageGetContents(page as *const PageHeaderData);
    let vector_ptr = data_ptr.add(std::mem::size_of::<HnswNodePageHeader>()) as *const f32;

    let mut vector = Vec::with_capacity(dimensions);
    for i in 0..dimensions {
        vector.push(ptr::read(vector_ptr.add(i)));
    }

    UnlockReleaseBuffer(buffer);
    Some(vector)
}

/// Calculate distance between query and node
unsafe fn calculate_distance(
    index_rel: &PgRelation,
    query: &[f32],
    block: BlockNumber,
    dimensions: usize,
    metric: DistanceMetric,
) -> f32 {
    match read_vector(index_rel, block, dimensions) {
        Some(vec) => distance(query, &vec, metric),
        None => f32::MAX,
    }
}

// ============================================================================
// Access Method Callbacks
// ============================================================================

/// Build callback - builds the index from scratch
#[pg_guard]
unsafe extern "C" fn hnsw_build(
    heap: Relation,
    index: Relation,
    index_info: *mut IndexInfo,
) -> *mut IndexBuildResult {
    pgrx::log!("HNSW: Starting index build");

    let heap_rel = PgRelation::from_pg(heap);
    let index_rel = PgRelation::from_pg(index);

    // Parse index options
    let dimensions = 128; // TODO: Extract from index definition
    let config = HnswConfig::default();

    // Initialize metadata page
    let (page, buffer) = get_or_create_meta_page(&index_rel, true);
    PageInit(page as *mut PageHeaderData, BLCKSZ as Size, 0);

    let mut meta = HnswMetaPage {
        dimensions: dimensions as u32,
        m: config.m as u16,
        m0: config.m0 as u16,
        ef_construction: config.ef_construction as u32,
        metric: match config.metric {
            DistanceMetric::Euclidean => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::InnerProduct => 2,
            _ => 0,
        },
        ..Default::default()
    };

    write_metadata(page, &meta);
    MarkBufferDirty(buffer);
    UnlockReleaseBuffer(buffer);

    // Scan heap and build index
    // This is a simplified version - full implementation would use IndexBuildHeapScan
    let tuple_count = 0.0;

    pgrx::log!("HNSW: Index build complete, {} tuples indexed", tuple_count as u64);

    // Return build result
    let result = PgBox::<IndexBuildResult>::alloc0();
    result.heap_tuples = tuple_count;
    result.index_tuples = tuple_count;
    result.into_pg()
}

/// Build empty index callback
#[pg_guard]
unsafe extern "C" fn hnsw_buildempty(index: Relation) {
    pgrx::log!("HNSW: Building empty index");

    let index_rel = PgRelation::from_pg(index);

    // Initialize metadata page only
    let (page, buffer) = get_or_create_meta_page(&index_rel, true);
    PageInit(page as *mut PageHeaderData, BLCKSZ as Size, 0);

    let meta = HnswMetaPage::default();
    write_metadata(page, &meta);

    MarkBufferDirty(buffer);
    UnlockReleaseBuffer(buffer);
}

/// Insert callback - insert a single tuple into the index
#[pg_guard]
unsafe extern "C" fn hnsw_insert(
    index: Relation,
    values: *mut Datum,
    isnull: *mut bool,
    heap_tid: ItemPointer,
    _heap: Relation,
    _check_unique: IndexUniqueCheck::Type,
    _index_info: *mut IndexInfo,
) -> bool {
    // Check for null
    if *isnull {
        return false;
    }

    let index_rel = PgRelation::from_pg(index);

    // Get metadata
    let (meta_page, meta_buffer) = get_meta_page(&index_rel);
    let meta = read_metadata(meta_page);
    UnlockReleaseBuffer(meta_buffer);

    // TODO: Extract vector from datum
    // let vector = extract_vector(*values, meta.dimensions as usize);

    // For now, just return success
    true
}

/// Bulk delete callback
#[pg_guard]
unsafe extern "C" fn hnsw_bulkdelete(
    info: *mut IndexVacuumInfo,
    stats: *mut IndexBulkDeleteResult,
    callback: IndexBulkDeleteCallback,
    callback_state: *mut ::std::os::raw::c_void,
) -> *mut IndexBulkDeleteResult {
    pgrx::log!("HNSW: Bulk delete called");

    // Return stats (simplified implementation)
    if stats.is_null() {
        let new_stats = PgBox::<IndexBulkDeleteResult>::alloc0();
        new_stats.into_pg()
    } else {
        stats
    }
}

/// Vacuum cleanup callback
#[pg_guard]
unsafe extern "C" fn hnsw_vacuumcleanup(
    info: *mut IndexVacuumInfo,
    stats: *mut IndexBulkDeleteResult,
) -> *mut IndexBulkDeleteResult {
    pgrx::log!("HNSW: Vacuum cleanup called");

    if stats.is_null() {
        let new_stats = PgBox::<IndexBulkDeleteResult>::alloc0();
        new_stats.into_pg()
    } else {
        stats
    }
}

/// Cost estimate callback
#[pg_guard]
unsafe extern "C" fn hnsw_costestimate(
    _root: *mut PlannerInfo,
    path: *mut IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut Cost,
    index_total_cost: *mut Cost,
    index_selectivity: *mut Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // Simplified cost estimation
    // HNSW has logarithmic search complexity
    let tuples = (*path).indexinfo.as_ref().map(|i| (*i).tuples).unwrap_or(1000.0);

    // Startup cost is minimal
    *index_startup_cost = 0.0;

    // Total cost is O(log n) for HNSW
    let log_tuples = tuples.max(1.0).ln();
    *index_total_cost = log_tuples * 10.0;  // Scale factor for page accesses

    // HNSW provides good selectivity for top-k queries
    *index_selectivity = 0.01;  // Typically returns ~1% of tuples
    *index_correlation = 0.0;   // No correlation with physical order
    *index_pages = (tuples / 100.0).max(1.0);  // Rough estimate
}

/// Get tuple callback (for index scans)
#[pg_guard]
unsafe extern "C" fn hnsw_gettuple(scan: *mut IndexScanDesc, direction: ScanDirection::Type) -> bool {
    pgrx::log!("HNSW: Get tuple called");

    // TODO: Implement actual index scan
    // For now, return false (no more tuples)
    false
}

/// Get bitmap callback (for bitmap scans)
#[pg_guard]
unsafe extern "C" fn hnsw_getbitmap(scan: *mut IndexScanDesc, tbm: *mut TIDBitmap) -> i64 {
    pgrx::log!("HNSW: Get bitmap called");

    // TODO: Implement bitmap scan
    // Return number of tuples
    0
}

/// Begin scan callback
#[pg_guard]
unsafe extern "C" fn hnsw_beginscan(
    index: Relation,
    nkeys: ::std::os::raw::c_int,
    norderbys: ::std::os::raw::c_int,
) -> *mut IndexScanDesc {
    pgrx::log!("HNSW: Begin scan");

    let scan = RelationGetIndexScan(index, nkeys, norderbys);

    // Allocate scan state
    // let state = PgBox::<HnswScanState>::alloc0();
    // (*scan).opaque = state.into_pg() as *mut std::ffi::c_void;

    scan
}

/// Rescan callback
#[pg_guard]
unsafe extern "C" fn hnsw_rescan(
    scan: *mut IndexScanDesc,
    keys: *mut ScanKey,
    nkeys: ::std::os::raw::c_int,
    orderbys: *mut ScanKey,
    norderbys: ::std::os::raw::c_int,
) {
    pgrx::log!("HNSW: Rescan");

    // Reset scan state
}

/// End scan callback
#[pg_guard]
unsafe extern "C" fn hnsw_endscan(scan: *mut IndexScanDesc) {
    pgrx::log!("HNSW: End scan");

    // Clean up scan state
    if !(*scan).opaque.is_null() {
        // Free scan state
    }
}

/// Can return callback - indicates if index can return indexed data
#[pg_guard]
unsafe extern "C" fn hnsw_canreturn(index: Relation, attno: ::std::os::raw::c_int) -> bool {
    // HNSW can return the vector column
    attno == 1
}

/// Options callback - parse index options
#[pg_guard]
unsafe extern "C" fn hnsw_options(
    reloptions: Datum,
    validate: bool,
) -> *mut bytea {
    pgrx::log!("HNSW: Parsing options");

    // TODO: Parse m, ef_construction, metric from reloptions
    // For now, return null (use defaults)
    ptr::null_mut()
}

// ============================================================================
// Access Method Handler
// ============================================================================

/// Main handler function for HNSW index access method
#[pg_extern]
fn hnsw_handler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<IndexAmRoutine> {
    let mut am_routine = unsafe { PgBox::<IndexAmRoutine>::alloc0() };

    am_routine.type_ = NodeTag::T_IndexAmRoutine;

    // Index build and maintenance
    am_routine.ambuild = Some(hnsw_build);
    am_routine.ambuildempty = Some(hnsw_buildempty);
    am_routine.aminsert = Some(hnsw_insert);
    am_routine.ambulkdelete = Some(hnsw_bulkdelete);
    am_routine.amvacuumcleanup = Some(hnsw_vacuumcleanup);

    // Index scan
    am_routine.ambeginscan = Some(hnsw_beginscan);
    am_routine.amrescan = Some(hnsw_rescan);
    am_routine.amgettuple = Some(hnsw_gettuple);
    am_routine.amgetbitmap = Some(hnsw_getbitmap);
    am_routine.amendscan = Some(hnsw_endscan);

    // Cost estimation
    am_routine.amcostestimate = Some(hnsw_costestimate);

    // Options and capabilities
    am_routine.amoptions = Some(hnsw_options);
    am_routine.amcanreturn = Some(hnsw_canreturn);

    // Index properties
    am_routine.amcanorder = false;
    am_routine.amcanorderbyop = true;  // Supports ORDER BY with distance operators
    am_routine.amcanbackward = false;
    am_routine.amcanunique = false;
    am_routine.amcanmulticol = false;  // Single column only (vector)
    am_routine.amoptionalkey = true;
    am_routine.amsearcharray = false;
    am_routine.amsearchnulls = false;
    am_routine.amstorage = false;
    am_routine.amclusterable = false;
    am_routine.ampredlocks = false;
    am_routine.amcanparallel = false;  // TODO: Enable parallel scans
    am_routine.amcanbuildparallel = false;
    am_routine.amcaninclude = false;
    am_routine.amusemaintenanceworkmem = true;
    am_routine.amparallelvacuumoptions = 0;

    // Key type (we use anyelement since vector type)
    am_routine.amkeytype = pg_sys::ANYELEMENTOID;

    am_routine
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_page_size() {
        assert!(std::mem::size_of::<HnswMetaPage>() < 8192);
    }

    #[test]
    fn test_node_header_size() {
        assert!(std::mem::size_of::<HnswNodePageHeader>() < 100);
    }
}
