//! CrossRef API Client Demo
//!
//! This example demonstrates how to use the CrossRefClient to fetch
//! scholarly publications and convert them to SemanticVectors.
//!
//! Run with:
//! ```bash
//! cargo run --example crossref_demo
//! ```

use ruvector_data_framework::{CrossRefClient, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    // Create client with polite pool email for better rate limits
    let client = CrossRefClient::new(Some("researcher@university.edu".to_string()));

    println!("=== CrossRef API Client Demo ===\n");

    // Example 1: Search publications by keywords
    println!("1. Searching for 'machine learning' publications...");
    match client.search_works("machine learning", 5).await {
        Ok(vectors) => {
            println!("   Found {} publications", vectors.len());
            if let Some(first) = vectors.first() {
                println!("   First result:");
                println!("     DOI: {}", first.metadata.get("doi").unwrap_or(&"N/A".to_string()));
                println!("     Title: {}", first.metadata.get("title").unwrap_or(&"N/A".to_string()));
                println!("     Citations: {}", first.metadata.get("citation_count").unwrap_or(&"0".to_string()));
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // Example 2: Get a specific work by DOI
    println!("2. Fetching work by DOI (AlphaFold paper)...");
    match client.get_work("10.1038/s41586-021-03819-2").await {
        Ok(Some(vector)) => {
            println!("   Found:");
            println!("     Title: {}", vector.metadata.get("title").unwrap_or(&"N/A".to_string()));
            println!("     Authors: {}", vector.metadata.get("authors").unwrap_or(&"N/A".to_string()));
            println!("     Journal: {}", vector.metadata.get("journal").unwrap_or(&"N/A".to_string()));
            println!("     Citations: {}", vector.metadata.get("citation_count").unwrap_or(&"0".to_string()));
        }
        Ok(None) => println!("   Work not found"),
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // Example 3: Search NSF-funded research
    println!("3. Searching NSF-funded research...");
    match client.search_by_funder("10.13039/100000001", 3).await {
        Ok(vectors) => {
            println!("   Found {} NSF-funded publications", vectors.len());
            for (i, vector) in vectors.iter().enumerate() {
                println!("   {}. {}", i + 1, vector.metadata.get("title").unwrap_or(&"N/A".to_string()));
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // Example 4: Search by subject area
    println!("4. Searching publications in 'computational biology'...");
    match client.search_by_subject("computational biology", 3).await {
        Ok(vectors) => {
            println!("   Found {} publications", vectors.len());
            for vector in vectors {
                println!("     - {}", vector.metadata.get("title").unwrap_or(&"N/A".to_string()));
                println!("       Subjects: {}", vector.metadata.get("subjects").unwrap_or(&"N/A".to_string()));
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // Example 5: Search recent publications
    println!("5. Searching recent 'quantum computing' publications...");
    match client.search_recent("quantum computing", "2024-01-01", 3).await {
        Ok(vectors) => {
            println!("   Found {} recent publications", vectors.len());
            for vector in vectors {
                println!("     - {}", vector.metadata.get("title").unwrap_or(&"N/A".to_string()));
                println!("       Published: {}", vector.timestamp.format("%Y-%m-%d"));
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // Example 6: Search by publication type
    println!("6. Searching for datasets...");
    match client.search_by_type("dataset", Some("climate"), 3).await {
        Ok(vectors) => {
            println!("   Found {} datasets", vectors.len());
            for vector in vectors {
                println!("     - {}", vector.metadata.get("title").unwrap_or(&"N/A".to_string()));
                println!("       Type: {}", vector.metadata.get("type").unwrap_or(&"N/A".to_string()));
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // Example 7: Get citations for a work
    println!("7. Finding papers that cite a specific DOI...");
    match client.get_citations("10.1038/nature12373", 3).await {
        Ok(vectors) => {
            println!("   Found {} citing papers", vectors.len());
            for vector in vectors {
                println!("     - {}", vector.metadata.get("title").unwrap_or(&"N/A".to_string()));
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    println!("=== Demo Complete ===");
    println!("\nNote: All results are converted to SemanticVector format with:");
    println!("  - Embedding vectors (384 dimensions by default)");
    println!("  - Domain: Research");
    println!("  - Rich metadata (DOI, title, abstract, authors, citations, etc.)");
    println!("  - Timestamps for temporal analysis");

    Ok(())
}
