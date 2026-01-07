//! Integration tests for Wikipedia Tool
//! 
//! Run with: cargo test --test wikipedia_integration -- --ignored
//! (tests are ignored by default as they require network access)

use langchain_rust::tools::{Tool, WikipediaQuery};
use serde_json::json;

#[tokio::test]
#[ignore]
async fn test_basic_query() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("Rust programming language")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
    assert!(content.contains("Summary:"));
    assert!(!content.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_multiple_results() {
    let wiki = WikipediaQuery::default().with_top_k_results(3);
    let result = wiki.run(json!("Python")).await.unwrap();
    
    // Should have multiple pages
    let page_count = result.matches("Page:").count();
    assert!(page_count > 0);
    assert!(page_count <= 3);
}

#[tokio::test]
#[ignore]
async fn test_content_truncation() {
    let max_length = 200;
    let wiki = WikipediaQuery::default()
        .with_max_doc_content_length(max_length)
        .with_top_k_results(1);
    
    let result = wiki.run(json!("World War II")).await.unwrap();
    
    // Extract summary content
    for line in result.lines() {
        if line.starts_with("Summary:") {
            let summary = &line[8..]; // Skip "Summary: "
            assert!(
                summary.len() <= max_length,
                "Summary length {} exceeds max {}",
                summary.len(),
                max_length
            );
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_spanish_wikipedia() {
    let wiki = WikipediaQuery::with_lang("es");
    let result = wiki.run(json!("España")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
    assert!(!content.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_french_wikipedia() {
    let wiki = WikipediaQuery::with_lang("fr");
    let result = wiki.run(json!("Paris")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
}

#[tokio::test]
#[ignore]
async fn test_german_wikipedia() {
    let wiki = WikipediaQuery::with_lang("de");
    let result = wiki.run(json!("Berlin")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
}

#[tokio::test]
#[ignore]
async fn test_object_input_format() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!({"input": "Albert Einstein"})).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
}

#[tokio::test]
#[ignore]
async fn test_disambiguation_page() {
    let wiki = WikipediaQuery::default().with_top_k_results(2);
    let result = wiki.run(json!("Rust")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    
    // Should contain multiple meanings of "Rust"
    assert!(content.contains("Page:"));
    let page_count = content.matches("Page:").count();
    assert!(page_count >= 2);
}

#[tokio::test]
#[ignore]
async fn test_scientific_topic() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("Quantum mechanics")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
    assert!(content.len() > 100); // Should have substantial content
}

#[tokio::test]
#[ignore]
async fn test_historical_figure() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("Marie Curie")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
    assert!(content.contains("Marie Curie"));
}

#[tokio::test]
#[ignore]
async fn test_geographical_location() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("Mount Everest")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
}

#[tokio::test]
#[ignore]
async fn test_company_information() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("Microsoft")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
}

#[tokio::test]
#[ignore]
async fn test_nonexistent_query() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("xyzabc999nonexistentpage123")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    // Should handle gracefully, either with "No results found" or empty
    assert!(
        content.contains("No results") || content.is_empty() || content.contains("Page:")
    );
}

#[tokio::test]
#[ignore]
async fn test_unicode_query() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("東京")).await; // Tokyo in Japanese
    
    assert!(result.is_ok());
}

#[tokio::test]
#[ignore]
async fn test_special_characters() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("C++")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
}

#[tokio::test]
#[ignore]
async fn test_long_query() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!(
        "history of computer science and artificial intelligence development"
    )).await;
    
    assert!(result.is_ok());
}

#[tokio::test]
#[ignore]
async fn test_case_insensitive() {
    let wiki = WikipediaQuery::default();
    let result1 = wiki.run(json!("RUST PROGRAMMING")).await.unwrap();
    let result2 = wiki.run(json!("rust programming")).await.unwrap();
    
    // Both should return results (Wikipedia handles case)
    assert!(result1.contains("Page:"));
    assert!(result2.contains("Page:"));
}

#[tokio::test]
#[ignore]
#[allow(dead_code)]
async fn test_concurrent_queries() {
    let queries = vec![
        "Rust programming",
        "Python programming",
        "JavaScript",
        "TypeScript",
    ];
    
    let mut handles = vec![];
    
    for query in queries {
        let handle = tokio::spawn(async move {
            let wiki_clone = WikipediaQuery::default();
            wiki_clone.run(json!(query)).await.map_err(|e| e.to_string())
        });
        handles.push(handle);
    }
    
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}

#[tokio::test]
#[ignore]
async fn test_numerical_query() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("42 (number)")).await;
    
    assert!(result.is_ok());
}

#[tokio::test]
#[ignore]
async fn test_acronym_query() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("NASA")).await;
    
    assert!(result.is_ok());
    let content = result.unwrap();
    assert!(content.contains("Page:"));
}

#[tokio::test]
#[ignore]
async fn test_tool_name_and_description() {
    let wiki = WikipediaQuery::default();
    
    assert_eq!(wiki.name(), "wikipedia-api");
    
    let description = wiki.description();
    assert!(description.contains("Wikipedia"));
    assert!(description.contains("search query"));
    assert!(!description.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_performance_benchmark() {
    use std::time::Instant;
    
    let wiki = WikipediaQuery::default().with_top_k_results(1);
    let start = Instant::now();
    
    let result = wiki.run(json!("Computer")).await;
    let duration = start.elapsed();
    
    assert!(result.is_ok());
    println!("Query took: {:?}", duration);
    
    // Should complete in reasonable time (adjust based on network)
    assert!(duration.as_secs() < 10);
}

#[tokio::test]
#[ignore]
async fn test_builder_pattern_chain() {
    let wiki = WikipediaQuery::default()
        .with_top_k_results(2)
        .with_max_doc_content_length(1000);
    
    assert_eq!(wiki.options.top_k_results, 2);
    assert_eq!(wiki.options.max_doc_content_length, 1000);
    
    let result = wiki.run(json!("Artificial Intelligence")).await;
    assert!(result.is_ok());
}

#[tokio::test]
#[ignore]
async fn test_empty_result_handling() {
    let wiki = WikipediaQuery::default();
    // Query that might return no results
    let result = wiki.run(json!("asdfghjklqwertyuiop123456789")).await;
    
    // Should handle gracefully, not panic
    assert!(result.is_ok());
}

#[tokio::test]
#[ignore]
async fn test_multiple_languages_same_topic() {
    let queries = vec![
        (WikipediaQuery::with_lang("en"), "Artificial Intelligence"),
        (WikipediaQuery::with_lang("es"), "Inteligencia Artificial"),
        (WikipediaQuery::with_lang("fr"), "Intelligence Artificielle"),
        (WikipediaQuery::with_lang("de"), "Künstliche Intelligenz"),
    ];
    
    for (wiki, query) in queries {
        let result = wiki.run(json!(query)).await;
        assert!(result.is_ok(), "Failed for query: {}", query);
        let content = result.unwrap();
        assert!(content.contains("Page:"), "No page found for: {}", query);
    }
}