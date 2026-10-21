use std::collections::HashMap;

use serde_json::json;

use crate::{
    schemas::Document,
    text_splitter::{
        MarkdownSplitter, PlainTextSplitter, PlainTextSplitterOptions, SplitterOptions,
        TextSplitter, TextSplitterError, TokenSplitter,
    },
};

const TEXT: &str = "The quick brown fox jumps over the lazy dog. \
    Pack my box with five dozen liquor jugs. \
    How vexingly quick daft zebras jump. \
    Sphinx of black quartz, judge my vow.";

#[test]
fn splitter_options_defaults_and_builders() {
    let o = SplitterOptions::default();
    assert_eq!(o.chunk_size, 512);
    assert_eq!(o.chunk_overlap, 0);
    assert_eq!(o.model_name, "gpt-3.5-turbo");
    assert_eq!(o.encoding_name, "cl100k_base");
    assert!(!o.trim_chunks);

    let o = o
        .with_chunk_size(10)
        .with_chunk_overlap(2)
        .with_model_name("gpt-4")
        .with_encoding_name("p50k_base")
        .with_trim_chunks(true);
    assert_eq!(
        (o.chunk_size, o.chunk_overlap, o.trim_chunks),
        (10, 2, true)
    );
    assert_eq!(o.model_name, "gpt-4");
    assert_eq!(o.encoding_name, "p50k_base");
}

#[test]
fn tokenizer_lookup_is_case_insensitive() {
    for name in ["cl100k_base", "P50K_BASE", "r50k_base", "p50k_edit", "gpt2"] {
        assert!(
            SplitterOptions::get_tokenizer_from_str(name).is_some(),
            "{name}"
        );
    }
    assert!(SplitterOptions::get_tokenizer_from_str("nope").is_none());
}

#[tokio::test]
async fn plain_text_splitter_respects_chunk_size() {
    let splitter = PlainTextSplitter::new(
        PlainTextSplitterOptions::default()
            .with_chunk_size(40)
            .with_trim_chunks(true),
    );
    let chunks = splitter.split_text(TEXT).await.unwrap();
    assert!(chunks.len() > 1);
    for c in &chunks {
        assert!(c.chars().count() <= 40, "chunk too long: {c:?}");
        assert_eq!(c.trim(), c);
    }
    // With trimming, re-joining by whitespace recovers every word.
    let words: Vec<_> = TEXT.split_whitespace().collect();
    let rejoined: Vec<_> = chunks.iter().flat_map(|c| c.split_whitespace()).collect();
    assert_eq!(words, rejoined);
}

#[tokio::test]
async fn plain_text_splitter_rejects_overlap_ge_size() {
    let splitter = PlainTextSplitter::new(
        PlainTextSplitterOptions::default()
            .with_chunk_size(10)
            .with_chunk_overlap(10),
    );
    let err = splitter.split_text(TEXT).await.unwrap_err();
    assert!(matches!(err, TextSplitterError::InvalidSplitterOptions));
}

#[tokio::test]
async fn token_splitter_respects_token_budget() {
    let options = SplitterOptions::default()
        .with_chunk_size(8)
        .with_trim_chunks(true);
    let bpe = tiktoken_rs::cl100k_base().unwrap();
    let chunks = TokenSplitter::new(options).split_text(TEXT).await.unwrap();
    assert!(chunks.len() > 1);
    for c in &chunks {
        let n = bpe.encode_ordinary(c).len();
        assert!(n <= 8, "chunk has {n} tokens: {c:?}");
    }
}

#[tokio::test]
async fn token_splitter_small_text_is_single_chunk() {
    let chunks = TokenSplitter::default()
        .split_text("hello world")
        .await
        .unwrap();
    assert_eq!(chunks, vec!["hello world".to_string()]);
}

#[tokio::test]
async fn token_splitter_falls_back_to_model_name() {
    let options = SplitterOptions::default()
        .with_encoding_name("")
        .with_model_name("gpt-4");
    let chunks = TokenSplitter::new(options).split_text(TEXT).await.unwrap();
    assert_eq!(chunks.len(), 1);
}

#[tokio::test]
async fn token_splitter_errors() {
    let bad_encoding = SplitterOptions::default().with_encoding_name("unknown");
    assert!(matches!(
        TokenSplitter::new(bad_encoding).split_text(TEXT).await,
        Err(TextSplitterError::TokenizerNotFound)
    ));

    let bad_model = SplitterOptions::default()
        .with_encoding_name("")
        .with_model_name("definitely-not-a-model");
    assert!(matches!(
        TokenSplitter::new(bad_model).split_text(TEXT).await,
        Err(TextSplitterError::InvalidModel)
    ));

    let bad_overlap = SplitterOptions::default()
        .with_chunk_size(4)
        .with_chunk_overlap(5);
    assert!(matches!(
        TokenSplitter::new(bad_overlap).split_text(TEXT).await,
        Err(TextSplitterError::InvalidSplitterOptions)
    ));
}

#[tokio::test]
async fn markdown_splitter_splits_on_structure() {
    let md = "# Title\n\nIntro paragraph with some words.\n\n## Section A\n\nAlpha alpha alpha alpha.\n\n## Section B\n\nBeta beta beta beta.\n";
    let options = SplitterOptions::default()
        .with_chunk_size(12)
        .with_trim_chunks(true);
    let chunks = MarkdownSplitter::new(options).split_text(md).await.unwrap();
    assert!(chunks.len() >= 3, "{chunks:?}");
    assert!(chunks.iter().any(|c| c.starts_with("## Section A")));
    assert!(chunks.iter().any(|c| c.starts_with("## Section B")));
}

#[tokio::test]
async fn create_documents_propagates_metadata() {
    let splitter = PlainTextSplitter::new(PlainTextSplitterOptions::default().with_chunk_size(20));
    let mut meta = HashMap::new();
    meta.insert("source".to_string(), json!("a.txt"));

    let docs = splitter
        .create_documents(
            &["aaaa bbbb cccc dddd eeee ffff".to_string()],
            std::slice::from_ref(&meta),
        )
        .await
        .unwrap();
    assert!(docs.len() > 1);
    assert!(docs.iter().all(|d| d.metadata == meta));
}

#[tokio::test]
async fn create_documents_without_metadata_and_mismatch() {
    let splitter = PlainTextSplitter::default();
    let docs = splitter
        .create_documents(&["one".to_string(), "two".to_string()], &[])
        .await
        .unwrap();
    assert_eq!(docs.len(), 2);
    assert!(docs.iter().all(|d| d.metadata.is_empty()));

    let err = splitter
        .create_documents(&["one".to_string(), "two".to_string()], &[HashMap::new()])
        .await
        .unwrap_err();
    assert!(matches!(err, TextSplitterError::MetadataTextMismatch));
}

#[tokio::test]
async fn split_documents_keeps_per_document_metadata() {
    let mut m1 = HashMap::new();
    m1.insert("id".to_string(), json!(1));
    let mut m2 = HashMap::new();
    m2.insert("id".to_string(), json!(2));
    let docs = vec![
        Document::new("first doc").with_metadata(m1.clone()),
        Document::new("second doc").with_metadata(m2.clone()),
    ];
    let out = PlainTextSplitter::default()
        .split_documents(&docs)
        .await
        .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].page_content, "first doc");
    assert_eq!(out[0].metadata, m1);
    assert_eq!(out[1].metadata, m2);
}
