//! Helpers for working with embedding vectors.

use super::EmbedderError;

/// Weighted average of `vectors`, scaled to unit length.
///
/// The usual use is embedding a text too long for the model: split it, embed the chunks,
/// then combine them weighted by chunk length into one vector for the whole text.
///
/// ```rust
/// use langchain_rust::embedding::combine_vectors;
///
/// let combined = combine_vectors(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[3.0, 1.0])?;
/// assert!((combined[0] - 0.9486833).abs() < 1e-6);
/// assert!((combined[1] - 0.3162278).abs() < 1e-6);
/// # Ok::<(), langchain_rust::embedding::EmbedderError>(())
/// ```
///
/// Returns an empty vector for no vectors. It's an error when the vectors differ in
/// length, when there isn't one weight per vector, when a weight is negative or not
/// finite, when the weights add up to zero, or when the average is the zero vector (it
/// has no direction to scale).
///
/// Like langchaingo's `embeddings.CombineVectors`, which takes integer weights.
pub fn combine_vectors(vectors: &[Vec<f64>], weights: &[f64]) -> Result<Vec<f64>, EmbedderError> {
    let Some(first) = vectors.first() else {
        return Ok(Vec::new());
    };
    let invalid = |msg: String| Err(EmbedderError::InvalidVectors(msg));
    if weights.len() != vectors.len() {
        return invalid(format!(
            "got {} weights for {} vectors",
            weights.len(),
            vectors.len()
        ));
    }
    if let Some(v) = vectors.iter().find(|v| v.len() != first.len()) {
        return invalid(format!(
            "vectors differ in length ({} and {})",
            first.len(),
            v.len()
        ));
    }
    if let Some(w) = weights.iter().find(|w| !w.is_finite() || **w < 0.0) {
        return invalid(format!("weights must be finite and not negative, got {w}"));
    }
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum == 0.0 {
        return invalid("weights add up to zero".to_string());
    }

    let mut average = vec![0.0; first.len()];
    for (vector, weight) in vectors.iter().zip(weights) {
        for (sum, value) in average.iter_mut().zip(vector) {
            *sum += value * weight;
        }
    }
    let norm = average.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return invalid("the weighted average is the zero vector".to_string());
    }
    // Dividing by the weight sum is folded into the normalization.
    Ok(average.into_iter().map(|x| x / norm).collect())
}

/// Cosine similarity of two vectors, from -1 to 1; 0 when either is all zeros.
///
/// Vectors of different lengths are compared over the shorter length.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denominator = norm(a) * norm(b);
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid(result: Result<Vec<f64>, EmbedderError>) -> String {
        match result {
            Err(EmbedderError::InvalidVectors(msg)) => msg,
            other => panic!("expected InvalidVectors, got {other:?}"),
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-9, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn combine_vectors_is_the_normalized_weighted_mean() {
        // (3*(1,0) + 1*(0,1)) / 4 = (0.75, 0.25), normalized.
        let n = (0.75f64.powi(2) + 0.25f64.powi(2)).sqrt();
        assert_close(
            &combine_vectors(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[3.0, 1.0]).unwrap(),
            &[0.75 / n, 0.25 / n],
        );
        // Equal weights: the direction of the plain mean.
        let c = combine_vectors(&[vec![2.0, 0.0, 0.0], vec![0.0, 0.0, 4.0]], &[1.0, 1.0]).unwrap();
        assert_close(&c, &[1.0 / 5f64.sqrt(), 0.0, 2.0 / 5f64.sqrt()]);
        // A zero weight ignores its vector.
        assert_close(
            &combine_vectors(&[vec![3.0, 4.0], vec![-9.0, 1.0]], &[2.0, 0.0]).unwrap(),
            &[0.6, 0.8],
        );
        // One vector: just normalized.
        assert_close(
            &combine_vectors(&[vec![0.0, -5.0]], &[7.0]).unwrap(),
            &[0.0, -1.0],
        );
        assert!(combine_vectors(&[], &[]).unwrap().is_empty());
    }

    #[test]
    fn combine_vectors_rejects_bad_input() {
        assert_eq!(
            invalid(combine_vectors(&[vec![1.0], vec![2.0]], &[1.0])),
            "got 1 weights for 2 vectors"
        );
        assert_eq!(
            invalid(combine_vectors(&[vec![1.0, 2.0], vec![2.0]], &[1.0, 1.0])),
            "vectors differ in length (2 and 1)"
        );
        assert_eq!(
            invalid(combine_vectors(&[vec![1.0], vec![2.0]], &[0.0, 0.0])),
            "weights add up to zero"
        );
        assert!(
            invalid(combine_vectors(&[vec![1.0], vec![2.0]], &[2.0, -1.0]))
                .contains("not negative, got -1")
        );
        assert!(
            invalid(combine_vectors(&[vec![1.0]], &[f64::NAN])).contains("got NaN"),
            "NaN weight"
        );
        assert!(invalid(combine_vectors(&[vec![1.0]], &[f64::INFINITY])).contains("got inf"));
        assert_eq!(
            invalid(combine_vectors(
                &[vec![1.0, 1.0], vec![-1.0, -1.0]],
                &[1.0, 1.0]
            )),
            "the weighted average is the zero vector"
        );
    }

    #[test]
    fn cosine_similarity_values() {
        assert!((cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!((cosine_similarity(&[1.0, 0.0], &[-3.0, 0.0]) + 1.0).abs() < 1e-12);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 5.0]), 0.0);
        assert!((cosine_similarity(&[1.0, 1.0], &[1.0, 0.0]) - 0.5f64.sqrt()).abs() < 1e-12);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    /// Chunk embeddings combined by length point close to the whole text's embedding.
    /// `OLLAMA_EMBED_MODEL=nomic-embed-text cargo test --features ollama --lib
    /// combine_vectors_live -- --ignored`
    #[cfg(feature = "ollama")]
    #[tokio::test]
    #[ignore]
    async fn combine_vectors_live_ollama() {
        use crate::embedding::{Embedder, OllamaEmbedder};

        let model =
            std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".into());
        let embedder = OllamaEmbedder::default().with_model(model);
        let chunks = [
            "Rust is a systems programming language focused on safety and speed.",
            "Its ownership model prevents data races at compile time, without a garbage collector.",
        ];
        let whole = chunks.join(" ");
        let chunk_vectors = embedder
            .embed_documents(&chunks.map(String::from))
            .await
            .unwrap();
        let weights = chunks.map(|c| c.len() as f64);
        let combined = combine_vectors(&chunk_vectors, &weights).unwrap();
        let unrelated = embedder
            .embed_query("A recipe for banana bread with walnuts.")
            .await
            .unwrap();
        let whole = embedder.embed_query(&whole).await.unwrap();

        let norm: f64 = combined.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "{norm}");
        let to_whole = cosine_similarity(&combined, &whole);
        let to_unrelated = cosine_similarity(&combined, &unrelated);
        println!("combined vs whole: {to_whole:.4}, vs unrelated: {to_unrelated:.4}");
        assert!(to_whole > 0.9, "{to_whole}");
        assert!(
            to_whole > to_unrelated + 0.2,
            "{to_whole} vs {to_unrelated}"
        );
    }
}
