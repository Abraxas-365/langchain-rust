pub fn combine_embeddings(embeddings: &[Vec<f64>]) -> Vec<f64> {
    embeddings
        .iter()
        // Initialize a vector with zeros based on the length of the first embedding vector.
        // It's assumed all embeddings have the same dimensions.
        .fold(
            vec![0f64; embeddings[0].len()],
            |mut accumulator, embedding_vec| {
                for (i, &value) in embedding_vec.iter().enumerate() {
                    accumulator[i] += value;
                }
                accumulator
            },
        )
        // Calculate the mean for each element across all embeddings.
        .iter()
        .map(|&sum| sum / embeddings.len() as f64)
        .collect()
}

pub fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude_vec1: f64 = vec1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let magnitude_vec2: f64 = vec2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    dot_product / (magnitude_vec1 * magnitude_vec2)
}

pub fn sum_vectors(vectors: &[Vec<f64>]) -> Vec<f64> {
    let mut sum_vec = vec![0.0; vectors[0].len()];
    for vec in vectors {
        for (i, &value) in vec.iter().enumerate() {
            sum_vec[i] += value;
        }
    }
    sum_vec
}
