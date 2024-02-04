pub fn batch_texts(texts: &[String], batch_size: usize) -> Vec<Vec<String>> {
    let mut batched_texts: Vec<Vec<String>> = Vec::with_capacity(texts.len());

    for text in texts {
        let rune_text: Vec<char> = text.chars().collect();
        let mut batched_text: Vec<String> = Vec::new();

        let mut j = 0;
        while j < rune_text.len() {
            if j + batch_size >= rune_text.len() {
                batched_text.push(rune_text[j..].iter().collect());
                break;
            }

            batched_text.push(rune_text[j..j + batch_size].iter().collect());
            j += batch_size;
        }

        batched_texts.push(batched_text);
    }

    batched_texts
}
