use crate::text_splitter::merge_splits;

use super::{SplitterOptions, TextSplitter};

// RecursiveCharacter is a text splitter that will split texts recursively by different
// characters.
pub struct RecursiveCharacter {
    pub separators: Vec<String>,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub len_func: fn(&str) -> usize,
}

impl RecursiveCharacter {
    pub fn new(opt: SplitterOptions) -> Self {
        RecursiveCharacter {
            separators: opt.separators,
            chunk_size: opt.chunk_size,
            chunk_overlap: opt.chunk_overlap,
            len_func: opt.len_func,
        }
    }
}

impl TextSplitter for RecursiveCharacter {
    fn split_text(&self, text: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut final_chunks = Vec::new();

        //Find the appropriate separator
        let mut separator = self.separators.last().ok_or("No separators")?.clone();
        let mut new_separators: Vec<String> = Vec::new();
        for (i, c) in self.separators.iter().enumerate() {
            if c.is_empty() || text.contains(c) {
                separator = c.to_string();
                new_separators = self.separators[i + 1..].to_vec();
                break;
            }
        }

        let splits = text.split(&separator).collect::<Vec<&str>>();
        let mut good_splits = Vec::new();

        for split in splits.iter() {
            if (self.len_func)(split) < self.chunk_size as usize {
                good_splits.push(split.to_string());
                continue;
            }

            if !good_splits.is_empty() {
                let merged_text = merge_splits(
                    &good_splits,
                    &separator,
                    self.chunk_size,
                    self.chunk_overlap,
                    self.len_func,
                );
                final_chunks.extend(merged_text);
                good_splits = Vec::new();
            }

            if new_separators.is_empty() {
                final_chunks.push(split.to_string());
            } else {
                let other_info = self.split_text(split)?;
                final_chunks.extend(other_info);
            }
        }

        if good_splits.len() > 0 {
            let merged_text = merge_splits(
                &good_splits,
                &separator,
                self.chunk_size,
                self.chunk_overlap,
                self.len_func,
            );
            final_chunks.extend(merged_text);
        }

        Ok(final_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple length function for testing purposes
    fn test_len_func(s: &str) -> usize {
        s.chars().count()
    }

    #[test]
    fn test_recursive_character_split() {
        let text = "哈里森\n很高兴遇见你\n欢迎来中国";
        let separators = vec![
            "\n\n".to_string(),
            "\n".to_string(),
            " ".to_string(),
            "".to_string(),
        ];
        let chunk_size = 10;
        let chunk_overlap = 0;

        let splitter = RecursiveCharacter {
            separators,
            chunk_size,
            chunk_overlap,
            len_func: test_len_func,
        };

        let expected = vec!["哈里森\n很高兴遇见你", "欢迎来中国"];
        let result = splitter.split_text(text).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_recursive_character_with_overlap() {
        let text = "Hi, Harrison. \nI am glad to meet you";
        let separators = vec!["\n".to_string(), "$".into()];
        let chunk_size = 20;
        let chunk_overlap = 1;

        let splitter = RecursiveCharacter {
            separators,
            chunk_size,
            chunk_overlap,
            len_func: test_len_func,
        };

        let expected = vec!["Hi, Harrison.", "I am glad to meet you"];
        let result = splitter.split_text(text).unwrap();

        assert_eq!(result, expected);
    }
}
