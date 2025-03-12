use std::fmt::Display;

use derive_new::new;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, new)]
pub struct SearchResult {
    title: String,
    link: String,
    snippet: String,
}

impl Display for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]({})\n{}", self.title, self.link, self.snippet)
    }
}

#[derive(new)]
pub struct SearchResults(Vec<SearchResult>);

impl Display for SearchResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.0.is_empty() {
            write!(
                f,
                "{}",
                self.0
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join("\n---\n")
            )?;
        } else {
            write!(f, "No results found, try adjusting your search query")?;
        }
        Ok(())
    }
}
