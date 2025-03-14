use std::fmt::Display;

use derive_new::new;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, new)]
pub struct Article {
    title: String,
    link: String,
    snippet: String,
}

impl Display for Article {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]({})\n{}", self.title, self.link, self.snippet)
    }
}

