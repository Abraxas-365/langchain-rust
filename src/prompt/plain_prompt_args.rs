use std::collections::HashMap;

use derive_new::new;

use crate::schemas::Message;

use super::PromptArgs;

#[derive(new, Clone)]
pub struct PlainPromptArgs {
    input: HashMap<String, String>,
    history: Vec<Message>,
}

impl PromptArgs for PlainPromptArgs {
    fn contains_key(&self, key: &str) -> bool {
        self.input.contains_key(key)
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.input.get(key).map(|s| s.as_str())
    }

    fn insert(&mut self, key: String, value: String) -> Option<String> {
        self.input.insert(key, value)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = (&String, &String)> + '_> {
        Box::new(self.input.iter())
    }
}
