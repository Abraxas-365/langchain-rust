use std::collections::HashMap;

use derive_new::new;

use super::Message;

#[derive(Clone, new)]
pub struct InputVariables(TextReplacements, PlaceholderReplacements);
pub type TextReplacements = HashMap<String, String>;
pub type PlaceholderReplacements = HashMap<String, Vec<Message>>;

impl InputVariables {
    pub fn contains_text_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    pub fn iter_test_replacements(&self) -> impl Iterator<Item = (&String, &String)> {
        self.0.iter()
    }

    pub fn get_text_replacement(&self, key: &str) -> Option<&String> {
        self.0.get(key)
    }

    pub fn insert_text_replacement(&mut self, key: &str, value: String) {
        self.0.insert(key.to_string(), value);
    }

    pub fn contains_placeholder_key(&self, key: &str) -> bool {
        self.1.contains_key(key)
    }

    pub fn get_placeholder_replacement(&self, key: &str) -> Option<&Vec<Message>> {
        self.1.get(key)
    }

    pub fn insert_placeholder_replacement(&mut self, key: &str, value: Vec<Message>) {
        self.1.insert(key.to_string(), value);
    }
}

impl From<TextReplacements> for InputVariables {
    fn from(text_replacements: TextReplacements) -> Self {
        Self(text_replacements, PlaceholderReplacements::new())
    }
}
impl From<PlaceholderReplacements> for InputVariables {
    fn from(placeholder_replacements: PlaceholderReplacements) -> Self {
        Self(TextReplacements::new(), placeholder_replacements)
    }
}

#[macro_export]
macro_rules! text_replacements {
    ( $($key:expr => $value:expr),* $(,)? ) => {
        $crate::schemas::TextReplacements::from([$(
            ($key.into(), $value.to_string()),
        )*])
    };
}

#[macro_export]
macro_rules! placeholder_replacements {
    ( $($key:expr => $value:expr),* $(,)? ) => {
        $crate::schemas::PlaceholderReplacements::from([$(
            ($key.into(), $value),
        )*])
    };
}
