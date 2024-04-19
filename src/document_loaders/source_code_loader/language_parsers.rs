use crate::schemas::Document;
use std::collections::HashMap;
use std::fmt::Debug;
use std::string::ToString;
use strum_macros::Display;
use tree_sitter::{Parser, Tree};

#[derive(Display, Debug, Clone)]
pub enum Language {
    Rust,
    C,
    Cpp,
    Javascript,
    // TODO add Typescript when tree-sitter-typescript-21+ is available,
    // Typescript,
    Go,
    Python,
}

pub enum LanguageContentTypes {
    SimplifiedCode,
    FunctionsImpls,
}

impl ToString for LanguageContentTypes {
    fn to_string(&self) -> String {
        match self {
            LanguageContentTypes::SimplifiedCode => "simplified_code".to_string(),
            LanguageContentTypes::FunctionsImpls => "functions_impls".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LanguageParserOptions {
    pub parser_threshold: u64,
    pub language: Language,
}

impl Default for LanguageParserOptions {
    fn default() -> Self {
        Self {
            parser_threshold: 1000,
            language: Language::Rust,
        }
    }
}

pub struct LanguageParser {
    parser: Parser,
    parser_options: LanguageParserOptions,
}

impl Debug for LanguageParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LanguageParser {{ language: {:?} }}",
            self.parser_options.language
        )
    }
}

impl Clone for LanguageParser {
    fn clone(&self) -> Self {
        LanguageParser {
            parser: get_language_parser(&self.parser_options.language),
            parser_options: self.parser_options.clone(),
        }
    }
}

pub fn get_language_by_filename(name: &String) -> Language {
    let extension = name.split('.').last().unwrap();
    match extension.to_lowercase().as_str() {
        "rs" => Language::Rust,
        "c" => Language::C,
        "cpp" => Language::Cpp,
        "h" => Language::C,
        "hpp" => Language::Cpp,
        "js" => Language::Javascript,
        // "ts" => Language::Typescript,
        "go" => Language::Go,
        "py" => Language::Python,
        _ => panic!("Unsupported language"),
    }
}

fn get_language_parser(language: &Language) -> Parser {
    let mut parser = Parser::new();
    let lang = match language {
        Language::Rust => tree_sitter_rust::language(),
        Language::C => tree_sitter_c::language(),
        Language::Cpp => tree_sitter_cpp::language(),
        Language::Javascript => tree_sitter_javascript::language(),
        // Language::Typescript => tree_sitter_typescript::language_typescript(),
        Language::Go => tree_sitter_go::language(),
        Language::Python => tree_sitter_python::language(),
    };
    parser.set_language(&lang).expect("Error loading grammar");
    parser
}

impl LanguageParser {
    pub fn from_language(language: Language) -> Self {
        Self {
            parser: get_language_parser(&language),
            parser_options: LanguageParserOptions {
                language,
                ..LanguageParserOptions::default()
            },
        }
    }

    pub fn with_parser_option(mut self, parser_option: LanguageParserOptions) -> Self {
        self.parser_options = parser_option;
        self
    }
}

impl LanguageParser {
    pub fn set_parser_threshold(&mut self, threshold: u64) {
        self.parser_options.parser_threshold = threshold;
    }

    pub fn parse_code(&mut self, code: &String) -> Vec<Document> {
        let tree = self.parser.parse(code, None).unwrap();
        if self.parser_options.parser_threshold > tree.root_node().end_position().row as u64 {
            return vec![Document::new(code).with_metadata(HashMap::from([
                (
                    "content_type".to_string(),
                    serde_json::Value::from(LanguageContentTypes::SimplifiedCode.to_string()),
                ),
                (
                    "language".to_string(),
                    serde_json::Value::from(self.parser_options.language.to_string()),
                ),
            ]))];
        }
        self.extract_functions_classes(tree, code)
    }

    pub fn extract_functions_classes(&self, tree: Tree, code: &String) -> Vec<Document> {
        let mut chunks = Vec::new();

        let count = tree.root_node().child_count();
        for i in 0..count {
            let node = tree.root_node().child(i).unwrap();
            let source_code = node.utf8_text(code.as_bytes()).unwrap().to_string();
            let lang_meta = (
                "language".to_string(),
                serde_json::Value::from(self.parser_options.language.to_string()),
            );
            if node.kind() == "function_item" || node.kind() == "impl_item" {
                let doc = Document::new(source_code).with_metadata(HashMap::from([
                    lang_meta.clone(),
                    (
                        "content_type".to_string(),
                        serde_json::Value::from(LanguageContentTypes::FunctionsImpls.to_string()),
                    ),
                ]));
                chunks.push(doc);
            } else {
                let doc = Document::new(source_code).with_metadata(HashMap::from([
                    lang_meta.clone(),
                    (
                        "content_type".to_string(),
                        serde_json::Value::from(LanguageContentTypes::SimplifiedCode.to_string()),
                    ),
                ]));
                chunks.push(doc);
            }
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_parser() {
        let code = r#"
        fn main() {
            println!("Hello, world!");
        }

        pub struct Person {
            name: String,
            age: i32,
        }

        impl Person {
            pub fn new(name: String, age: i32) -> Self {
                Self { name, age }
            }

            pub fn get_name(&self) -> &str {
                &self.name
            }

            pub fn get_age(&self) -> i32 {
                self.age
            }
        }
        "#;

        let mut parser = LanguageParser::from_language(Language::Rust);

        let documents = parser.parse_code(&code.to_string());
        assert_eq!(documents.len(), 1);

        // Set the parser threshold to 10 for testing
        parser.set_parser_threshold(10);

        let documents = parser.parse_code(&code.to_string());
        assert_eq!(documents.len(), 3);
        assert_eq!(
            documents[0].page_content,
            "fn main() {\n            println!(\"Hello, world!\");\n        }"
        );
        assert_eq!(
            documents[1].metadata.get("content_type").unwrap(),
            LanguageContentTypes::SimplifiedCode.to_string().as_str()
        );
    }
}
