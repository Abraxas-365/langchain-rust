use serde_json::{Map, Value};

use super::ToolField;

pub struct ObjectField {
    name: String,
    description: Option<String>,
    required: bool,
    properties: Vec<Box<dyn ToolField>>,
    additional_properties: Option<bool>,
}

impl ObjectField {
    pub fn new<S>(
        name: S,
        description: Option<String>,
        required: bool,
        properties: Vec<Box<dyn ToolField>>,
        additional_properties: Option<bool>,
    ) -> Self
    where
        S: Into<String>,
    {
        Self {
            name: name.into(),
            description,
            required,
            properties,
            additional_properties,
        }
    }

    pub fn new_tool_input(properties: Vec<Box<dyn ToolField>>) -> Self {
        Self::new("input", None, true, properties, None)
    }

    pub fn properties_description(&self) -> String {
        let properties = self
            .properties
            .iter()
            .map(|property| property.to_plain_description())
            .collect::<Vec<_>>()
            .join(",\n");

        let properties = properties
            .lines()
            .map(|line| format!("    {}", line))
            .collect::<Vec<_>>()
            .join("\n");

        if properties.is_empty() {
            "{}".into()
        } else {
            format!("{{\n{}\n}}", properties)
        }
    }
}

impl ToolField for ObjectField {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn required(&self) -> bool {
        self.required
    }

    fn to_openai_field(&self) -> Value {
        let mut fields = Map::<String, Value>::new();

        fields.insert("type".into(), "object".into());
        fields.insert(
            "properties".into(),
            Map::from_iter(
                self.properties
                    .iter()
                    .map(|property| (property.name().into(), property.to_openai_field())),
            )
            .into(),
        );
        fields.insert(
            "required".into(),
            self.properties
                .iter()
                .filter(|property| property.required())
                .map(|property| property.name())
                .collect::<Vec<_>>()
                .into(),
        );
        if let Some(description) = self.description() {
            fields.insert("description".into(), description.into());
        }

        Value::Object(fields)
    }

    fn to_plain_description(&self) -> String {
        let type_info = if self.required {
            "object"
        } else {
            "object, optional"
        };

        format!(
            "{} ({}): {}",
            self.name,
            type_info,
            self.properties_description()
        )
    }
}

impl From<ObjectField> for Box<dyn ToolField> {
    fn from(value: ObjectField) -> Self {
        Box::new(value)
    }
}

#[cfg(test)]
mod tests {
    use crate::tools::tool_field::{IntegerField, StringField};

    use super::*;
    use indoc::indoc;
    use serde_json::json;

    #[test]
    fn test_object_field_properties_description() {
        let field = ObjectField::new("test", None, true, vec![], None);
        assert_eq!(field.properties_description(), "{}");

        let field_complicated = ObjectField::new(
            "test",
            None,
            false,
            vec![
                Box::new(StringField::new(
                    "query",
                    Some("A query to search for".into()),
                    true,
                    None,
                )),
                Box::new(IntegerField::new(
                    "limit",
                    Some("Max number of articles to search".into()),
                    false,
                    None,
                )),
            ],
            None,
        );
        assert_eq!(
            field_complicated.properties_description(),
            indoc! {"
            {
                query (string): A query to search for,
                limit (integer, optional): Max number of articles to search
            }"}
        )
    }

    #[test]
    fn test_object_field_plain_description() {
        let field = ObjectField::new("test", None, true, vec![], None);
        assert_eq!(field.to_plain_description(), "test (object): {}");

        let field_complicated = ObjectField::new(
            "test",
            None,
            false,
            vec![
                Box::new(StringField::new(
                    "query",
                    Some("A query to search for".into()),
                    true,
                    None,
                )),
                Box::new(IntegerField::new(
                    "limit",
                    Some("Max number of articles to search".into()),
                    false,
                    None,
                )),
            ],
            None,
        );
        assert_eq!(
            field_complicated.to_plain_description(),
            indoc! {"
            test (object, optional): {
                query (string): A query to search for,
                limit (integer, optional): Max number of articles to search
            }"}
        )
    }

    #[test]
    fn test_object_field_openai() {
        let field = ObjectField::new("test", None, true, vec![], None);
        assert_eq!(
            field.to_openai_field(),
            json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        );

        let field_complicated = ObjectField::new(
            "test",
            None,
            false,
            vec![
                Box::new(StringField::new(
                    "query",
                    Some("A query to search for".into()),
                    true,
                    None,
                )),
                Box::new(IntegerField::new(
                    "limit",
                    Some("Max number of articles to search".into()),
                    false,
                    None,
                )),
            ],
            None,
        );
        assert_eq!(
            field_complicated.to_openai_field(),
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A query to search for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of articles to search"
                    }
                },
                "required": ["query"]
            })
        )
    }
}
