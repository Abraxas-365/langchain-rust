use crate::tools::tool_field::ToolField;

use super::ToolFieldPrimitive;

pub struct StringField {
    name: String,
    description: Option<String>,
    required: bool,
    r#enum: Option<Vec<String>>,
}

impl StringField {
    pub fn new<S>(
        name: S,
        description: Option<String>,
        required: bool,
        r#enum: Option<Vec<String>>,
    ) -> Self
    where
        S: Into<String>,
    {
        StringField {
            name: name.into(),
            description,
            required,
            r#enum: r#enum.map(|options| {
                let mut options = options.clone();
                options.dedup();
                options
            }),
        }
    }
}

impl ToolFieldPrimitive for StringField {
    type FieldType = String;

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn required(&self) -> bool {
        self.required
    }

    fn type_name(&self) -> &str {
        "string"
    }

    fn r#enum(&self) -> Option<&Vec<String>> {
        self.r#enum.as_ref()
    }
}

impl From<StringField> for Box<dyn ToolField> {
    fn from(value: StringField) -> Self {
        Box::new(value)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tools::tool_field::ToolField;

    #[test]
    fn test_boolean_field_plain_description() {
        let field = StringField::new("test", Some("test description".into()), true, None);
        assert_eq!(
            field.to_plain_description(),
            "test (string): test description"
        );

        let optional_field = StringField::new("test", Some("test description".into()), false, None);
        assert_eq!(
            optional_field.to_plain_description(),
            "test (string, optional): test description"
        );

        let enum_field = StringField::new(
            "test",
            Some("test description".into()),
            true,
            Some(["lala".into(), "blah".into()].into_iter().collect()),
        );
        assert_eq!(
            enum_field.to_plain_description(),
            "test (string): test description, should be one of [lala, blah]"
        );

        let enum_field_without_description = StringField::new(
            "test",
            None,
            true,
            Some(["true".into(), "blah".into()].into_iter().collect()),
        );
        assert_eq!(
            enum_field_without_description.to_plain_description(),
            "test (string): should be one of [true, blah]"
        );

        let field_without_description = StringField::new("test", None, true, None);
        assert_eq!(
            field_without_description.to_plain_description(),
            "test (string)"
        )
    }

    #[test]
    fn test_boolean_field_openai() {
        let field = StringField::new("test", Some("test description".into()), true, None);
        assert_eq!(
            field.to_openai_field(),
            json!({
                "type": "string",
                "description": "test description"
            })
        );

        let enum_field = StringField::new(
            "test",
            Some("test description".into()),
            true,
            Some(["lala".into(), "blah".into()].into_iter().collect()),
        );
        assert_eq!(
            enum_field.to_openai_field(),
            json!({
                "type": "string",
                "description": "test description",
                "enum": ["lala", "blah"]
            })
        );

        let field_without_description = StringField::new("test", None, true, None);
        assert_eq!(
            field_without_description.to_openai_field(),
            json!({
                "type": "string"
            })
        );
    }
}
