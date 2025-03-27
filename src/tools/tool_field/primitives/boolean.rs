use crate::tools::tool_field::ToolField;

use super::ToolFieldPrimitive;

pub struct BooleanField {
    name: String,
    description: Option<String>,
    required: bool,
    r#enum: Option<Vec<bool>>,
}

impl BooleanField {
    pub fn new<S>(
        name: S,
        description: Option<String>,
        required: bool,
        r#enum: Option<Vec<bool>>,
    ) -> Self
    where
        S: Into<String>,
    {
        BooleanField {
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

impl ToolFieldPrimitive for BooleanField {
    type FieldType = bool;

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
        "boolean"
    }

    fn r#enum(&self) -> Option<&Vec<bool>> {
        self.r#enum.as_ref()
    }
}

impl From<BooleanField> for Box<dyn ToolField> {
    fn from(value: BooleanField) -> Self {
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
        let field = BooleanField::new("test", Some("test description".into()), true, None);
        assert_eq!(
            field.to_plain_description(),
            "test (boolean): test description"
        );

        let optional_field =
            BooleanField::new("test", Some("test description".into()), false, None);
        assert_eq!(
            optional_field.to_plain_description(),
            "test (boolean, optional): test description"
        );

        let enum_field = BooleanField::new(
            "test",
            Some("test description".into()),
            true,
            Some([true, false].into_iter().collect()),
        );
        assert_eq!(
            enum_field.to_plain_description(),
            "test (boolean): test description, should be one of [true, false]"
        );

        let enum_field_without_description = BooleanField::new(
            "test",
            None,
            true,
            Some([true, false].into_iter().collect()),
        );
        assert_eq!(
            enum_field_without_description.to_plain_description(),
            "test (boolean): should be one of [true, false]"
        );

        let field_without_description = BooleanField::new("test", None, true, None);
        assert_eq!(
            field_without_description.to_plain_description(),
            "test (boolean)"
        )
    }

    #[test]
    fn test_boolean_field_openai() {
        let field = BooleanField::new("test", Some("test description".into()), true, None);
        assert_eq!(
            field.to_openai_field(),
            json!({
                "type": "boolean",
                "description": "test description"
            })
        );

        let enum_field = BooleanField::new(
            "test",
            Some("test description".into()),
            true,
            Some([true, false].into_iter().collect()),
        );
        assert_eq!(
            enum_field.to_openai_field(),
            json!({
                "type": "boolean",
                "description": "test description",
                "enum": [true, false]
            })
        );

        let field_without_description = BooleanField::new("test", None, true, None);
        assert_eq!(
            field_without_description.to_openai_field(),
            json!({
                "type": "boolean"
            })
        );
    }
}
