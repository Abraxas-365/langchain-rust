use crate::tools::tool_field::ToolField;

use super::ToolFieldPrimitive;

pub struct IntegerField {
    name: String,
    description: Option<String>,
    required: bool,
    r#enum: Option<Vec<i64>>,
}

impl IntegerField {
    pub fn new<S>(
        name: S,
        description: Option<String>,
        required: bool,
        r#enum: Option<Vec<i64>>,
    ) -> Self
    where
        S: Into<String>,
    {
        IntegerField {
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

impl ToolFieldPrimitive for IntegerField {
    type FieldType = i64;

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
        "integer"
    }

    fn r#enum(&self) -> Option<&Vec<i64>> {
        self.r#enum.as_ref()
    }
}

impl From<IntegerField> for Box<dyn ToolField> {
    fn from(value: IntegerField) -> Self {
        Box::new(value)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tools::tool_field::ToolField;

    #[test]
    fn test_integer_field_plain_description() {
        let field = IntegerField::new("test", Some("test description".into()), true, None);
        assert_eq!(
            field.to_plain_description(),
            "test (integer): test description"
        );

        let optional_field =
            IntegerField::new("test", Some("test description".into()), false, None);
        assert_eq!(
            optional_field.to_plain_description(),
            "test (integer, optional): test description"
        );

        let enum_field = IntegerField::new(
            "test",
            Some("test description".into()),
            true,
            Some([0, 1, 3].into_iter().collect()),
        );
        assert_eq!(
            enum_field.to_plain_description(),
            "test (integer): test description, should be one of [0, 1, 3]"
        );

        let enum_field_without_description =
            IntegerField::new("test", None, true, Some([0, 1, 5].into_iter().collect()));
        assert_eq!(
            enum_field_without_description.to_plain_description(),
            "test (integer): should be one of [0, 1, 5]"
        );

        let field_without_description = IntegerField::new("test", None, true, None);
        assert_eq!(
            field_without_description.to_plain_description(),
            "test (integer)"
        )
    }

    #[test]
    fn test_integer_field_openai() {
        let field = IntegerField::new("test", Some("test description".into()), true, None);
        assert_eq!(
            field.to_openai_field(),
            json!({
                "type": "integer",
                "description": "test description"
            })
        );

        let enum_field = IntegerField::new(
            "test",
            Some("test description".into()),
            true,
            Some([4, 8].into_iter().collect()),
        );
        assert_eq!(
            enum_field.to_openai_field(),
            json!({
                "type": "integer",
                "description": "test description",
                "enum": [4, 8]
            })
        );

        let field_without_description = IntegerField::new("test", None, true, None);
        assert_eq!(
            field_without_description.to_openai_field(),
            json!({
                "type": "integer"
            })
        );
    }
}
