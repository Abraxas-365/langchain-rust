use crate::tools::tool_field::ToolField;

use super::ToolFieldPrimitive;

pub struct NumberField {
    name: String,
    description: Option<String>,
    required: bool,
    r#enum: Option<Vec<f64>>,
}

impl NumberField {
    pub fn new<S>(
        name: S,
        description: Option<String>,
        required: bool,
        r#enum: Option<Vec<f64>>,
    ) -> Self
    where
        S: Into<String>,
    {
        NumberField {
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

impl ToolFieldPrimitive for NumberField {
    type FieldType = f64;

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
        "number"
    }

    fn r#enum(&self) -> Option<&Vec<f64>> {
        self.r#enum.as_ref()
    }
}

impl From<NumberField> for Box<dyn ToolField> {
    fn from(value: NumberField) -> Self {
        Box::new(value)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tools::tool_field::ToolField;

    #[test]
    fn test_number_field_plain_description() {
        let field = NumberField::new("test", Some("test description".into()), true, None);
        assert_eq!(
            field.to_plain_description(),
            "test (number): test description"
        );

        let optional_field = NumberField::new("test", Some("test description".into()), false, None);
        assert_eq!(
            optional_field.to_plain_description(),
            "test (number, optional): test description"
        );

        let enum_field = NumberField::new(
            "test",
            Some("test description".into()),
            true,
            Some([0.1, 3f64].into_iter().collect()),
        );
        assert_eq!(
            enum_field.to_plain_description(),
            "test (number): test description, should be one of [0.1, 3]"
        );

        let enum_field_without_description =
            NumberField::new("test", None, true, Some([3.2, 5f64].into_iter().collect()));
        assert_eq!(
            enum_field_without_description.to_plain_description(),
            "test (number): should be one of [3.2, 5]"
        );

        let field_without_description = NumberField::new("test", None, true, None);
        assert_eq!(
            field_without_description.to_plain_description(),
            "test (number)"
        )
    }

    #[test]
    fn test_boolean_field_openai() {
        let field = NumberField::new("test", Some("test description".into()), true, None);
        assert_eq!(
            field.to_openai_field(),
            json!({
                "type": "number",
                "description": "test description"
            })
        );

        let enum_field = NumberField::new(
            "test",
            Some("test description".into()),
            true,
            Some([3.1, 3.12].into_iter().collect()),
        );
        assert_eq!(
            enum_field.to_openai_field(),
            json!({
                "type": "number",
                "description": "test description",
                "enum": [3.1, 3.12]
            })
        );

        let field_without_description = NumberField::new("test", None, true, None);
        assert_eq!(
            field_without_description.to_openai_field(),
            json!({
                "type": "number"
            })
        );
    }
}
