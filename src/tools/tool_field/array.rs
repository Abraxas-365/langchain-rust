use serde_json::{Map, Value};

use super::{BooleanField, IntegerField, NumberField, StringField, ToolField};

pub struct ArrayField {
    name: String,
    description: Option<String>,
    required: bool,
    field: Box<dyn ToolField>,
}

impl ArrayField {
    pub fn new<S>(
        name: S,
        description: Option<String>,
        required: bool,
        field: Box<dyn ToolField>,
    ) -> Self
    where
        S: Into<String>,
    {
        ArrayField {
            name: name.into(),
            description,
            required,
            field,
        }
    }

    pub fn new_string_array<S>(name: S, description: Option<String>, required: bool) -> Self
    where
        S: Into<String>,
    {
        ArrayField::new(
            name,
            description,
            required,
            StringField::new("items", None, true, None).into(),
        )
    }

    pub fn new_integer_array<S>(name: S, description: Option<String>, required: bool) -> Self
    where
        S: Into<String>,
    {
        ArrayField::new(
            name,
            description,
            required,
            IntegerField::new("items", None, true, None).into(),
        )
    }

    pub fn new_number_array<S>(name: S, description: Option<String>, required: bool) -> Self
    where
        S: Into<String>,
    {
        ArrayField::new(
            name,
            description,
            required,
            NumberField::new("items", None, true, None).into(),
        )
    }

    pub fn new_boolean_array<S>(name: S, description: Option<String>, required: bool) -> Self
    where
        S: Into<String>,
    {
        ArrayField::new(
            name,
            description,
            required,
            BooleanField::new("items", None, true, None).into(),
        )
    }
}

impl ToolField for ArrayField {
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

        fields.insert("type".into(), "array".into());
        fields.insert("items".into(), self.field.to_openai_field());
        if let Some(description) = self.description() {
            fields.insert("description".into(), description.into());
        }

        Value::Object(fields)
    }

    fn to_plain_description(&self) -> String {
        let type_info = if self.required {
            "array"
        } else {
            "array, optional"
        };

        let items_description = self
            .field
            .to_plain_description()
            .lines()
            .map(|line| format!("    {}", line))
            .collect::<Vec<_>>()
            .join("\n");

        match &self.description {
            Some(description) => format!(
                "{} ({}): {}\n{}",
                self.name, type_info, description, items_description
            ),
            None => format!("{} ({})\n{}", self.name, type_info, items_description),
        }
    }
}

impl From<ArrayField> for Box<dyn ToolField> {
    fn from(value: ArrayField) -> Self {
        Box::new(value)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_array_field_plain_description() {
        let field = ArrayField::new_integer_array("test", Some("test description".into()), true);

        assert_eq!(
            field.to_plain_description(),
            "test (array): test description\n    items (integer)"
        );

        let field_optional =
            ArrayField::new_string_array("test", Some("test description".into()), false);
        assert_eq!(
            field_optional.to_plain_description(),
            "test (array, optional): test description\n    items (string)"
        );

        let field_optional_no_description = ArrayField::new_number_array("test", None, false);
        assert_eq!(
            field_optional_no_description.to_plain_description(),
            "test (array, optional)\n    items (number)"
        );
    }

    #[test]
    fn test_array_field_openai() {
        let field = ArrayField::new_integer_array("test", Some("test description".into()), true);
        assert_eq!(
            field.to_openai_field(),
            json!({
                "type": "array",
                "description": "test description",
                "items": {
                    "type": "integer"
                }
            })
        );

        let field_optional =
            ArrayField::new_string_array("test", Some("test description".into()), false);
        assert_eq!(
            field_optional.to_openai_field(),
            json!({
                "type": "array",
                "description": "test description",
                "items": {
                    "type": "string"
                }
            })
        );

        let field_optional_no_description = ArrayField::new_number_array("test", None, false);
        assert_eq!(
            field_optional_no_description.to_openai_field(),
            json!({
                "type": "array",
                "items": {
                    "type": "number"
                }
            })
        );
    }
}
