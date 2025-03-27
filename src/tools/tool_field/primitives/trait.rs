use std::fmt::Display;

use serde_json::{Map, Value};

use crate::tools::tool_field::ToolField;

pub trait ToolFieldPrimitive {
    type FieldType: Into<Value> + Display + Clone;

    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
    fn required(&self) -> bool;
    fn type_name(&self) -> &str;
    fn r#enum(&self) -> Option<&Vec<Self::FieldType>>;
}

impl<F, T> ToolField for F
where
    F: ToolFieldPrimitive<FieldType = T>,
    T: Into<Value> + Display + Clone,
{
    fn name(&self) -> &str {
        self.name()
    }

    fn description(&self) -> Option<&str> {
        self.description()
    }

    fn required(&self) -> bool {
        self.required()
    }

    fn to_openai_field(&self) -> Value {
        let mut field = Map::<String, Value>::new();

        field.insert("type".into(), self.type_name().into());

        if let Some(description) = self.description() {
            field.insert("description".into(), description.into());
        }

        if let Some(r#enum) = self.r#enum() {
            field.insert("enum".into(), r#enum.clone().into());
        }

        Value::Object(field)
    }

    fn to_plain_description(&self) -> String {
        let type_info = if self.required() {
            self.type_name().into()
        } else {
            format!("{}, optional", self.type_name())
        };

        let enum_options = self.r#enum().map(|options| {
            options
                .iter()
                .map(|option| option.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        });

        let description = match (self.description(), enum_options) {
            (None, None) => "".into(),
            (None, Some(options)) => format!(": should be one of [{}]", options),
            (Some(description), None) => format!(": {}", description),
            (Some(description), Some(options)) => {
                format!(": {}, should be one of [{}]", description, options)
            }
        };

        format!("{} ({}){}", self.name(), type_info, description)
    }
}
