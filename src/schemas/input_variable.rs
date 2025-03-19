use std::collections::HashMap;

pub type InputVariables = HashMap<String, String>;

/// `input_variables!` is a utility macro used for creating a `std::collections::HashMap<String, serde_json::Value>`.
/// This HashMap can then be passed as arguments to a function or method.
///
/// # Usage
/// In this macro, the keys are `&str` and values are arbitrary types that get serialized into `serde_json::Value`:
/// ```rust,ignore
/// prompt_args! {
///     "input" => "Who is the writer of 20,000 Leagues Under the Sea, and what is my name?",
///     "history" => vec![
///         Message::new_human_message("My name is: Luis"),
///         Message::new_ai_message("Hi Luis"),
///     ],
/// }
/// ```
///
/// # Arguments
/// * `key` - A `&str` that will be used as the key in the resulting HashMap.<br>
/// * `value` - An arbitrary type that will be serialized into `serde_json::Value` and associated with the corresponding key.
///
/// The precise keys and values are dependent on your specific use case. In this example, "input" and "history" are keys,
/// and
#[macro_export]
macro_rules! input_variables {
    ( $($key:expr => $value:expr),* $(,)? ) => {
        std::collections::HashMap::<String, String>::from([$(
            ($key.into(), $value.to_string()),
        )*])
    };
}
