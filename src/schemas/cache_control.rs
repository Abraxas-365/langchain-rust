use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Marks a prompt prefix for the provider to cache (Anthropic prompt caching).
///
/// Set it on a [`Message`](crate::schemas::Message) to cache everything up to and including
/// that message, or on [`CallOptions`](crate::language_models::options::CallOptions) to let
/// the provider move the breakpoint to the end of the prompt on every call. Providers that
/// cache automatically (OpenAI, DeepSeek) ignore it but still report cached tokens in
/// [`TokenUsage`](crate::language_models::TokenUsage).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheControl {
    /// Cached for 5 minutes, refreshed each time it is read.
    #[default]
    Ephemeral,
    /// Cached for an hour (writes cost more).
    EphemeralOneHour,
}

impl CacheControl {
    /// Anthropic's `cache_control` object.
    pub(crate) fn to_anthropic(self) -> Value {
        match self {
            CacheControl::Ephemeral => json!({"type": "ephemeral"}),
            CacheControl::EphemeralOneHour => json!({"type": "ephemeral", "ttl": "1h"}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_wire_format() {
        assert_eq!(
            CacheControl::Ephemeral.to_anthropic(),
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            CacheControl::EphemeralOneHour.to_anthropic(),
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert_eq!(CacheControl::default(), CacheControl::Ephemeral);
        assert_eq!(
            serde_json::to_value(CacheControl::EphemeralOneHour).unwrap(),
            json!("ephemeral_one_hour")
        );
    }
}
