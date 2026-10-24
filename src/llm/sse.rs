//! Minimal Server-Sent Events decoder shared by the hand-written HTTP LLM clients.
//!
//! Network chunks do not align with SSE events: one chunk may carry several events and an
//! event may be split across chunks. This decoder buffers bytes until a full line is
//! available and yields the JSON payload of every `data:` line (skipping `[DONE]`).
use async_stream::stream;
use futures::{Stream, StreamExt};
use serde_json::Value;

use crate::language_models::LLMError;

pub(crate) fn sse_json_events<S, B, E>(bytes: S) -> impl Stream<Item = Result<Value, LLMError>>
where
    S: Stream<Item = Result<B, E>> + Send + 'static,
    B: AsRef<[u8]>,
    E: Into<LLMError>,
{
    stream! {
        let mut buffer: Vec<u8> = Vec::new();
        futures::pin_mut!(bytes);
        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(chunk) => buffer.extend_from_slice(chunk.as_ref()),
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            }
            while let Some(pos) = buffer.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=pos).collect();
                if let Some(event) = parse_line(&line) {
                    yield event;
                }
            }
        }
        if let Some(event) = parse_line(&buffer) {
            yield event;
        }
    }
}

fn parse_line(line: &[u8]) -> Option<Result<Value, LLMError>> {
    let line = String::from_utf8_lossy(line);
    let data = line
        .trim_end_matches(['\r', '\n'])
        .strip_prefix("data:")?
        .trim_start();
    if data.is_empty() || data == "[DONE]" {
        return None;
    }
    Some(
        serde_json::from_str(data)
            .map_err(|e| LLMError::ParsingError(format!("Failed to parse SSE data: {e}"))),
    )
}

/// Extracts OpenAI-compatible `usage` from a streamed chunk, if present.
pub(crate) fn usage_from_chunk(chunk: &Value) -> Option<crate::language_models::TokenUsage> {
    chunk
        .get("usage")
        .filter(|u| !u.is_null())
        .map(usage_from_value)
}

/// Parses an OpenAI-compatible `usage` object. Cached prompt tokens come as
/// `prompt_tokens_details.cached_tokens` (OpenAI, Qwen) or `prompt_cache_hit_tokens`
/// (DeepSeek).
pub(crate) fn usage_from_value(usage: &Value) -> crate::language_models::TokenUsage {
    let get = |v: &Value| v.as_u64().unwrap_or(0) as u32;
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .or_else(|| usage["prompt_cache_hit_tokens"].as_u64())
        .unwrap_or(0) as u32;
    crate::language_models::TokenUsage {
        prompt_tokens: get(&usage["prompt_tokens"]),
        completion_tokens: get(&usage["completion_tokens"]),
        total_tokens: get(&usage["total_tokens"]),
        cache_read_tokens: cached,
        cache_creation_tokens: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn decode(chunks: &[&str]) -> Vec<Result<Value, LLMError>> {
        let owned: Vec<Result<Vec<u8>, LLMError>> =
            chunks.iter().map(|c| Ok(c.as_bytes().to_vec())).collect();
        sse_json_events(futures::stream::iter(owned))
            .collect()
            .await
    }

    #[tokio::test]
    async fn multiple_events_in_one_chunk() {
        let out = decode(&["data: {\"a\":1}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\n"]).await;
        let values: Vec<Value> = out.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            values,
            vec![serde_json::json!({"a":1}), serde_json::json!({"a":2})]
        );
    }

    #[tokio::test]
    async fn event_split_across_chunks_and_crlf() {
        let out = decode(&[
            "data: {\"a\"",
            ":1}\r\n",
            "\r\n: comment\r\nevent: x\r\ndata:{\"b\":2}",
        ])
        .await;
        let values: Vec<Value> = out.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            values,
            vec![serde_json::json!({"a":1}), serde_json::json!({"b":2})]
        );
    }

    #[tokio::test]
    async fn invalid_json_is_an_error() {
        let out = decode(&["data: {nope}\n"]).await;
        assert!(matches!(out[0], Err(LLMError::ParsingError(_))));
    }

    #[test]
    fn usage_extraction() {
        let chunk = serde_json::json!({"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}});
        assert_eq!(usage_from_chunk(&chunk).unwrap().total_tokens, 3);
        assert!(usage_from_chunk(&serde_json::json!({"usage": null})).is_none());
        assert!(usage_from_chunk(&serde_json::json!({})).is_none());
    }

    #[test]
    fn usage_reports_cached_prompt_tokens() {
        use crate::language_models::TokenUsage;

        let openai = serde_json::json!({
            "prompt_tokens": 2006, "completion_tokens": 300, "total_tokens": 2306,
            "prompt_tokens_details": {"cached_tokens": 1920}
        });
        assert_eq!(
            usage_from_value(&openai),
            TokenUsage::new(2006, 300).with_cache_tokens(1920, 0)
        );
        let deepseek = serde_json::json!({
            "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
            "prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 36
        });
        assert_eq!(usage_from_value(&deepseek).cache_read_tokens, 64);
        let none =
            serde_json::json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3});
        assert_eq!(usage_from_value(&none), TokenUsage::new(1, 2));
    }
}
