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
    let usage = chunk.get("usage").filter(|u| !u.is_null())?;
    let get = |k: &str| usage.get(k).and_then(|t| t.as_u64()).unwrap_or(0) as u32;
    Some(crate::language_models::TokenUsage {
        prompt_tokens: get("prompt_tokens"),
        completion_tokens: get("completion_tokens"),
        total_tokens: get("total_tokens"),
    })
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
}
