use super::*;
use futures_util::StreamExt;
use mockito::{Matcher, Server};

const ID: &str = "5551722-f677-48a7-9b3a-0b0b9b9b9b9b";

fn loader(server: &Server, source: AudioSource) -> AssemblyAILoader {
    AssemblyAILoader::new(source)
        .with_api_key("test-key")
        .with_base_url(server.url())
        .with_poll_interval(Duration::from_millis(10))
        // A loader that never sees a final status fails in seconds, not after an hour.
        .with_max_wait(Duration::from_secs(5))
}

fn transcript(status: &str) -> String {
    json!({
        "id": ID,
        "status": status,
        "audio_url": "https://example.com/a.mp3",
        "text": if status == "completed" { json!("Hello there. General Kenobi.") } else { Value::Null },
        "words": [{"text": "Hello", "start": 0, "end": 300}],
        "audio_duration": 4,
        "confidence": 0.93,
        "language_code": "en_us"
    })
    .to_string()
}

async fn mock_submit(server: &mut Server, audio_url: &str) -> mockito::Mock {
    server
        .mock("POST", "/v2/transcript")
        .match_header("authorization", "test-key")
        .match_body(Matcher::PartialJson(json!({"audio_url": audio_url})))
        .with_header("content-type", "application/json")
        .with_body(transcript("queued"))
        .create_async()
        .await
}

async fn mock_get(server: &mut Server, status: &str, hits: usize) -> mockito::Mock {
    server
        .mock("GET", format!("/v2/transcript/{ID}").as_str())
        .match_header("authorization", "test-key")
        .with_header("content-type", "application/json")
        .with_body(transcript(status))
        .expect(hits)
        .create_async()
        .await
}

async fn collect(loader: AssemblyAILoader) -> Vec<Document> {
    let mut stream = loader.load().await.unwrap();
    let mut docs = Vec::new();
    while let Some(doc) = stream.next().await {
        docs.push(doc.unwrap());
    }
    docs
}

#[tokio::test]
async fn url_submit_poll_and_text_document() {
    let mut server = Server::new_async().await;
    let submit = server
        .mock("POST", "/v2/transcript")
        .match_header("authorization", "test-key")
        .match_body(Matcher::Json(json!({
            "audio_url": "https://example.com/a.mp3",
            "speaker_labels": true,
            "language_code": "es"
        })))
        .with_header("content-type", "application/json")
        .with_body(transcript("queued"))
        .create_async()
        .await;
    // Served in order: two polls still working, then done.
    let queued = mock_get(&mut server, "queued", 1).await;
    let processing = mock_get(&mut server, "processing", 1).await;
    let completed = mock_get(&mut server, "completed", 1).await;

    let docs = collect(
        loader(&server, AudioSource::url("https://example.com/a.mp3"))
            .with_param("speaker_labels", json!(true))
            .with_param("language_code", json!("es"))
            // audio_url always comes from the source.
            .with_param("audio_url", json!("https://ignored.example")),
    )
    .await;
    submit.assert_async().await;
    queued.assert_async().await;
    processing.assert_async().await;
    completed.assert_async().await;

    assert_eq!(docs.len(), 1);
    let doc = &docs[0];
    assert_eq!(doc.page_content, "Hello there. General Kenobi.");
    let m = &doc.metadata;
    assert_eq!(m["source"], json!("https://example.com/a.mp3"));
    assert_eq!(m["transcript_id"], json!(ID));
    assert_eq!(m["audio_duration"], json!(4));
    assert_eq!(m["language_code"], json!("en_us"));
    assert_eq!(m["status"], json!("completed"));
    for dropped in ["text", "words", "id"] {
        assert!(!m.contains_key(dropped), "{dropped} should not be metadata");
    }
}

#[tokio::test]
async fn file_and_bytes_are_uploaded_first() {
    let path = std::env::temp_dir().join(format!("lcr-assemblyai-{}.wav", std::process::id()));
    std::fs::write(&path, b"RIFF-fake-audio").unwrap();

    for (source, label) in [
        (AudioSource::file(&path), path.display().to_string()),
        (
            AudioSource::Bytes(b"RIFF-fake-audio".to_vec()),
            "bytes".to_string(),
        ),
    ] {
        let mut server = Server::new_async().await;
        let upload = server
            .mock("POST", "/v2/upload")
            .match_header("authorization", "test-key")
            .match_header("content-type", "application/octet-stream")
            .match_body("RIFF-fake-audio")
            .with_header("content-type", "application/json")
            .with_body(r#"{"upload_url": "https://cdn.assemblyai.com/upload/abc"}"#)
            .create_async()
            .await;
        let submit = mock_submit(&mut server, "https://cdn.assemblyai.com/upload/abc").await;
        let _get = mock_get(&mut server, "completed", 1).await;

        let docs = collect(loader(&server, source)).await;
        upload.assert_async().await;
        submit.assert_async().await;
        assert_eq!(docs[0].metadata["source"], json!(label));
        assert_eq!(docs[0].page_content, "Hello there. General Kenobi.");
    }
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn transcript_id_source_skips_submit() {
    let mut server = Server::new_async().await;
    let submit = server
        .mock("POST", Matcher::Any)
        .expect(0)
        .create_async()
        .await;
    let get = mock_get(&mut server, "completed", 1).await;
    let docs = collect(loader(&server, AudioSource::transcript_id(ID))).await;
    submit.assert_async().await;
    get.assert_async().await;
    assert_eq!(
        docs[0].metadata["source"],
        json!(format!("assemblyai:{ID}"))
    );
}

#[tokio::test]
async fn sentences_and_paragraphs_become_one_document_each() {
    for (format, field) in [
        (TranscriptFormat::Sentences, "sentences"),
        (TranscriptFormat::Paragraphs, "paragraphs"),
    ] {
        let mut server = Server::new_async().await;
        let _get = mock_get(&mut server, "completed", 1).await;
        let items = server
            .mock("GET", format!("/v2/transcript/{ID}/{field}").as_str())
            .match_header("authorization", "test-key")
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "id": ID, "confidence": 0.9, "audio_duration": 4,
                    field: [
                        {"text": "Hello there.", "start": 0, "end": 900, "confidence": 0.95,
                         "speaker": "A", "words": [{"text": "Hello"}]},
                        {"text": "General Kenobi.", "start": 1000, "end": 2100,
                         "confidence": 0.91, "speaker": "B", "words": []},
                        "not an object"
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let docs =
            collect(loader(&server, AudioSource::transcript_id(ID)).with_format(format)).await;
        items.assert_async().await;
        assert_eq!(docs.len(), 2, "{field}");
        assert_eq!(docs[0].page_content, "Hello there.");
        assert_eq!(docs[1].page_content, "General Kenobi.");
        let m = &docs[1].metadata;
        assert_eq!(m["start"], json!(1000));
        assert_eq!(m["end"], json!(2100));
        assert_eq!(m["speaker"], json!("B"));
        assert_eq!(m["transcript_id"], json!(ID));
        assert_eq!(m["source"], json!(format!("assemblyai:{ID}")));
        assert!(!m.contains_key("words"));
        assert!(!m.contains_key("text"));
    }
}

#[tokio::test]
async fn subtitles_srt_and_vtt() {
    for (format, kind, chars) in [
        (TranscriptFormat::SubtitlesSrt, "srt", Some(32)),
        (TranscriptFormat::SubtitlesVtt, "vtt", None),
    ] {
        let mut server = Server::new_async().await;
        let _get = mock_get(&mut server, "completed", 1).await;
        let query = match chars {
            Some(c) => Matcher::UrlEncoded("chars_per_caption".into(), c.to_string()),
            None => Matcher::Missing,
        };
        let body = format!("1\n00:00:00,000 --> 00:00:01,000\nHello there ({kind})\n");
        let subs = server
            .mock("GET", format!("/v2/transcript/{ID}/{kind}").as_str())
            .match_query(query)
            .with_header("content-type", "text/plain")
            .with_body(&body)
            .create_async()
            .await;

        let mut l = loader(&server, AudioSource::transcript_id(ID)).with_format(format);
        if let Some(c) = chars {
            l = l.with_chars_per_caption(c);
        }
        let docs = collect(l).await;
        subs.assert_async().await;
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, body);
        assert_eq!(docs[0].metadata["format"], json!(kind));
        assert_eq!(docs[0].metadata["transcript_id"], json!(ID));
    }
}

#[tokio::test]
async fn failed_transcript_reports_reason() {
    let mut server = Server::new_async().await;
    let _get = server
        .mock("GET", format!("/v2/transcript/{ID}").as_str())
        .with_header("content-type", "application/json")
        .with_body(
            json!({"id": ID, "status": "error", "error": "Download error, unable to download"})
                .to_string(),
        )
        .create_async()
        .await;
    let err = loader(&server, AudioSource::transcript_id(ID))
        .load_documents()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("Download error"), "{err}");
    assert!(err.contains(ID), "{err}");
}

#[tokio::test]
async fn gives_up_after_max_wait_with_the_id() {
    let mut server = Server::new_async().await;
    let get = server
        .mock("GET", format!("/v2/transcript/{ID}").as_str())
        .with_header("content-type", "application/json")
        .with_body(transcript("processing"))
        // 10 ms polls over 300 ms make dozens of calls; sleeping 20x too long makes 3.
        .expect_at_least(5)
        .create_async()
        .await;
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        loader(&server, AudioSource::transcript_id(ID))
            .with_poll_interval(Duration::from_millis(10))
            .with_max_wait(Duration::from_millis(300))
            .load_documents(),
    )
    .await
    .expect("max_wait must stop polling");
    let err = result.unwrap_err().to_string();
    get.assert_async().await;
    assert!(err.contains("not ready"), "{err}");
    assert!(err.contains(ID), "{err}");
}

#[tokio::test]
async fn api_errors_carry_status_and_cut_body() {
    let mut server = Server::new_async().await;
    let _submit = server
        .mock("POST", "/v2/transcript")
        .with_status(401)
        .with_body(format!(
            "{{\"error\": \"Invalid API key\"}}{}",
            "x".repeat(3000)
        ))
        .create_async()
        .await;
    let err = loader(&server, AudioSource::url("https://example.com/a.mp3"))
        .load_documents()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("401"), "{err}");
    assert!(err.contains("Invalid API key"), "{err}");
    assert!(err.contains("submit"), "{err}");
    assert!(err.len() < 600, "{} chars", err.len());
}

#[tokio::test]
async fn malformed_responses_are_errors() {
    // submit without id
    let mut server = Server::new_async().await;
    let _s = server
        .mock("POST", "/v2/transcript")
        .with_header("content-type", "application/json")
        .with_body(r#"{"status": "queued"}"#)
        .create_async()
        .await;
    let err = loader(&server, AudioSource::url("https://example.com/a.mp3"))
        .load_documents()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("no id"), "{err}");

    // submit id that would change the URL path
    let mut server = Server::new_async().await;
    let _s = server
        .mock("POST", "/v2/transcript")
        .with_header("content-type", "application/json")
        .with_body(r#"{"id": "../v2/upload"}"#)
        .create_async()
        .await;
    let err = loader(&server, AudioSource::url("https://example.com/a.mp3"))
        .load_documents()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid AssemblyAI transcript id"), "{err}");

    // upload without upload_url
    let mut server = Server::new_async().await;
    let _u = server
        .mock("POST", "/v2/upload")
        .with_header("content-type", "application/json")
        .with_body(r#"{}"#)
        .create_async()
        .await;
    let err = loader(&server, AudioSource::Bytes(vec![1, 2, 3]))
        .load_documents()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("upload_url"), "{err}");

    // transcript that is not JSON / not an object
    for body in ["not json", "[1, 2]"] {
        let mut server = Server::new_async().await;
        let _g = server
            .mock("GET", format!("/v2/transcript/{ID}").as_str())
            .with_header("content-type", "application/json")
            .with_body(body)
            .create_async()
            .await;
        assert!(
            loader(&server, AudioSource::transcript_id(ID))
                .load_documents()
                .await
                .is_err(),
            "{body}"
        );
    }

    // sentences response without the array
    let mut server = Server::new_async().await;
    let _g = mock_get(&mut server, "completed", 1).await;
    let _s = server
        .mock("GET", format!("/v2/transcript/{ID}/sentences").as_str())
        .with_header("content-type", "application/json")
        .with_body(r#"{"id": "x"}"#)
        .create_async()
        .await;
    let err = loader(&server, AudioSource::transcript_id(ID))
        .with_format(TranscriptFormat::Sentences)
        .load_documents()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("no sentences array"), "{err}");

    // subtitles error status
    let mut server = Server::new_async().await;
    let _g = mock_get(&mut server, "completed", 1).await;
    let _s = server
        .mock("GET", format!("/v2/transcript/{ID}/vtt").as_str())
        .with_status(400)
        .with_body("bad format")
        .create_async()
        .await;
    let err = loader(&server, AudioSource::transcript_id(ID))
        .with_format(TranscriptFormat::SubtitlesVtt)
        .load_documents()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("400") && err.contains("bad format"), "{err}");
}

#[tokio::test]
async fn request_timeout_applies() {
    // Accepts connections and never answers.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hold = tokio::spawn(async move {
        let mut open = Vec::new();
        while let Ok((socket, _)) = listener.accept().await {
            open.push(socket);
        }
    });
    // Without the 200 ms request timeout the 60 s default would trip the outer timeout.
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        AssemblyAILoader::new(AudioSource::transcript_id(ID))
            .with_api_key("k")
            .with_base_url(format!("http://{addr}"))
            .with_request_timeout(Duration::from_millis(200))
            .load_documents(),
    )
    .await
    .expect("request timeout should fire first");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("request failed"), "{err}");
    hold.abort();
}

#[tokio::test]
async fn rejects_bad_input_before_any_request() {
    // Nothing listens on port 1: these must fail before sending anything.
    let base = |source| {
        AssemblyAILoader::new(source)
            .with_api_key("k")
            .with_base_url("http://127.0.0.1:1")
    };
    let cases = [
        (base(AudioSource::url("  ")), "audio URL is empty"),
        (base(AudioSource::Bytes(vec![])), "audio is empty"),
        (
            base(AudioSource::transcript_id("")),
            "invalid AssemblyAI transcript id",
        ),
        (
            base(AudioSource::transcript_id("a/b")),
            "invalid AssemblyAI transcript id",
        ),
        (
            base(AudioSource::transcript_id("a?b")),
            "invalid AssemblyAI transcript id",
        ),
        (
            base(AudioSource::url("https://x")).with_api_key(""),
            "ASSEMBLYAI_API_KEY",
        ),
    ];
    for (loader, expected) in cases {
        let err = loader.load_documents().await.unwrap_err().to_string();
        assert!(err.contains(expected), "{expected}: {err}");
    }
    let err = base(AudioSource::file("/definitely/not/here.wav"))
        .load_documents()
        .await
        .unwrap_err();
    assert!(matches!(err, LoaderError::IOError(_)), "{err}");
}

#[test]
fn ids_and_options() {
    for ok in [ID, "abc", "A_b-9"] {
        assert!(check_id(ok).is_ok(), "{ok}");
    }
    for bad in ["", "a b", "a/b", "a.b", "é"] {
        assert!(check_id(bad).is_err(), "{bad}");
    }
    let l = AssemblyAILoader::new(AudioSource::url("u")).with_base_url("http://x/");
    assert_eq!(l.base_url, "http://x");
    assert_eq!(l.format, TranscriptFormat::Text);
    assert_eq!(l.poll_interval, Duration::from_secs(3));
    assert_eq!(AudioSource::Bytes(vec![1]).label(), "bytes");
}

#[tokio::test]
async fn load_and_split_chunks_the_transcript() {
    use crate::text_splitter::{SplitterOptions, TokenSplitter};

    let mut server = Server::new_async().await;
    let _get = mock_get(&mut server, "completed", 1).await;
    let splitter = TokenSplitter::new(
        SplitterOptions::default()
            .with_chunk_size(3)
            .with_chunk_overlap(0),
    );
    let mut stream = loader(&server, AudioSource::transcript_id(ID))
        .load_and_split(splitter)
        .await
        .unwrap();
    let mut chunks = Vec::new();
    while let Some(doc) = stream.next().await {
        chunks.push(doc.unwrap());
    }
    assert!(chunks.len() > 1, "{} chunks", chunks.len());
    assert_eq!(
        chunks
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<String>(),
        "Hello there. General Kenobi."
    );
    assert!(chunks
        .iter()
        .all(|d| d.metadata["transcript_id"] == json!(ID)));
}

/// `ASSEMBLYAI_API_KEY=... cargo test --lib assemblyai_live -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn assemblyai_live() {
    if std::env::var("ASSEMBLYAI_API_KEY").is_err() {
        eprintln!("ASSEMBLYAI_API_KEY not set, skipping");
        return;
    }
    let docs = AssemblyAILoader::new(AudioSource::url("https://assembly.ai/wildfires.mp3"))
        .with_format(TranscriptFormat::Sentences)
        .load_documents()
        .await
        .unwrap();
    assert!(!docs.is_empty());
    println!("{}", docs[0].page_content);
}
