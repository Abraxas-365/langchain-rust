use std::{io::Cursor, process::Stdio};

use futures::StreamExt;
use langchain_rust::{
    document_loaders::{HtmlLoader, Loader},
    schemas::Document,
    text_splitter::{PlainTextSplitter, PlainTextSplitterOptions, TextSplitter},
    tools::{Text2SpeechOpenAI, Tool},
};
use serde_json::Value;
use tokio::{io::AsyncReadExt, process::Command};
use url::Url;

#[tokio::main]
async fn main() {
    // URL to generate audio from.
    let url = "https://en.m.wikivoyage.org/wiki/Seoul";
    let output_path = "output.mp3";

    // Use reqwest to fetch the raw HTML content.
    println!("Fetching URL: {}\n", url);
    let html = reqwest::get(url).await.unwrap().text().await.unwrap();

    // Use HtmlLoader to load the HTML content and extract plain text without html tags.
    let html_loader = HtmlLoader::new(Cursor::new(html), Url::parse(url).unwrap());
    let documents: Vec<Document> = html_loader
        .load()
        .await
        .unwrap()
        .map(|x| x.unwrap())
        .collect()
        .await;

    // Since OpenAI has limits for input text size, use PlainTextSplitter to split the text into
    // chunks that are acceptable by OpenAI.
    let splitter = PlainTextSplitter::new(
        PlainTextSplitterOptions::default()
            // NOTE: PlainTextSplitter doesn't handle unicode chars, so make
            // sure to put some buffer if you are using unicode characters.
            .with_chunk_size(3000)
            .with_chunk_overlap(0)
            .with_trim_chunks(true),
    );
    let text_chunks = splitter
        .split_documents(&documents)
        .await
        .unwrap()
        .into_iter()
        .take(2) // Take only 2 for now to save time and cost as example.
        .collect::<Vec<Document>>();

    // Loop through each text chunks and generate audio using OpenAI and save it to disk.
    for (i, chunk) in text_chunks.iter().enumerate() {
        println!(
            "Processing chunk {} of {} with chunk size {}: \n{}\n",
            i,
            text_chunks.len(),
            chunk.page_content.len(),
            &chunk.page_content
        );

        let openai = Text2SpeechOpenAI::default().with_path(format!("chunk_{}.mp3", i));
        let path = openai
            .call(Value::String(chunk.page_content.to_string()))
            .await
            .unwrap();

        let path = std::path::Path::new(&path).canonicalize().unwrap();
        println!("Chunk file saved at: {:?}\n\n", path);
    }

    // Use ffmpeg to concatenate all the audio chunks into a single audio file.
    // ffmpeg -hide_banner -i "concat:chunk_0.mp3|chunk_1.mp3" -acodec copy -y output.mp3
    let mut args = vec![];

    let chunks_paths_list = text_chunks
        .iter()
        .enumerate()
        .map(|(i, _)| format!("chunk_{}.mp3", i))
        .collect::<Vec<String>>()
        .join("|");

    args.extend_from_slice(&[
        "-hide_banner".into(),
        "-i".into(),
        format!("concat:{}", &chunks_paths_list),
        "-acodec".into(),
        "copy".into(),
        "-y".into(), // overwite output file
        output_path.into(),
    ]);

    println!(
        "Merging {} audio chunks using: ffmpeg {}\n",
        text_chunks.len(),
        &args.join(" ")
    );

    let mut child = Command::new("ffmpeg")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start ffmpeg process.");

    let mut stdout = child.stdout.take().expect("Failed to open stdout");
    let mut stderr = child.stderr.take().expect("Failed to open stderr");

    let stdout_handle = tokio::spawn(async move {
        let mut buffer = vec![0; 1024];
        while let Ok(size) = stdout.read(&mut buffer).await {
            if size == 0 {
                break;
            }
            let output = String::from_utf8_lossy(&buffer[..size]);
            print!("FFmpeg STDOUT: {}", output);
        }
    });

    let stderr_handle = tokio::spawn(async move {
        let mut buffer = vec![0; 1024];
        while let Ok(size) = stderr.read(&mut buffer).await {
            if size == 0 {
                break;
            }
            let error = String::from_utf8_lossy(&buffer[..size]);
            eprint!("FFmpeg STDERR: {}", error);
        }
    });

    let ffmpeg_exit_status = child.wait().await.unwrap();
    stdout_handle.await.unwrap();
    stderr_handle.await.unwrap();

    println!(
        "FFmpeg process finished with exit status {}",
        ffmpeg_exit_status
    );

    println!("Cleaning up intermediate audio chunk files...");
    for (i, _) in text_chunks.iter().enumerate() {
        let path = std::path::Path::new(&format!("chunk_{}.mp3", i))
            .canonicalize()
            .unwrap();
        tokio::fs::remove_file(path).await.unwrap();
    }
    println!("Cleaning up intermediate audio chunk files complete.");

    let path = std::path::Path::new(&output_path).canonicalize().unwrap();
    println!("Final audio saved at: {:?}", path);
}
