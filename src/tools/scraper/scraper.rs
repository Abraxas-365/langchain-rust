use async_trait::async_trait;
use scraper::{ElementRef, Html, Selector};
use serde_json::Value;
use std::{error::Error, sync::Arc, time::Duration};

use crate::tools::Tool;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_USER_AGENT: &str = concat!("langchain-rust/", env!("CARGO_PKG_VERSION"));
const DEFAULT_EXCLUDED_TAGS: [&str; 4] = ["script", "style", "noscript", "template"];

/// Fetches a web page and returns its visible text, whitespace collapsed.
///
/// By default it reads the whole `<body>` and skips `script`, `style`, `noscript` and
/// `template`. Everything is configurable:
///
/// ```rust
/// use langchain_rust::tools::WebScrapper;
/// use std::time::Duration;
///
/// let scraper = WebScrapper::new()
///     .with_selector("article, main")
///     .unwrap()
///     .with_excluded_tags(["script", "style", "nav", "footer"])
///     .with_user_agent("my-bot/1.0")
///     .with_timeout(Duration::from_secs(10))
///     .with_max_chars(4_000);
/// ```
///
/// As a tool, failures (bad URL, HTTP error status, timeout, no match) come back as an
/// `Error scraping <url>: ...` message rather than an error, so the agent can try
/// another page. [`scrape`](Self::scrape) returns them as errors.
#[derive(Debug, Clone)]
pub struct WebScrapper {
    client: reqwest::Client,
    selector: Option<(String, Selector)>,
    excluded_tags: Vec<String>,
    user_agent: String,
    timeout: Duration,
    max_chars: Option<usize>,
}

impl Default for WebScrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl WebScrapper {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            selector: None,
            excluded_tags: DEFAULT_EXCLUDED_TAGS.map(String::from).to_vec(),
            user_agent: DEFAULT_USER_AGENT.to_string(),
            timeout: DEFAULT_TIMEOUT,
            max_chars: None,
        }
    }

    /// Only read the elements matching this CSS selector (for example `article`,
    /// `main, #content` or `div.post > p`) instead of the whole body. Text of every match
    /// is joined in document order.
    pub fn with_selector(mut self, css: &str) -> Result<Self, Box<dyn Error>> {
        let selector =
            Selector::parse(css).map_err(|err| format!("invalid CSS selector {css:?}: {err}"))?;
        self.selector = Some((css.to_string(), selector));
        Ok(self)
    }

    /// Tags whose text is skipped, replacing the default list (`script`, `style`,
    /// `noscript`, `template`). Matching is case-insensitive.
    pub fn with_excluded_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.excluded_tags = tags
            .into_iter()
            .map(|t| t.into().to_ascii_lowercase())
            .collect();
        self
    }

    /// `User-Agent` header, `langchain-rust/<version>` by default. Some sites block
    /// unknown agents.
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Request timeout (default 30 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Cut the text to this many characters (default: no limit), so a long page does not
    /// fill the model's context. A cut ends with ` [truncated]`.
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = Some(max_chars);
        self
    }

    /// Fetches `url` and extracts its text.
    pub async fn scrape(&self, url: &str) -> Result<String, Box<dyn Error>> {
        // `Url::parse` also strips surrounding whitespace.
        let parsed = url::Url::parse(url).map_err(|err| format!("invalid URL: {err}"))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(format!(
                "only http and https URLs are supported, got {}",
                parsed.scheme()
            )
            .into());
        }
        let response = self
            .client
            .get(parsed)
            .header(reqwest::header::USER_AGENT, &self.user_agent)
            .timeout(self.timeout)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("HTTP {status}").into());
        }
        let html = response.text().await?;
        let text = self.extract(&html)?;
        Ok(self.truncate(text))
    }

    fn extract(&self, html: &str) -> Result<String, Box<dyn Error>> {
        let document = Html::parse_document(html);
        let mut parts = Vec::new();
        match &self.selector {
            Some((css, selector)) => {
                let mut matched = false;
                for element in document.select(selector) {
                    matched = true;
                    self.collect_text(&element, &mut parts);
                }
                if !matched {
                    return Err(format!("no element matches selector {css:?}").into());
                }
            }
            None => {
                let body = Selector::parse("body").expect("valid selector");
                for element in document.select(&body) {
                    self.collect_text(&element, &mut parts);
                }
            }
        }
        Ok(parts
            .iter()
            .flat_map(|part| part.split_whitespace())
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn collect_text(&self, element: &ElementRef, text: &mut Vec<String>) {
        for node in element.children() {
            if let Some(child) = ElementRef::wrap(node) {
                let tag = child.value().name();
                // html5ever lowercases tag names; the setter lowercases the list.
                if self.excluded_tags.iter().any(|t| t == tag) {
                    continue;
                }
                self.collect_text(&child, text);
            } else if let Some(t) = node.value().as_text() {
                text.push(t.text.to_string());
            }
        }
    }

    fn truncate(&self, text: String) -> String {
        match self.max_chars {
            Some(max) if text.chars().count() > max => {
                let cut: String = text.chars().take(max).collect();
                format!("{cut} [truncated]")
            }
            _ => text,
        }
    }
}

#[async_trait]
impl Tool for WebScrapper {
    fn name(&self) -> String {
        String::from("Web Scraper")
    }
    fn description(&self) -> String {
        String::from(
            "Web Scraper will scan a url and return the content of the web page.
		Input should be a working url.",
        )
    }
    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let input = input.as_str().ok_or("Invalid input")?;
        match self.scrape(input).await {
            Ok(content) => Ok(content),
            Err(e) => Ok(format!("Error scraping {}: {}\n", input, e)),
        }
    }
}

impl Into<Arc<dyn Tool>> for WebScrapper {
    fn into(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = "<html><head><title>T</title><style>.x{color:red}</style></head><body>\n\
        <nav>Home | About</nav><style>.y{margin:0}</style>\
        <article><h1>Title</h1>\t<p>Hello\n   <b>bold</b></p>\
        <script>var x = 1;</script><noscript>enable js</noscript><template>tpl</template>\
        <p class=\"note\">Note one</p></article>\
        <SCRIPT>var y = 2;</SCRIPT>\
        <footer>Footer <p class=\"note\">Note two</p></footer></body></html>";

    async fn serve(body: &str) -> (mockito::ServerGuard, mockito::Mock) {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/page")
            .with_header("content-type", "text/html")
            .with_body(body)
            .create_async()
            .await;
        (server, mock)
    }

    #[tokio::test]
    async fn test_scrape_url() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/")
            .with_status(200)
            .with_header("content-type", "text/plain")
            .with_body("<html><body>Hello World</body></html>")
            .create();

        let result = WebScrapper::new().call(&server.url()).await;
        assert_eq!(result.unwrap(), "Hello World");
        mock.assert();
    }

    #[tokio::test]
    async fn default_reads_body_and_skips_code_and_styles() {
        let (server, _m) = serve(PAGE).await;
        let content = WebScrapper::new()
            .scrape(&format!("{}/page", server.url()))
            .await
            .unwrap();
        assert_eq!(
            content,
            "Home | About Title Hello bold Note one Footer Note two"
        );
    }

    #[tokio::test]
    async fn selector_limits_extraction() {
        let (server, _m) = serve(PAGE).await;
        let url = format!("{}/page", server.url());
        let article = WebScrapper::new()
            .with_selector("article")
            .unwrap()
            .scrape(&url)
            .await
            .unwrap();
        assert_eq!(article, "Title Hello bold Note one");

        // Several matches are joined in document order.
        let notes = WebScrapper::new()
            .with_selector("p.note")
            .unwrap()
            .scrape(&url)
            .await
            .unwrap();
        assert_eq!(notes, "Note one Note two");

        let err = WebScrapper::new()
            .with_selector("section.missing")
            .unwrap()
            .scrape(&url)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("section.missing"), "{err}");
    }

    #[tokio::test]
    async fn excluded_tags_replace_the_default() {
        let (server, _m) = serve(PAGE).await;
        let url = format!("{}/page", server.url());
        let content = WebScrapper::new()
            .with_excluded_tags(["NAV", "footer", "script"])
            .scrape(&url)
            .await
            .unwrap();
        // style and noscript are no longer skipped. (html5ever keeps <template> content
        // out of the tree, so it never shows up.)
        assert_eq!(content, ".y{margin:0} Title Hello bold enable js Note one");
    }

    #[tokio::test]
    async fn sends_user_agent() {
        let mut server = mockito::Server::new_async().await;
        let default_ua = server
            .mock("GET", "/ua")
            .match_header("user-agent", DEFAULT_USER_AGENT)
            .with_body("<body>default</body>")
            .create_async()
            .await;
        let url = format!("{}/ua", server.url());
        assert_eq!(WebScrapper::new().scrape(&url).await.unwrap(), "default");
        default_ua.assert_async().await;

        let custom = server
            .mock("GET", "/ua2")
            .match_header("user-agent", "my-bot/1.0")
            .with_body("<body>custom</body>")
            .create_async()
            .await;
        let out = WebScrapper::new()
            .with_user_agent("my-bot/1.0")
            .scrape(&format!("{}/ua2", server.url()))
            .await
            .unwrap();
        assert_eq!(out, "custom");
        custom.assert_async().await;
    }

    #[tokio::test]
    async fn http_errors_are_reported_not_scraped() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/gone")
            .with_status(404)
            .with_body("<body>Not Found page text</body>")
            .create_async()
            .await;
        let url = format!("{}/gone", server.url());
        let err = WebScrapper::new()
            .scrape(&url)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("404"), "{err}");

        // The tool turns it into a message for the agent.
        let out = WebScrapper::new().call(&url).await.unwrap();
        assert!(
            out.starts_with(&format!("Error scraping {url}: HTTP 404")),
            "{out}"
        );
        assert!(!out.contains("Not Found page text"));
    }

    #[tokio::test]
    async fn max_chars_truncates_on_char_boundaries() {
        let (server, _m) = serve("<body>héllo wörld ünïcode</body>").await;
        let url = format!("{}/page", server.url());
        let out = WebScrapper::new()
            .with_max_chars(7)
            .scrape(&url)
            .await
            .unwrap();
        assert_eq!(out, "héllo w [truncated]");
        // Exactly at the limit: untouched.
        let out = WebScrapper::new()
            .with_max_chars(19)
            .scrape(&url)
            .await
            .unwrap();
        assert_eq!(out, "héllo wörld ünïcode");
    }

    #[tokio::test]
    async fn times_out() {
        // Accepts connections and never answers.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hold = tokio::spawn(async move {
            let mut open = Vec::new();
            while let Ok((socket, _)) = listener.accept().await {
                open.push(socket);
            }
        });
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            WebScrapper::new()
                .with_timeout(Duration::from_millis(200))
                .scrape(&format!("http://{addr}/")),
        )
        .await
        .expect("the 200 ms timeout should fire, not the 30 s default");
        assert!(result.is_err());
        hold.abort();
    }

    #[tokio::test]
    async fn rejects_bad_urls_and_selectors() {
        for (url, expected) in [
            ("not a url", "invalid URL"),
            ("file:///etc/passwd", "only http and https"),
            ("ftp://example.com/x", "only http and https"),
        ] {
            let err = WebScrapper::new()
                .scrape(url)
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains(expected), "{url}: {err}");
        }
        let err = WebScrapper::new()
            .with_selector("div[")
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid CSS selector"), "{err}");
    }

    #[tokio::test]
    async fn trims_input_and_accepts_tool_call_json() {
        let (server, mock) = serve("<body>ok</body>").await;
        let url = format!("{}/page", server.url());
        let scraper = WebScrapper::default();
        assert_eq!(scraper.call(&format!("  {url}\n")).await.unwrap(), "ok");
        assert_eq!(
            scraper
                .call(&serde_json::json!({ "input": url }).to_string())
                .await
                .unwrap(),
            "ok"
        );
        mock.expect(2).assert_async().await;
    }

    #[tokio::test]
    #[ignore = "network: fetches example.com and wikipedia.org"]
    async fn scrape_live() {
        let text = WebScrapper::new()
            .scrape("https://example.com")
            .await
            .unwrap();
        assert!(
            text.starts_with("This domain is for use in documentation"),
            "{text}"
        );

        let title = WebScrapper::new()
            .with_selector("h1#firstHeading")
            .unwrap()
            .scrape("https://en.wikipedia.org/wiki/Rust_(programming_language)")
            .await
            .unwrap();
        assert_eq!(title, "Rust (programming language)");
    }
}
