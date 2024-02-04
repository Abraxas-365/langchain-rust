use reqwest_eventsource::EventSource;

pub enum LlmResponse {
    Text(String),
    Stream(EventSource),
}
