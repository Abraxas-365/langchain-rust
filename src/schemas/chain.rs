use tokio::sync::mpsc;

pub enum ChainResponse {
    Text(String),
    Stream(mpsc::Receiver<Result<String, reqwest_eventsource::Error>>),
}
