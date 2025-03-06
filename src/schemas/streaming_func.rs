use futures::Future;
use std::pin::Pin;

pub type StreamingFunc =
    dyn FnMut(String) -> Pin<Box<dyn Future<Output = Result<(), ()>> + Send>> + Send;
