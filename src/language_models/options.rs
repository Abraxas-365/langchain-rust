use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

// FunctionCallBehavior is the behavior to use when calling functions.
#[derive(Clone, Copy, Debug)]
enum FunctionCallBehavior {
    None,
    Auto,
}

// FunctionDefinition is a definition of a function that can be called by the model.
#[derive(Clone, Debug)]
struct FunctionDefinition {
    name: String,
    description: String,
    parameters: HashMap<String, String>, // Assuming parameters as a simple key-value pair for simplicity
}

// CallOptions is a set of options for calling models, Not all the llms have all this options.
#[derive(Clone)]
struct CallOptions {
    candidate_count: Option<usize>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    stop_words: Option<Vec<String>>,
    streaming_func: Option<Arc<Mutex<dyn FnMut(Vec<u8>) -> Result<(), ()> + Send>>>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: Option<usize>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    n: Option<usize>,
    repetition_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    functions: Option<Vec<FunctionDefinition>>,
    function_call_behavior: Option<FunctionCallBehavior>,
}

impl CallOptions {
    fn new() -> Self {
        CallOptions {
            candidate_count: None,
            max_tokens: None,
            temperature: None,
            stop_words: None,
            streaming_func: None,
            top_k: None,
            top_p: None,
            seed: None,
            min_length: None,
            max_length: None,
            n: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            functions: None,
            function_call_behavior: None,
        }
    }
}

// CallOption is a function that configures a CallOptions.
type CallOption = Box<dyn FnOnce(&mut CallOptions)>;

// Builder pattern for CallOptions
impl CallOptions {
    fn with_option(mut self, opt: CallOption) -> Self {
        opt(&mut self);
        self
    }
}

fn with_max_tokens(max_tokens: usize) -> CallOption {
    Box::new(move |opts| opts.max_tokens = Some(max_tokens))
}

fn with_candidate_count(candidate_count: usize) -> CallOption {
    Box::new(move |opts| opts.candidate_count = Some(candidate_count))
}

fn with_temperature(temperature: f64) -> CallOption {
    Box::new(move |opts| opts.temperature = Some(temperature))
}

fn with_stop_words(stop_words: Vec<String>) -> CallOption {
    Box::new(move |opts| opts.stop_words = Some(stop_words))
}

fn with_streaming_func<F>(func: F) -> CallOption
where
    F: FnMut(Vec<u8>) -> Result<(), ()> + Send + 'static,
{
    let func = Arc::new(Mutex::new(func));
    Box::new(move |opts| opts.streaming_func = Some(func.clone()))
}

fn with_top_k(top_k: usize) -> CallOption {
    Box::new(move |opts| opts.top_k = Some(top_k))
}

fn with_top_p(top_p: f64) -> CallOption {
    Box::new(move |opts| opts.top_p = Some(top_p))
}

fn with_seed(seed: usize) -> CallOption {
    Box::new(move |opts| opts.seed = Some(seed))
}

fn with_min_length(min_length: usize) -> CallOption {
    Box::new(move |opts| opts.min_length = Some(min_length))
}

fn with_max_length(max_length: usize) -> CallOption {
    Box::new(move |opts| opts.max_length = Some(max_length))
}

fn with_n(n: usize) -> CallOption {
    Box::new(move |opts| opts.n = Some(n))
}

fn with_repetition_penalty(repetition_penalty: f64) -> CallOption {
    Box::new(move |opts| opts.repetition_penalty = Some(repetition_penalty))
}

fn with_frequency_penalty(frequency_penalty: f64) -> CallOption {
    Box::new(move |opts| opts.frequency_penalty = Some(frequency_penalty))
}

fn with_presence_penalty(presence_penalty: f64) -> CallOption {
    Box::new(move |opts| opts.presence_penalty = Some(presence_penalty))
}

fn with_functions(functions: Vec<FunctionDefinition>) -> CallOption {
    Box::new(move |opts| opts.functions = Some(functions))
}

fn with_function_call_behavior(behavior: FunctionCallBehavior) -> CallOption {
    Box::new(move |opts| opts.function_call_behavior = Some(behavior))
}
