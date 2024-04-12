use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::{
    language_models::{llm::LLM, GenerateResult},
    prompt::PromptArgs,
    prompt_args,
    schemas::{messages::Message, Document, StreamData},
    template_jinja2,
};

use super::{
    options::ChainCallOptions, Chain, ChainError, LLMChain, LLMChainBuilder, StuffDocument,
};

const DEFAULTCONDENSEQUESTIONTEMPLATE: &str = r#"Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{{chat_history}}
Follow Up Input: {{question}}
Standalone question:"#;

pub struct CondenseQuestionPromptBuilder {
    chat_history: String,
    question: String,
}
impl CondenseQuestionPromptBuilder {
    pub fn new() -> Self {
        Self {
            chat_history: "".to_string(),
            question: "".to_string(),
        }
    }

    pub fn question<S: Into<String>>(mut self, question: S) -> Self {
        self.question = question.into();
        self
    }

    pub fn chat_history(mut self, chat_history: &[Message]) -> Self {
        self.chat_history = Message::messages_to_string(chat_history);
        self
    }

    pub fn build(self) -> PromptArgs {
        prompt_args! {
            "chat_history" => self.chat_history,
            "question" => self.question
        }
    }
}

pub struct CondenseQuetionGeneratorChain {
    chain: LLMChain,
}

impl CondenseQuetionGeneratorChain {
    pub fn new<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        let condense_question_prompt_template =
            template_jinja2!(DEFAULTCONDENSEQUESTIONTEMPLATE, "chat_history", "question");

        let chain = LLMChainBuilder::new()
            .llm(llm)
            .prompt(condense_question_prompt_template)
            .build()
            .unwrap(); //Its safe to unwrap here because we are sure that the prompt and the LLM are
                       //set.
        Self { chain }
    }

    pub fn prompt_builder(&self) -> CondenseQuestionPromptBuilder {
        CondenseQuestionPromptBuilder::new()
    }
}

#[async_trait]
impl Chain for CondenseQuetionGeneratorChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        self.chain.call(input_variables).await
    }

    async fn stream(
        &self,
        input_variables: PromptArgs,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        self.chain.stream(input_variables).await
    }
}

const DEFAULT_STUFF_QA_TEMPLATE: &str = r#"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{{context}}

Question:{{question}}
Helpful Answer:
"#;

pub struct StuffQAPromptBuilder<'a> {
    input_documents: Vec<&'a Document>,
    question: String,
}

impl<'a> StuffQAPromptBuilder<'a> {
    pub fn new() -> Self {
        Self {
            input_documents: vec![],
            question: "".to_string(),
        }
    }

    pub fn documents(mut self, documents: &'a [Document]) -> Self {
        self.input_documents = documents.iter().collect();
        self
    }

    pub fn question<S: Into<String>>(mut self, question: S) -> Self {
        self.question = question.into();
        self
    }

    pub fn build(self) -> PromptArgs {
        prompt_args! {
            "input_documents" => self.input_documents,
            "question" => self.question
        }
    }
}

pub(crate) fn load_stuff_qa<L: Into<Box<dyn LLM>>>(
    llm: L,
    options: Option<ChainCallOptions>,
) -> StuffDocument {
    let default_qa_prompt_template =
        template_jinja2!(DEFAULT_STUFF_QA_TEMPLATE, "context", "question");

    let llm_chain_builder = LLMChainBuilder::new()
        .prompt(default_qa_prompt_template)
        .options(options.unwrap_or_default())
        .llm(llm)
        .build()
        .unwrap();

    let llm_chain = llm_chain_builder;

    StuffDocument::new(llm_chain)
}

#[cfg(test)]
mod tests {
    use crate::{
        chain::{Chain, StuffDocument},
        llm::openai::OpenAI,
        schemas::Document,
    };

    #[tokio::test]
    #[ignore]
    async fn test_qa() {
        let llm = OpenAI::default();
        let chain = StuffDocument::load_stuff_qa(llm);
        let input = chain
            .qa_prompt_builder()
            .documents(&[
                Document::new(format!(
                    "\nQuestion: {}\nAnswer: {}\n",
                    "Which is the favorite text editor of luis", "Nvim"
                )),
                Document::new(format!(
                    "\nQuestion: {}\nAnswer: {}\n",
                    "How old is Luis", "24"
                )),
            ])
            .question("How old is luis and whats his favorite text editor")
            .build();

        let ouput = chain.invoke(input).await.unwrap();

        println!("{}", ouput);
    }
}
