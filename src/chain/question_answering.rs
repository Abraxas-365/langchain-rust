use std::{error::Error, pin::Pin};

use crate::{
    input_variables,
    language_models::{llm::LLM, GenerateResult},
    schemas::{
        messages::Message, Document, InputVariables, MessageTemplate, MessageType, StreamData,
    },
};
use async_trait::async_trait;
use futures::Stream;

use super::{
    options::ChainCallOptions, Chain, ChainError, LLMChain, LLMChainBuilder, StuffDocument,
};

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

    pub fn build(self) -> InputVariables {
        input_variables! {
            "chat_history" => self.chat_history,
            "question" => self.question
        }
    }
}

impl Default for CondenseQuestionPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CondenseQuestionGeneratorChain {
    chain: LLMChain,
}

impl CondenseQuestionGeneratorChain {
    pub fn new<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        let condense_question_prompt_template = MessageTemplate::from_jinja2(
            MessageType::SystemMessage,
            r#"
            Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

            Chat History:
            {{chat_history}}
            Follow Up Input: {{question}}
            Standalone question:
            "#,
        );

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
impl Chain for CondenseQuestionGeneratorChain {
    async fn call(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<GenerateResult, ChainError> {
        self.chain.call(input_variables).await
    }

    async fn stream(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        self.chain.stream(input_variables).await
    }

    fn log_messages(&self, inputs: &InputVariables) -> Result<(), Box<dyn Error>> {
        self.chain.log_messages(inputs)
    }
}

pub struct StuffQABuilder {
    input_documents: Vec<Document>,
    question: String,
}

impl StuffQABuilder {
    pub fn new() -> Self {
        Self {
            input_documents: vec![],
            question: "".to_string(),
        }
    }

    pub fn documents(mut self, documents: &[Document]) -> Self {
        self.input_documents = documents.to_vec();
        self
    }

    pub fn question<S: Into<String>>(mut self, question: S) -> Self {
        self.question = question.into();
        self
    }

    pub fn build(self) -> InputVariables {
        input_variables! {
            "documents" => self.input_documents.iter().map(|doc| doc.page_content.clone()).collect::<Vec<String>>().join("\n"),
            "question" => self.question
        }
    }
}

impl Default for StuffQABuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn load_stuff_qa<L: Into<Box<dyn LLM>>>(
    llm: L,
    options: Option<ChainCallOptions>,
) -> StuffDocument {
    let default_qa_prompt_template = MessageTemplate::from_jinja2(
        MessageType::SystemMessage,
        r#"
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {{context}}
        
        Question:{{question}}
        Helpful Answer:
        "#,
    );

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
        chain::{Chain, StuffDocument, StuffQABuilder},
        llm::openai::OpenAI,
        schemas::Document,
    };

    #[tokio::test]
    #[ignore]
    async fn test_qa() {
        let llm = OpenAI::default();
        let chain = StuffDocument::load_stuff_qa(llm);
        let mut input = StuffQABuilder::new()
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

        let ouput = chain.invoke(&mut input).await.unwrap();

        println!("{}", ouput);
    }
}
