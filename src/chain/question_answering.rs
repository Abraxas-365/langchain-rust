use crate::{language_models::llm::LLM, template_jinja2};

use super::{LLMChain, LLMChainBuilder, StuffDocument};

const DEFAULT_STUFF_QA_TEMPLATE: &str = r#"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{{context}}

Question:{{question}}
Helpful Answer:
"#;

const DEFAULTCONDENSEQUESTIONTEMPLATE: &str = r#"Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{{chat_history}}
Follow Up Input: {{question}}
Standalone question:"#;

pub fn load_condense_question_generator<L: LLM + 'static>(llm: L) -> LLMChain {
    let condense_question_prompt_template =
        template_jinja2!(DEFAULTCONDENSEQUESTIONTEMPLATE, "chat_history", "question");

    LLMChainBuilder::new()
        .llm(llm)
        .prompt(condense_question_prompt_template)
        .build()
        .unwrap() //Its safe to unwrap here because we are sure that the prompt and the LLM are
                  //set.
}

pub fn load_stuff_qa<L: LLM + 'static>(llm: L) -> StuffDocument {
    let default_qa_prompt_template =
        template_jinja2!(DEFAULT_STUFF_QA_TEMPLATE, "context", "question");

    let llm_chain = LLMChainBuilder::new()
        .prompt(default_qa_prompt_template)
        .llm(llm)
        .build()
        .unwrap(); //Its safe to unwrap here because we are sure that the prompt and the LLM are set.

    StuffDocument::new(llm_chain)
}

#[cfg(test)]
mod tests {
    use crate::{
        chain::{load_stuff_qa, Chain},
        llm::openai::OpenAI,
        prompt_args,
        schemas::Document,
    };

    #[tokio::test]
    async fn test_qa() {
        let llm = OpenAI::default();
        let chain = load_stuff_qa(llm);

        let ouput = chain
            .invoke(prompt_args! {
                "input_documents"=>vec![
Document::new(format!("\nQuestion: {}\nAnswer: {}\n", "Which is the favorite text editor of luis", "Nvim")),
Document::new(format!("\nQuestion: {}\nAnswer: {}\n", "How old is Luis", "24")),
                ],
                "question"=>"How old is luis and whats his favorite text editor"
            })
            .await
            .unwrap();

        println!("{}", ouput);
    }
}
