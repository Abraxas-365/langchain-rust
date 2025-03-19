use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    input_variables,
    language_models::llm::LLM,
    llm::openai::OpenAI,
    prompt_template,
    schemas::{messages::Message, MessageOrTemplate, MessageTemplate, MessageType},
};

#[tokio::main]
async fn main() {
    //We can then initialize the model:
    // If you'd prefer not to set an environment variable you can pass the key in directly via the `openai_api_key` named parameter when initiating the OpenAI LLM class:
    //let open_ai = OpenAI::default().with_api_key("...");
    let open_ai = OpenAI::default();

    //Once you've installed and initialized the LLM of your choice, we can try using it! Let's ask it what LangSmith is - this is something that wasn't present in the training data so it shouldn't have a very good response.
    let resp = open_ai.invoke("What is rust").await.unwrap();
    println!("{}", resp);

    // We can also guide it's response with a prompt template. Prompt templates are used to convert raw user input to a better input to the LLM.
    let prompt = prompt_template![
        Message::new(
            MessageType::SystemMessage,
            "You are world class technical documentation writer."
        ),
        MessageTemplate::from_fstring(MessageType::HumanMessage, "{input}",)
    ];

    //We can now combine these into a simple LLM chain:

    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(open_ai.clone())
        .build()
        .unwrap();

    //We can now invoke it and ask the same question. It still won't know the answer, but it should respond in a more proper tone for a technical writer!

    match chain
        .invoke(&mut input_variables! {
            "input" => "Quien es el escritor de 20000 millas de viaje submarino",
        })
        .await
    {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }

    //If you want to prompt to have a list of messages you could use the `fmt_placeholder` macro

    let prompt = prompt_template![
        Message::new(
            MessageType::SystemMessage,
            "You are world class technical documentation writer."
        ),
        MessageOrTemplate::Placeholder("history".into()),
        MessageTemplate::from_fstring(MessageType::HumanMessage, "{input}",)
    ];

    // let chain = LLMChainBuilder::new()
    //     .prompt(prompt)
    //     .llm(open_ai)
    //     .build()
    //     .unwrap();
    // match chain
    //     .invoke(
    //         &mut input_variables! {
    //         "input" => "Who is the writer of 20,000 Leagues Under the Sea, and what is my name?",
    //         },
    //         vec![
    //             Message::new_human_message("My name is: luis"),
    //             Message::new_ai_message("Hi luis"),
    //         ],
    //     )
    //     .await
    // {
    //     Ok(result) => {
    //         println!("Result: {:?}", result);
    //     }
    //     Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    // }
}
