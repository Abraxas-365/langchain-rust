// To run this example execute: cargo run --example sql_chain --features postgres

#[cfg(feature = "postgres")]
use langchain_rust::{
    chain::{Chain, SQLDatabaseChainBuilder},
    llm::openai::OpenAI,
    tools::{postgres::PostgreSQLEngine, SQLDatabaseBuilder},
};

#[cfg(feature = "postgres")]
use std::io::{self, Write}; // Include io Library for terminal input

#[cfg(feature = "postgres")]
#[tokio::main]
async fn main() {
    use langchain_rust::schemas::InputVariables;

    let llm = OpenAI::default();

    let db = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let engine = PostgreSQLEngine::new(&db).await.unwrap();
    let db = SQLDatabaseBuilder::new(engine).build().await.unwrap();
    let chain = SQLDatabaseChainBuilder::new()
        .llm(llm)
        .top_k(4)
        .database(db)
        .build()
        .expect("Failed to build LLMChain");

    print!("Please enter a question: ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    let input = input.trim();
    let mut input_variables: InputVariables = chain.prompt_builder().query(input).build().into();
    match chain.invoke(&mut input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}

#[cfg(not(feature = "postgres"))]
fn main() {
    println!("This example requires the 'postgres' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example sql_chain --features postgres");
}

//You can use this docker migrations for example, you can ask , whats the phone number of John

// -- Migrations file
//
// -- Create the 'users' table
// CREATE TABLE users (
//     id serial PRIMARY KEY,
//     name varchar(255),
//     address text
// );
//
// -- Create the 'more_info' table
// CREATE TABLE more_info (
//     id serial PRIMARY KEY,
//     user_id int references users(id),
//     ig_nickname varchar(255),
//     phone_number varchar(255)
// );
//
//
// -- Dummy Data
//
// -- Inserting into 'users' table
// INSERT INTO users(name, address)
// VALUES
//     ('John Doe', '123 Main St'),
//     ('Jane Doe', '456 Oak St'),
//     ('Jim Doe', '789 Pine St');
//
// -- Inserting into 'more_info' table
// INSERT INTO more_info(user_id, ig_nickname, phone_number)
// VALUES
//     (1, 'john_ig', '123-456-7890'),
//     (2, 'jane_ig', '456-789-0123'),
//     (3, 'jim_ig', '789-012-3456');
