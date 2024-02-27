mod builder;
mod chain;
mod prompt;

pub use builder::*;
pub use chain::*;
pub use prompt::*;

const STOP_WORD: &str = "\nSQLResult:";
const SQL_CHAIN_DEFAULT_INPUT_KEY_QUERY: &str = "query";
const SQL_CHAIN_DEFAULT_INPUT_KEY_TABLE_NAMES: &str = "table_names_to_use";
const SQL_CHAIN_DEFAULT_OUTPUT_KEY: &str = "result";
const QUERY_PREFIX_WITH: &str = "\nSQLQuery:";
