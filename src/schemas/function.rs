use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionCallResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: FunctionDetail,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionDetail {
    pub name: String,
    ///this should be an string, and this should be passed to the tool, to
    ///then be deserilised inside the tool, becuase just the tools knows the names of the arguments.
    pub arguments: String,
}

impl FromStr for FunctionCallResponse {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
