use std::sync::LazyLock;

use regex::Regex;

use crate::{
    agent::AgentError,
    schemas::agent::{AgentAction, AgentEvent, AgentFinish},
};

/// Marks the answer in a ReAct response.
pub const FINAL_ANSWER: &str = "Final Answer:";

const MISSING_ACTION: &str = "Invalid Format: Missing 'Action:' after 'Thought:'. To use a tool, write 'Action:' and 'Action Input:'; if you know the answer, write 'Final Answer:' followed by it.";
const MISSING_ACTION_INPUT: &str =
    "Invalid Format: Missing 'Action Input:' after 'Action:'. Write the tool's input after 'Action Input:'.";
const ANSWER_AND_ACTION: &str = "Parsing LLM output produced both a final answer and a parse-able action. Write either an action or 'Final Answer:', not both.";

static ACTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)")
        .expect("valid regex")
});
static ACTION_ONLY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)Action\s*\d*\s*:").expect("valid regex"));

/// Parses the text of a ReAct (MRKL) model turn:
///
/// ```text
/// Thought: I should look this up
/// Action: search
/// Action Input: rust release date
/// ```
///
/// becomes an [`AgentEvent::Action`] (the whole text is its `log`), and a turn with
/// `Final Answer: ...` becomes an [`AgentEvent::Finish`]. Anything else is an
/// [`AgentError::OutputParse`], which
/// [`AgentExecutor::with_parser_error_handler`](crate::agent::AgentExecutor::with_parser_error_handler)
/// can send back to the model.
#[derive(Debug, Clone, Default)]
pub struct ReActOutputParser;

impl ReActOutputParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, text: &str) -> Result<AgentEvent, AgentError> {
        let answer_at = text.find(FINAL_ANSWER);
        if let Some(action) = ACTION.captures(text) {
            if let Some(answer_at) = answer_at {
                // An answer followed by a made-up next step: keep the answer.
                if answer_at < action.get(0).map_or(0, |m| m.start()) {
                    let start = answer_at + FINAL_ANSWER.len();
                    let end = text[start..].find("\n\n").map_or(text.len(), |i| start + i);
                    return Ok(finish(&text[start..end]));
                }
                return Err(parse_error(ANSWER_AND_ACTION, text));
            }
            let tool = action[1].trim().to_string();
            let mut input = &action[2];
            // Models that ignore the stop word go on to invent the observation.
            if let Some(cut) = input.find("\nObservation") {
                input = &input[..cut];
            }
            let tool_input = input.trim().trim_matches('"').to_string();
            return Ok(AgentEvent::Action(vec![AgentAction {
                tool,
                tool_input,
                log: text.to_string(),
            }]));
        }
        if let Some(answer_at) = answer_at {
            // The last answer wins, as in LangChain.
            let last = text.rfind(FINAL_ANSWER).unwrap_or(answer_at);
            return Ok(finish(&text[last + FINAL_ANSWER.len()..]));
        }
        if !ACTION_ONLY.is_match(text) {
            return Err(parse_error(MISSING_ACTION, text));
        }
        Err(parse_error(MISSING_ACTION_INPUT, text))
    }
}

fn finish(answer: &str) -> AgentEvent {
    AgentEvent::Finish(AgentFinish {
        output: answer.trim().to_string(),
    })
}

fn parse_error(message: &str, text: &str) -> AgentError {
    AgentError::OutputParse {
        message: message.to_string(),
        llm_output: text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(text: &str) -> (String, String, String) {
        match ReActOutputParser::new().parse(text).unwrap() {
            AgentEvent::Action(mut actions) => {
                assert_eq!(actions.len(), 1);
                let a = actions.remove(0);
                (a.tool, a.tool_input, a.log)
            }
            other => panic!("expected an action, got {other:?}"),
        }
    }

    fn answer(text: &str) -> String {
        match ReActOutputParser::new().parse(text).unwrap() {
            AgentEvent::Finish(finish) => finish.output,
            other => panic!("expected a finish, got {other:?}"),
        }
    }

    fn error(text: &str) -> String {
        match ReActOutputParser::new().parse(text).unwrap_err() {
            AgentError::OutputParse {
                message,
                llm_output,
            } => {
                assert_eq!(llm_output, text);
                message
            }
            other => panic!("expected OutputParse, got {other:?}"),
        }
    }

    #[test]
    fn parses_an_action() {
        let text = " I should search.\nAction: search\nAction Input: rust 2024";
        assert_eq!(
            action(text),
            ("search".into(), "rust 2024".into(), text.into())
        );
    }

    #[test]
    fn action_variants() {
        // Numbered, extra whitespace, quoted input, multi-line input.
        assert_eq!(
            action("Action 1 :  calc \n Action 1 Input 1:  \"2 + 2\" ").0,
            "calc"
        );
        assert_eq!(action("Action: calc\nAction Input: \"2 + 2\"").1, "2 + 2");
        assert_eq!(
            action("Action: write\nAction Input: line one\nline two").1,
            "line one\nline two"
        );
        assert_eq!(action("Action: noop\nAction Input:").1, "");
        // Only the outer quotes go, as in LangChain.
        assert_eq!(
            action("Action: say\nAction Input: \"a\" and \"b\"").1,
            "a\" and \"b"
        );
    }

    #[test]
    fn input_stops_at_an_invented_observation() {
        let text = "Action: search\nAction Input: weather\nObservation: sunny\nThought: done";
        let (tool, input, log) = action(text);
        assert_eq!((tool.as_str(), input.as_str()), ("search", "weather"));
        assert_eq!(log, text, "the log keeps the full text");
    }

    #[test]
    fn parses_a_final_answer() {
        assert_eq!(answer(" I know.\nFinal Answer: 42 "), "42");
        assert_eq!(
            answer("Final Answer: first\nFinal Answer: second"),
            "second"
        );
        assert_eq!(
            answer("Final Answer: a\nmulti-line\nanswer"),
            "a\nmulti-line\nanswer"
        );
    }

    #[test]
    fn answer_before_an_invented_action_wins() {
        let text = "Final Answer: 42\n\nThought: more\nAction: search\nAction Input: x";
        assert_eq!(answer(text), "42");
    }

    #[test]
    fn action_before_an_answer_is_an_error() {
        let text = "Action: search\nAction Input: x\nObservation: y\nFinal Answer: z";
        assert_eq!(error(text), ANSWER_AND_ACTION);
    }

    #[test]
    fn malformed_turns_are_parse_errors() {
        assert_eq!(error("I think the answer is 42."), MISSING_ACTION);
        assert_eq!(error("Thought: hmm\nAction: search"), MISSING_ACTION_INPUT);
        assert_eq!(error(""), MISSING_ACTION);
    }

    #[test]
    fn parse_error_display_shows_the_output() {
        let err = ReActOutputParser::new().parse("just text").unwrap_err();
        let shown = err.to_string();
        assert!(shown.contains(MISSING_ACTION), "{shown}");
        assert!(shown.contains("just text"), "{shown}");
    }
}
