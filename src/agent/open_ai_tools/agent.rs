use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

use crate::{
    agent::{
        agent::{box_plan_events, PlanEvent, PlanEventStream},
        Agent, AgentError,
    },
    chain::Chain,
    fmt_message, fmt_placeholder, fmt_template,
    language_models::{GenerateResult, LLMEvent},
    message_formatter,
    prompt::{HumanMessagePromptTemplate, MessageFormatterStruct, PromptArgs},
    schemas::{
        agent::{AgentAction, AgentEvent, AgentFinish, LogTools},
        messages::Message,
        ToolCall,
    },
    template_jinja2,
    tools::Tool,
};

pub struct OpenAiToolAgent {
    pub(crate) chain: Box<dyn Chain>,
    pub(crate) tools: Vec<Arc<dyn Tool>>,
}

impl OpenAiToolAgent {
    pub fn create_prompt(prefix: &str) -> Result<MessageFormatterStruct, AgentError> {
        let prompt = message_formatter![
            fmt_message!(Message::new_system_message(prefix)),
            fmt_placeholder!("chat_history"),
            fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
                "{{input}}",
                "input"
            ))),
            fmt_placeholder!("agent_scratchpad")
        ];

        Ok(prompt)
    }

    fn construct_scratchpad(
        &self,
        intermediate_steps: &[(AgentAction, String)],
    ) -> Result<Vec<Message>, AgentError> {
        let mut thoughts: Vec<Message> = Vec::new();

        let mut tools_ai_message_seen: HashMap<String, ()> = HashMap::default();
        for (action, observation) in intermediate_steps {
            // Deserialize directly and embed in method calls to streamline code.
            // Extract the tool ID and tool calls from the log.
            let LogTools { tool_id, tools } = serde_json::from_str(&action.log)?;
            let tools_vec: Vec<ToolCall> = serde_json::from_str(&tools)?;

            // one action can trigger multiple observations.  But make sure we add it to
            // the scratchpad before the related observations.  There can also be multiple
            // different actions in the same thought chain.
            if tools_ai_message_seen.insert(tools, ()).is_none() {
                thoughts.push(Message::new_ai_message_with_tool_calls("", &tools_vec));
            }

            // Add a tool message for each observation. Observation is the ouput of the tool call.
            // tool_id is the id of the tool.
            thoughts.push(Message::new_tool_message(observation, tool_id));
        }

        Ok(thoughts)
    }
}

#[async_trait]
impl Agent for OpenAiToolAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError> {
        let inputs = self.plan_inputs(intermediate_steps, inputs)?;
        let result = self.chain.call(inputs).await?;
        Self::event_from_result(result)
    }

    /// Streams the model's text as it arrives (with native tool calling that is the
    /// answer, or a short preamble like "Let me check" before tool calls).
    fn plan_stream<'a>(
        &'a self,
        intermediate_steps: &'a [(AgentAction, String)],
        inputs: PromptArgs,
    ) -> PlanEventStream<'a> {
        box_plan_events(async_stream::try_stream! {
            let inputs = self.plan_inputs(intermediate_steps, inputs)?;
            let mut events = self.chain.call_stream(inputs);
            // LLMs without native tool calling write the calls as a JSON array in the text.
            // Output starting with `[` is held back until the plan is known, so that JSON
            // never reaches the caller as answer text.
            let mut held = String::new();
            let mut hold: Option<bool> = None;
            while let Some(event) = events.next().await {
                match event? {
                    LLMEvent::Text(text) => {
                        if hold == Some(false) {
                            yield PlanEvent::Text(text);
                            continue;
                        }
                        held.push_str(&text);
                        if let Some(first) = held.trim_start().chars().next() {
                            hold = Some(first == '[');
                            if first != '[' {
                                yield PlanEvent::Text(std::mem::take(&mut held));
                            }
                        }
                    }
                    LLMEvent::Done(result) => {
                        let event = Self::event_from_result(result)?;
                        if matches!(event, AgentEvent::Finish(_)) && !held.is_empty() {
                            yield PlanEvent::Text(held);
                        }
                        yield PlanEvent::Done(event);
                        return;
                    }
                }
            }
            Err(AgentError::OtherError("LLM stream ended without a result".into()))?;
        })
    }

    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}

impl OpenAiToolAgent {
    fn plan_inputs(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        mut inputs: PromptArgs,
    ) -> Result<PromptArgs, AgentError> {
        let scratchpad = self.construct_scratchpad(intermediate_steps)?;
        inputs.insert("agent_scratchpad".to_string(), json!(scratchpad));
        Ok(inputs)
    }

    fn event_from_result(result: GenerateResult) -> Result<AgentEvent, AgentError> {
        // Providers fill `tool_calls`; LLMs that only encode calls as JSON text in
        // `generation` (custom impls, older code) still work through the fallback.
        let calls = if result.has_tool_calls() {
            Some(result.tool_calls)
        } else {
            ToolCall::parse_list(&result.generation).filter(|c| !c.is_empty())
        };
        let Some(calls) = calls else {
            return Ok(AgentEvent::Finish(AgentFinish {
                output: result.generation,
            }));
        };

        // Every action carries the full list: the scratchpad replays it as one AI message.
        let tools = ToolCall::list_to_string(&calls);
        let mut actions: Vec<AgentAction> = Vec::new();
        for call in calls {
            let log = LogTools {
                tool_id: call.id,
                tools: tools.clone(),
            };
            actions.push(AgentAction {
                tool: call.name,
                tool_input: call.arguments,
                log: serde_json::to_string(&log)?, //We send this as string to minimise changes
            });
        }
        Ok(AgentEvent::Action(actions))
    }
}
