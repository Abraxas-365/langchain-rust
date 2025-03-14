use std::collections::VecDeque;

use regex::Regex;
use serde_json::Value;

use crate::{agent::AgentError, schemas::agent::AgentEvent};

pub fn parse_agent_output(text: &str) -> Result<AgentEvent, AgentError> {
    let agent_event = parse_json_markdown(text)
        .or_else(|| parse_partial_json(text, false))
        .ok_or(AgentError::InvalidFormatError)?;
    let agent_event: AgentEvent =
        serde_json::from_value(agent_event).map_err(|_| AgentError::InvalidFormatError)?;
    Ok(agent_event)
}

fn parse_partial_json(s: &str, strict: bool) -> Option<Value> {
    // First, attempt to parse the string as-is.
    match serde_json::from_str::<Value>(s) {
        Ok(val) => return Some(val),
        Err(_) if !strict => (),
        Err(_) => return None,
    }

    let mut new_s = String::new();
    let mut stack: VecDeque<char> = VecDeque::new();
    let mut is_inside_string = false;
    let mut escaped = false;

    for char in s.chars() {
        match char {
            '"' if !escaped => is_inside_string = !is_inside_string,
            '{' if !is_inside_string => stack.push_back('}'),
            '[' if !is_inside_string => stack.push_back(']'),
            '}' | ']' if !is_inside_string => {
                if let Some(c) = stack.pop_back() {
                    if c != char {
                        return None; // Mismatched closing character
                    }
                } else {
                    return None; // Unbalanced closing character
                }
            }
            '\\' if is_inside_string => escaped = !escaped,
            _ => escaped = false,
        }
        new_s.push(char);
    }

    // Close any open structures.
    while let Some(c) = stack.pop_back() {
        new_s.push(c);
    }

    // Attempt to parse again.
    serde_json::from_str(&new_s).ok()
}

fn parse_json_markdown(json_markdown: &str) -> Option<Value> {
    let re = Regex::new(r"```(?:json)?\s*([\s\S]+?)\s*```").unwrap();
    if let Some(caps) = re.captures(json_markdown) {
        if let Some(json_str) = caps.get(1) {
            return parse_partial_json(json_str.as_str(), false);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use super::*;

    #[test]
    fn test_parse_agent_output() {
        let test_output = indoc! {r#"
            ```json
            {
                "action": "generate",
                "action_input": "Hello, world!"
            }
            ```
        "#};

        let parsed_output = parse_agent_output(test_output).unwrap();

        match parsed_output {
            AgentEvent::Action(agent_actions) => {
                assert!(agent_actions.len() == 1);
                let agent_action = &agent_actions[0];
                assert_eq!(agent_action.action, "generate");
                assert_eq!(agent_action.action_input, "Hello, world!");
            }
            AgentEvent::Finish(_) => panic!("Expected AgentEvent::Action, got AgentEvent::Finish"),
        }

        let test_final_answer = indoc! {r#"
            ```json
            {
                "final_answer": "Goodbye, world!"
            }
            ```
        "#};

        let parsed_output = parse_agent_output(test_final_answer).unwrap();

        match parsed_output {
            AgentEvent::Action(_) => panic!("Expected AgentEvent::Finish, got AgentEvent::Action"),
            AgentEvent::Finish(final_answer) => {
                assert_eq!(final_answer, "Goodbye, world!");
            }
        }
    }

    #[test]
    fn test_complicated_output() {
        let test_final_answer = indoc! {r#"
            ```json
            {
                "final_answer": [
                    {
                        "ingredients": "Universal ANI threshold validation: Established a 95–96% ANI species boundary across 6,787 prokaryotic genomes spanning 22 phyla, supported by empirical evidence of a distinct ANI distribution valley at this value, forming a universal genomic criterion supplanting DDH."
                    },
                    {
                        "ingredients": "Optimized 16S threshold via cross-validation: Derived a 98.65% 16S rRNA sequence similarity threshold for species demarcation through F-score optimization and logarithmic transformation of ANI-16S correlations, enabling alignment with genomic standards and resolving prior linear model discrepancies."
                    },
                    {
                        "ingredients": "Statistical methodology innovation: Introduced a precision-recall framework combining F-score maximization and cross-validation to objectively determine genomic-phenotypic species boundaries, overcoming subjective reliance on historical DDH values."
                    },
                    {
                        "ingredients": "Taxonomic reclassification mandates: Identified inconsistent species classifications (e.g., Bacillus anthracis-cereus, Shigella-E. coli) requiring genomic reevaluation as ANI exceeds 96% despite existing taxonomic separation, necessitating revised microbial nomenclature."
                    },
                    {
                        "ingredients": "16S taxonomic limitations: Revealed intra-species 16S rRNA heterogeneity (up to 9.8% in Halomicrobium) and genus-specific sequencing artifacts undermining solely marker-based classification, necessitating multi-genome analyses for accuracy."
                    },
                    {
                        "ingredients": "Genomic data quality criteria: Established stringent quality controls (>7× 16S sequencing depth, full-genome completion) to eliminate errors from low-coverage drafts (e.g., Neisseria meningitidis exceptions), ensuring valid ANI calculations."
                    },
                    {
                        "ingredients": "Bimodal ANI distribution proof: Empirically validated species boundary via bimodal ANI histograms displaying consistent inter-species valley at 95–96%, confirming universal lineage-independent validity across prokaryotes."
                    }
                ]
            }
            ```
        "#};

        let result = parse_agent_output(test_final_answer);

        println!("{:#?}", result);
    }
}
