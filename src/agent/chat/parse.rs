use std::collections::VecDeque;

use regex::Regex;
use serde_json::Value;

use crate::schemas::{agent::AgentEvent, AgentAction};

pub fn parse_agent_output(text: &str) -> AgentEvent {
    parse_json_markdown(text)
        .or_else(|| parse_partial_json(text, false))
        .and_then(|agent_event| serde_json::from_value(agent_event).ok())
        .or_else(|| parse_with_regex(text))
        .unwrap_or_else(|| AgentEvent::Finish(text.into()))
}

fn fix_text(text: &str) -> String {
    let re = Regex::new(r"\\(.)").unwrap();
    re.replace_all(text, |caps: &regex::Captures| match &caps[1] {
        "n" => "\n".to_owned(),
        "t" => "\t".to_owned(),
        "r" => "\r".to_owned(),
        other => other.to_owned(), // Leave unknown sequences unchanged
    })
    .to_string()
}

fn parse_with_regex(text: &str) -> Option<AgentEvent> {
    let final_answer_re = Regex::new(r#"(?m)"final_answer"\s*:\s*"(.*)"\s*\n"#).unwrap();
    let action_regex = Regex::new(r#"(?m)"action"\s*:\s*"(.*)"\s*\n"#).unwrap();
    let action_input_regex = Regex::new(r#"(?m)"action_input"\s*:\s*"(.*)"\s*\n"#).unwrap();

    if let Some(final_answer) = final_answer_re.captures(text) {
        let final_answer = final_answer.get(1)?.as_str();
        Some(AgentEvent::Finish(fix_text(final_answer)))
    } else if let (Some(action), Some(action_input)) = (
        action_regex.captures(text),
        action_input_regex.captures(text),
    ) {
        let action = action.get(1)?.as_str();
        let action_input = action_input.get(1)?.as_str();
        Some(AgentEvent::Action(vec![AgentAction {
            id: uuid::Uuid::new_v4().to_string(),
            action: fix_text(action),
            action_input: serde_json::from_str(action_input).ok()?,
        }]))
    } else {
        None
    }
}

fn parse_partial_json(s: &str, strict: bool) -> Option<Value> {
    if let Ok(val) = serde_json::from_str::<Value>(s) {
        return Some(val);
    }

    if strict {
        return None;
    }

    // Step 2: Try parsing the cleaned version
    let cleaned = remove_trailing_commas(s);
    if let Ok(val) = serde_json::from_str::<Value>(&cleaned) {
        return Some(val);
    }

    // Step 3: Attempt to balance braces/brackets
    let balanced = balance_parenthesis(&cleaned);
    serde_json::from_str(&balanced).ok()
}

fn remove_trailing_commas(s: &str) -> String {
    let mut cleaned = String::new();
    let mut chars = s.chars();
    let mut inside_string = false;
    let mut escaped = false;

    while let Some(c) = chars.next() {
        match c {
            '"' if !escaped => {
                inside_string = !inside_string;
                cleaned.push(c);
            }
            '\\' if inside_string => {
                escaped = !escaped;
                cleaned.push(c);
                continue;
            }
            ',' if !inside_string => {
                // Peek ahead for } or ]
                if let Some(next_non_ws) = chars.clone().find(|c| !c.is_whitespace()) {
                    if next_non_ws == '}' || next_non_ws == ']' {
                        // Skip this comma
                        continue;
                    }
                }
            }
            _ => {
                escaped = false;
                cleaned.push(c);
            }
        }
    }

    cleaned
}

fn balance_parenthesis(s: &str) -> String {
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
                        return s.into(); // Mismatched closing character, return unmodified
                    }
                } else {
                    return s.into(); // Unbalanced closing character, return unmodified
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

    new_s
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

        let parsed_output = parse_agent_output(test_output);

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

        let parsed_output = parse_agent_output(test_final_answer);

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

        if let AgentEvent::Finish(final_answer) = result {
            println!("{}", final_answer);
        } else {
            panic!("Expected AgentEvent::Finish, got {:?}", result);
        }
    }

    #[test]
    fn test_other_output() {
        let test_final_answer: &str = indoc! {r#"
            ```json
            {
            "final_answer": "\## Thinking process\n\n### Step 1:\n- Proposal Candidate 1: Combining SetA’s `ANI 95-96% as universal boundary\` {A1} with SetB’s `log-transformed linear correlation between ANI and 16S\` {B3} and `statistical statistical framework for threshold validation\` {B5} to unify genomic and 16S classifications while avoiding sampling bias. \n- Proposal Candidate 2: Exploring discrepancies caused by intragenomic 16S variability (SetA4 和 SetB2/4) by integrating heterogeneity analyses {B4}, cross-disciplinary examples {B2}, and statistical rigor {B5} to establish multi-locus metrics for species delineation.\n- Proposal Candidate 3: Improving statistical validation methods using SetA2’s cross-validation and SetB5’s formal statistical framework to create the gold-standard accuracy for genomic/prokaryotic species boundaries.\n- Proposal Candidate 4: Creating quality control protocols by combining SetA5’s genome filtering criteria and SetB6’s sequencing standards to eliminate bias in threshold testing.\n- Proposal Candidate 5: Genomically reclassify extremophiles (SetB7) using ANI data {A1} and adjustment for discrepancies (SetA4) to address taxonomic inconsistencies.\n- Proposal Candidate 6: Evaluating the `hollow at ANI 95-96% {A3} THROUGH SETB에의한 log-linear model {B3} AND 多 factor in technical complexities, such as sequencing errors DISCLAIMED by rigorous quality control {A5 - B6}.\n\n### Step 2:\n- Enhancing Proposal Candidate 1: Incorporating SetB1’s `1 million pairwise comparisons for 16S准确性\` AND statistical precision-recall curves {A2} to ensure ANI/16S Correlation’s robustness across all prokaryotes.\n- Enhancing Proposal Candidate 2: Adding SetA’s `genetic quality metrics\` {A5} AND SetB’s统计" mathematical framework\` to quantify the heterogeneity’s impact on thresholds.\n- Enhancing Proposal Candidate 3: Supplementing with SetB3’s logarithmic analysis to unify non-linear relationships, enabling broader applicability of thresholds AND QUANTIFICATION 统一性.\n- Enhancing Proposal Candidate 5: Using SetB3’s modelo和 log-linear correlations to clarify which borderline cases BEST WARRANT reclassification.\n- Splitting Proposal Candidate 4: Forming独立的 PROMAA4L about data purification ({A5 - B6}) AND the sibling proposal that integrates these with fundamental threshold validation (Proposal1 and 5).\n- Forming PROPS Focused on heterogeneity mechanisms: adding SetB’s `extremophiles’高 rRNA异质性 needing reclassification\` {B7} to discrepancy studies (Proposal2와5) to?>### Step 3:\n- 각 proposal’s 구성 ideas:\n\n1. 자 props1 Enhanced: {A1, A2, A5, B1, B3, B5, B6}\n2. Props2-1 (detailed):(A4, B2, B4, B5, B7)\n3. Props2-2: 同上  but separate - but better to calc each: \n3.@@ PROPOSAL 2 Enhances to Become Two Paths:\n  -  **Path A**: getAddressin异质性 WITHOUT recurrent的 stats得到  {A4, B2, B4}\n  -  **Path B**: Including统计framework得到  {A4, B2， B4， B5}\n Thus needing_average counting but 和 非为例， i'll& proceed ，\n\n但 total the following：:\n\n- **Proposal1**: {A1, A2, A5, B1, B3, B5, B6} → SetA ideas count: A1+A2+A5 →3， SetB ideas:5（B1,B3,B5,B6）의? No: B ideas는 B1, B3, B5, B6 each once →4次.\n- **Proposal2-1**: {A4, B2, B4} → SetA:1（a4）, SetB:3（each）.\n- **Proposal2-2**: {A4, B4, B5} → 추가 يعد 1次 for B5（already counted in P1）.\n可能 it's better to recount with the acceptable grouping.\n\nHere hypothetically consider the following grouped proposals后 enhancement ， stating all used ideas..\n\n\n\n- **Proposal Candidate1-Enhanced**: \"Unifying genomic and phenotypic species boundaries通过 integrated thresholds):\n  - Ideas: A1 (95-96% ANI), B1 (16S threshold), B3 (log-linear correlation), B5 (statistical validation), A2 (cross-val methods), AND A5 (quali control).\n- **Prop2 (Lineage-centric threshold adjustments motivoel)**:\n  - Ideas: A4 (works已经). B2 (ANI vs 16S exceptions), B4 (rRNA variability), B5 (statistical methods).\n- **Proposal3 (Statistical assortment DEFAULT):\n  - **{A2, B5} → Simple 2 ideas.\n- **Proposal4 (Quality protocols):\n  - **{A5, B6} → 2 ideas.\n- **Proposal5 (Extremophile reclassification):\n  - **{A1, A4, B7} → count accordingly.\n- **Proposal6 (hollow ANI time **? perhaps merged into 1, but if seperate:\n   - {A3, B3, A5, B6} → additional A3.\n这样 total counts would be:\n\n### From SetA:\n- A1: used in Constituent1 和5 → 2次\n- A2:在 constituent1 →1次\n- A3:在 constituent6 →1次\n- A4:在 Prop2 and5→ 2次\n- A5:在 constituent1、4、6→ 3次\n- **Total SetA: 2 +1 +1+2+3 =9 ?** \n\n### From SetB,\n- B1: 1次\n- B2:1次\n- B3:在 constituent1和6 →2次\n- B4:1次\n- B5: 在 constituent1 and2, and3, and6 → 더, accounted as３?\n- **总 SetB:\n- B1:1, B2:1; B3:2; B4:1; B5:3； B6:2； B7:1.\n总 =1+1+2+1+3+2+1 =11.\n\nThis requires A precise grouping ，假设 my initial counts在 the absence of. overlapping, this would be the case.\n\n최終:\n\nAfter carefully considering the optimal combinations and reapments of the proposals， the following counts are derived：\n- **From SetA**: The ideas are used a total of **8** times across all proposals. This is because:\n  - A1 é employed in 2 proposals, \n  - A2 in 1, \n  - A3 in 1,\n  - A4 in 2,\n  - A5 in 2.\n  Total: 2+1+1+2+2=8.\n\n- **From SetB**: The total of **9**次使用:\n  - B1:1,\n  - B2:1,\n  - B3:1,\n  - B4:1,\n  - B5:3（appear在 3 proposals）,\n  - B6:2,\n  - B7:1.\n adding together:1+1+1+1+3+2+1=9.\n\nTherefore:\\n\n# From set A = 8\\n# From set B = 9"
            }
            ```
        "#};

        let result = parse_agent_output(test_final_answer);

        if let AgentEvent::Finish(final_answer) = result {
            println!("{}", final_answer);
        } else {
            panic!("Expected AgentEvent::Finish, got {:?}", result);
        }
    }

    #[test]
    fn test_more_output() {
        let test_final_answer: &str = indoc! {r#"
            {
                "final_answer": "Understanding the circadian regulation of sphingosine-1-phosphate (S1P) signaling is crucial for advancing our comprehension of how lipid mediators influence cardiovascular physiology across different times of the day. The research findings suggest that fluctuations in S1P levels, guided by intrinsic circadian rhythms, play a significant role in regulating cardiovascular parameters such as vascular tone and heart rate variability. With the emerging insights into the time-dependent nature of S1P signaling, future studies in cardiovascular physiology can build upon this foundation to investigate how disruptions in these rhythms might contribute to cardiovascular diseases.\n\nThe amalgamation of circadian biology with lipid signalling not only enhances our understanding of the mechanisms underlying cardiovascular homeostasis but also opens new avenues for therapeutic interventions. By elucidating the fundamental principles that govern lipid mediator activity in relation to time, researchers may develop innovative strategies to address cardiovascular health issues that fluctuate with the circadian cycle. This knowledge serves to not only improve our grasp of cardiovascular responses to physiological changes but also enrich the field of chronobiology, which seeks to understand how biological processes are influenced by time. Thus, there lies a tremendous potential for translating these findings into clinical practice, optimising treatment protocols according to the circadian patterns of S1P signaling in individuals, and ultimately fostering advancements in the management of cardiovascular diseases. As research continues, the integration of these insights may significantly contribute to the evolution of both cardiovascular physiology and lipid signalling research, paving the way for comprehensive understanding and novel therapeutic approaches.",
            }
        "#};

        let result = parse_agent_output(test_final_answer);

        if let AgentEvent::Finish(final_answer) = result {
            println!("{}", final_answer);
        } else {
            panic!("Expected AgentEvent::Finish, got {:?}", result);
        }
    }
}
