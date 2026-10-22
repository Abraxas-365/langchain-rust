//! Prompts for the document-combining chains, ported from langchaingo / LangChain.

pub(crate) const STUFF_QA_TEMPLATE: &str = r#"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{{context}}

Question:{{question}}
Helpful Answer:
"#;

pub(crate) const REFINE_QA_TEMPLATE: &str = r#"The original question is as follows: {{question}}
We have provided an existing answer: {{existing_answer}}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
------------
{{context}}
------------
Given the new context, refine the original answer to better answer the question.
If the context isn't useful, return the original answer."#;

pub(crate) const MAP_REDUCE_MAP_QA_TEMPLATE: &str = r#"Use the following portion of a long document to see if any of the text is relevant to answer the question.
Return any relevant text verbatim.
{{context}}
Question: {{question}}
Relevant text, if any:"#;

pub(crate) const MAP_REDUCE_COMBINE_QA_TEMPLATE: &str = r#"Given the following extracted parts of a long document and a question, create a final answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {{question}}
=========
{{context}}
=========
FINAL ANSWER:"#;

pub(crate) const SUMMARIZE_TEMPLATE: &str = r#"Write a concise summary of the following:


"{{context}}"


CONCISE SUMMARY:"#;

pub(crate) const REFINE_SUMMARIZE_TEMPLATE: &str = r#"Your job is to produce a final concise summary
We have provided an existing summary up to a certain point: "{{existing_answer}}"
We have the opportunity to refine the existing summary
(only if needed) with some more context below.
------------
"{{context}}"
------------

Given the new context, refine the original summary
If the context isn't useful, return the original summary.

REFINED SUMMARY:"#;

pub(crate) const MAP_RERANK_QA_TEMPLATE: &str = r#"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

Example #1

Context:
---------
Apples are red
---------
Question: what color are apples?
Helpful Answer: red
Score: 100

Example #2

Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
Question: what type was the car?
Helpful Answer: a sports car or an suv
Score: 60

Example #3

Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: This document does not answer the question
Score: 0

Begin!

Context:
---------
{{context}}
---------
Question: {{question}}
Helpful Answer:"#;
