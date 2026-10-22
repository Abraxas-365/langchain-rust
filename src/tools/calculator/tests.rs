use serde_json::{json, Value};

use super::{evaluate_expression, format_number, Calculator, CalculatorError};
use crate::tools::Tool;

fn eval(expression: &str) -> f64 {
    evaluate_expression(expression).unwrap_or_else(|e| panic!("{expression:?} failed: {e}"))
}

fn assert_close(expression: &str, expected: f64) {
    let got = eval(expression);
    assert!(
        (got - expected).abs() < 1e-9,
        "{expression:?} = {got}, expected {expected}"
    );
}

#[test]
fn precedence_and_associativity() {
    assert_eq!(eval("2 + 3 * 4"), 14.0);
    assert_eq!(eval("(2 + 3) * 4"), 20.0);
    assert_eq!(eval("10 - 4 - 3"), 3.0);
    assert_eq!(eval("100 / 10 / 5"), 2.0);
    assert_eq!(eval("2 ^ 3 ^ 2"), 512.0);
    assert_eq!(eval("2 ** 3 ** 2"), 512.0);
    assert_eq!(eval("(2 ^ 3) ^ 2"), 64.0);
    assert_eq!(eval("2 * 3 ^ 2"), 18.0);
    assert_eq!(eval("1 + 2 * 3 - 4 / 2"), 5.0);
    assert_eq!(eval("((((7))))"), 7.0);
}

#[test]
fn unary_signs() {
    assert_eq!(eval("-3"), -3.0);
    assert_eq!(eval("+3"), 3.0);
    assert_eq!(eval("--3"), 3.0);
    assert_eq!(eval("2 - -3"), 5.0);
    assert_eq!(eval("-2 ^ 2"), -4.0);
    assert_eq!(eval("2 ^ -1"), 0.5);
    assert_eq!(eval("-(1 + 2) * 3"), -9.0);
}

#[test]
fn division_modulo_and_floor_division() {
    assert_eq!(eval("7 / 2"), 3.5);
    assert_eq!(eval("7 // 2"), 3.0);
    assert_eq!(eval("-7 // 2"), -4.0);
    assert_eq!(eval("7 % 3"), 1.0);
    // Python semantics: sign of the divisor.
    assert_eq!(eval("-7 % 3"), 2.0);
    assert_eq!(eval("7 % -3"), -2.0);
    assert_eq!(eval("5.5 % 2"), 1.5);
    for zero in ["1 / 0", "1 // 0", "1 % 0", "1 / (2 - 2)"] {
        assert_eq!(
            evaluate_expression(zero),
            Err(CalculatorError::DivisionByZero),
            "{zero}"
        );
    }
    assert_eq!(eval("0 * 5"), 0.0);
}

#[test]
fn numbers_and_symbols() {
    assert_eq!(eval("1.5e3"), 1500.0);
    assert_eq!(eval("2E-2"), 0.02);
    assert_eq!(eval("1e+2"), 100.0);
    assert_eq!(eval(".5 + 1."), 1.5);
    assert_eq!(eval("6 × 7"), 42.0);
    assert_eq!(eval("84 ÷ 2"), 42.0);
    assert_eq!(eval("50 − 8"), 42.0);
    assert_eq!(eval("  \t1\n+\n1 "), 2.0);
}

#[test]
fn constants_and_functions() {
    assert_close("pi", std::f64::consts::PI);
    assert_close("e", std::f64::consts::E);
    assert_close("tau / 2", std::f64::consts::PI);
    assert_close("PI", std::f64::consts::PI);
    assert_eq!(eval("sqrt(16)"), 4.0);
    assert_eq!(eval("math.sqrt(16)"), 4.0);
    assert_eq!(eval("cbrt(27)"), 3.0);
    assert_eq!(eval("abs(-2.5)"), 2.5);
    assert_close("exp(1)", std::f64::consts::E);
    assert_close("ln(e)", 1.0);
    assert_close("log(e ^ 2)", 2.0);
    assert_close("log(8, 2)", 3.0);
    assert_close("log10(1000)", 3.0);
    assert_close("log2(1024)", 10.0);
    assert_close("sin(pi / 2)", 1.0);
    assert_close("cos(0)", 1.0);
    assert_close("tan(pi / 4)", 1.0);
    assert_close("asin(1)", std::f64::consts::FRAC_PI_2);
    assert_close("acos(1)", 0.0);
    assert_close("atan(1)", std::f64::consts::FRAC_PI_4);
    assert_close("atan2(1, 1)", std::f64::consts::FRAC_PI_4);
    assert_close("sinh(0) + cosh(0) + tanh(0)", 1.0);
    assert_eq!(eval("floor(-2.5)"), -3.0);
    assert_eq!(eval("ceil(2.1)"), 3.0);
    assert_eq!(eval("round(2.5)"), 3.0);
    assert_eq!(eval("round(-2.5)"), -3.0);
    assert_eq!(eval("round(1.23456, 3)"), 1.235);
    assert_eq!(eval("trunc(-2.7)"), -2.0);
    assert_eq!(eval("pow(2, 10)"), 1024.0);
    assert_eq!(eval("hypot(3, 4)"), 5.0);
    assert_eq!(eval("min(3, 1, 2)"), 1.0);
    assert_eq!(eval("max(3, 1, 2)"), 3.0);
    assert_eq!(eval("max(-1)"), -1.0);
    assert_eq!(eval("factorial(5)"), 120.0);
    assert_eq!(eval("factorial(0)"), 1.0);
    assert_eq!(eval("sqrt(max(9, 4)) * 2 + min(1, abs(-5))"), 7.0);
}

#[test]
fn errors_say_what_is_wrong() {
    let syntax = |expression: &str| match evaluate_expression(expression) {
        Err(CalculatorError::Syntax {
            position, found, ..
        }) => (position, found),
        other => panic!("{expression:?}: expected a syntax error, got {other:?}"),
    };
    assert_eq!(syntax("2 +"), (3, "end of expression".to_string()));
    assert_eq!(syntax("2 + * 3"), (4, "`*`".to_string()));
    assert_eq!(syntax("(1 + 2"), (6, "end of expression".to_string()));
    assert_eq!(syntax("1 + 2)"), (5, "`)`".to_string()));
    assert_eq!(syntax("2 3"), (2, "number 3".to_string()));
    assert_eq!(syntax("2 $ 3"), (2, "`$`".to_string()));
    assert_eq!(syntax("1.2.3"), (0, "`1.2.3`".to_string()));
    assert_eq!(syntax(""), (0, "end of expression".to_string()));
    assert_eq!(syntax("max(1,)"), (6, "`)`".to_string()));

    assert_eq!(
        evaluate_expression("foo(2)"),
        Err(CalculatorError::UnknownName("foo".into()))
    );
    assert_eq!(
        evaluate_expression("x + 1"),
        Err(CalculatorError::UnknownName("x".into()))
    );
    assert_eq!(
        evaluate_expression("sqrt(1, 2)"),
        Err(CalculatorError::Arity {
            function: "sqrt".into(),
            expected: "1",
            got: 2
        })
    );
    assert_eq!(
        evaluate_expression("pow(2)"),
        Err(CalculatorError::Arity {
            function: "pow".into(),
            expected: "2",
            got: 1
        })
    );
    assert!(matches!(
        evaluate_expression("max()"),
        Err(CalculatorError::Arity { got: 0, .. })
    ));
    assert!(matches!(
        evaluate_expression("log(1, 2, 3)"),
        Err(CalculatorError::Arity { got: 3, .. })
    ));
    assert!(matches!(
        evaluate_expression("round(1, 2, 3)"),
        Err(CalculatorError::Arity { got: 3, .. })
    ));
    for not_finite in [
        "sqrt(-1)",
        "10 ^ 400",
        "ln(0)",
        "factorial(-1)",
        "factorial(2.5)",
        "factorial(171)",
    ] {
        assert_eq!(
            evaluate_expression(not_finite),
            Err(CalculatorError::NotFinite),
            "{not_finite}"
        );
    }
    assert_eq!(
        eval("factorial(170)"),
        (1..=170u32).map(f64::from).product::<f64>()
    );

    let msg = evaluate_expression("2 + * 3").unwrap_err().to_string();
    assert_eq!(msg, "unexpected `*` at position 4 in `2 + * 3`");
    let msg = evaluate_expression("foo(1)").unwrap_err().to_string();
    assert!(
        msg.contains("unknown name `foo`") && msg.contains("sqrt"),
        "{msg}"
    );
}

#[test]
fn deep_nesting_is_rejected_not_a_stack_overflow() {
    let deep = format!("{}1{}", "(".repeat(10_000), ")".repeat(10_000));
    assert_eq!(evaluate_expression(&deep), Err(CalculatorError::TooDeep));
    let signs = format!("{}1", "-".repeat(10_000));
    assert_eq!(evaluate_expression(&signs), Err(CalculatorError::TooDeep));
    // Reasonable nesting is fine.
    let ok = format!("{}1{}", "(".repeat(95), ")".repeat(95));
    assert_eq!(eval(&ok), 1.0);
    let too_deep = format!("{}1{}", "(".repeat(105), ")".repeat(105));
    assert_eq!(
        evaluate_expression(&too_deep),
        Err(CalculatorError::TooDeep)
    );
    let ok = format!("{}1", "-".repeat(190));
    assert_eq!(eval(&ok), 1.0);
    // Depth is per nesting level, not per operation.
    let long = ["1"; 1000].join(" + ");
    assert_eq!(eval(&long), 1000.0);
    let groups = ["(1)"; 500].join(" + ");
    assert_eq!(eval(&groups), 500.0);
    let args = format!("max({})", ["-(1)"; 500].join(", "));
    assert_eq!(eval(&args), -1.0);
}

#[test]
fn format_number_hides_float_noise() {
    assert_eq!(format_number(14.0), "14");
    assert_eq!(format_number(-3.0), "-3");
    assert_eq!(format_number(-0.0), "0");
    assert_eq!(format_number(0.1 + 0.2), "0.3");
    assert_eq!(format_number(1.0 / 3.0), "0.333333333333");
    assert_eq!(format_number(2.5), "2.5");
    assert_eq!(format_number(1e20), "100000000000000000000");
    // Exact below 1e21, rounded to 12 digits above.
    assert_eq!(format_number(2f64.powi(60)), "1152921504606846976");
    assert_eq!(format_number(1e25), "10000000000000000000000000");
    assert_eq!(format_number(123456789012345.0), "123456789012345");
    assert_eq!(format_number(1.5e-7), "0.00000015");
}

#[tokio::test]
async fn tool_interface() {
    let calc = Calculator::new();
    assert_eq!(calc.name(), "calculator");
    assert!(calc.description().contains("math expression"));
    assert_eq!(
        calc.parameters()["required"],
        json!(["input"]),
        "{}",
        calc.parameters()
    );
    assert_eq!(calc.parameters()["properties"]["input"]["type"], "string");

    // OpenAI-style arguments, raw text, and code-fenced input.
    assert_eq!(calc.call(r#"{"input": "2 + 2"}"#).await.unwrap(), "4");
    assert_eq!(calc.call("(1234.5 * 3) / 3").await.unwrap(), "1234.5");
    assert_eq!(calc.call(" `0.1 + 0.2` ").await.unwrap(), "0.3");
    // A bare number from a JSON-happy model.
    assert_eq!(calc.call("42").await.unwrap(), "42");
    assert_eq!(calc.run(Value::from(7)).await.unwrap(), "7");

    let err = calc.call(r#"{"input": "2 +"}"#).await.unwrap_err();
    assert!(err.to_string().contains("end of expression"), "{err}");
}

#[tokio::test]
async fn agent_uses_the_calculator() {
    use crate::{
        agent::{AgentExecutor, OpenAiToolAgentBuilder},
        chain::Chain,
        prompt_args,
        schemas::{FunctionCallResponse, FunctionDetail},
        test_utils::FakeLLM,
    };
    use std::sync::Arc;

    let call = FunctionCallResponse {
        id: "call_1".into(),
        type_field: "function".into(),
        function: FunctionDetail {
            name: "calculator".into(),
            arguments: r#"{"input": "sqrt(2) * 10 // 1"}"#.into(),
        },
    };
    let llm = FakeLLM::new([
        serde_json::to_string(&vec![call]).unwrap(),
        "It's 14.".to_string(),
    ]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(Calculator::new())])
        .build(llm.clone())
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "what's floor(sqrt(2) * 10)?"})
        .await
        .unwrap();
    assert_eq!(out, "It's 14.");

    let calls = llm.calls();
    let observation = calls[1]
        .iter()
        .find(|m| m.message_type == crate::schemas::MessageType::ToolMessage)
        .expect("tool result sent back to the model");
    assert_eq!(observation.content, "14");
}
