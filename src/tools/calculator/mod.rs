//! A calculator tool: evaluates arithmetic expressions without calling out to anything.

use std::error::Error;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::Tool;

/// Why an expression couldn't be evaluated.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CalculatorError {
    #[error("unexpected {found} at position {position} in `{expression}`")]
    Syntax {
        expression: String,
        position: usize,
        found: String,
    },
    #[error("unknown name `{0}`; available functions: {FUNCTIONS}; constants: pi, e, tau")]
    UnknownName(String),
    #[error("{function}() takes {expected} argument(s), got {got}")]
    Arity {
        function: String,
        expected: &'static str,
        got: usize,
    },
    #[error("division by zero")]
    DivisionByZero,
    #[error("the result is not a finite number")]
    NotFinite,
    #[error("expression is nested too deeply")]
    TooDeep,
}

const FUNCTIONS: &str = "sqrt, cbrt, abs, exp, ln, log (natural, or log(x, base)), log10, log2, \
    sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, floor, ceil, round, trunc, \
    pow, min, max, hypot, factorial";

// One paren level costs two (expression + unary): about 100 nested parentheses.
const MAX_DEPTH: usize = 200;

/// Evaluates an arithmetic expression.
///
/// Supports `+ - * / % //` (floor division), `^` and `**` (power, right associative),
/// unary signs, parentheses, scientific notation (`1.5e3`), `×`/`÷`, the constants `pi`,
/// `e` and `tau`, and the functions listed in [`CalculatorError::UnknownName`]. A `math.`
/// prefix (`math.sqrt(2)`) is accepted, since models often write Python.
///
/// ```rust
/// use langchain_rust::tools::evaluate_expression;
/// assert_eq!(evaluate_expression("2 + 3 * 4").unwrap(), 14.0);
/// assert_eq!(evaluate_expression("2 ^ 3 ^ 2").unwrap(), 512.0);
/// assert_eq!(evaluate_expression("sqrt(16) + max(1, 5, 3)").unwrap(), 9.0);
/// ```
pub fn evaluate_expression(expression: &str) -> Result<f64, CalculatorError> {
    let mut parser = Parser {
        expression,
        tokens: tokenize(expression)?,
        position: 0,
        depth: 0,
    };
    let value = parser.expression()?;
    if let Some((position, token)) = parser.tokens.get(parser.position) {
        return Err(parser.unexpected(*position, token));
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CalculatorError::NotFinite)
    }
}

/// Formats a result the way a person would write it: integers without a decimal point (up
/// to 1e21, like JavaScript), otherwise at most 12 significant digits so float noise
/// (`0.1 + 0.2`) doesn't show.
pub fn format_number(value: f64) -> String {
    if value == value.trunc() && value.abs() < 1e21 {
        // `+ 0.0` turns -0 into 0.
        return format!("{:.0}", value + 0.0);
    }
    let rounded: f64 = format!("{value:.11e}").parse().unwrap_or(value);
    rounded.to_string()
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Op(&'static str),
    Open,
    Close,
    Comma,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Number(n) => write!(f, "number {n}"),
            Token::Name(n) => write!(f, "`{n}`"),
            Token::Op(op) => write!(f, "`{op}`"),
            Token::Open => write!(f, "`(`"),
            Token::Close => write!(f, "`)`"),
            Token::Comma => write!(f, "`,`"),
        }
    }
}

fn tokenize(expression: &str) -> Result<Vec<(usize, Token)>, CalculatorError> {
    let syntax = |position: usize, found: String| CalculatorError::Syntax {
        expression: expression.to_string(),
        position,
        found,
    };
    let chars: Vec<(usize, char)> = expression.char_indices().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (position, c) = chars[i];
        let next = chars.get(i + 1).map(|(_, c)| *c);
        let (token, len) = match c {
            c if c.is_whitespace() => {
                i += 1;
                continue;
            }
            '0'..='9' | '.' => {
                let mut end = i;
                while end < chars.len() && (chars[end].1.is_ascii_digit() || chars[end].1 == '.') {
                    end += 1;
                }
                // Exponent: e/E, optional sign, digits.
                if end < chars.len() && matches!(chars[end].1, 'e' | 'E') {
                    let mut exp = end + 1;
                    if exp < chars.len() && matches!(chars[exp].1, '+' | '-') {
                        exp += 1;
                    }
                    if exp < chars.len() && chars[exp].1.is_ascii_digit() {
                        while exp < chars.len() && chars[exp].1.is_ascii_digit() {
                            exp += 1;
                        }
                        end = exp;
                    }
                }
                let text: String = chars[i..end].iter().map(|(_, c)| c).collect();
                let number = text
                    .parse()
                    .map_err(|_| syntax(position, format!("`{text}`")))?;
                (Token::Number(number), end - i)
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut end = i;
                while end < chars.len()
                    && (chars[end].1.is_alphanumeric() || matches!(chars[end].1, '_' | '.'))
                {
                    end += 1;
                }
                let text: String = chars[i..end].iter().map(|(_, c)| c).collect();
                let name = text.strip_prefix("math.").unwrap_or(&text).to_lowercase();
                (Token::Name(name), end - i)
            }
            '*' if next == Some('*') => (Token::Op("^"), 2),
            '/' if next == Some('/') => (Token::Op("//"), 2),
            '+' => (Token::Op("+"), 1),
            '-' | '−' => (Token::Op("-"), 1),
            '*' | '×' => (Token::Op("*"), 1),
            '/' | '÷' => (Token::Op("/"), 1),
            '%' => (Token::Op("%"), 1),
            '^' => (Token::Op("^"), 1),
            '(' => (Token::Open, 1),
            ')' => (Token::Close, 1),
            ',' => (Token::Comma, 1),
            other => return Err(syntax(position, format!("`{other}`"))),
        };
        tokens.push((position, token));
        i += len;
    }
    Ok(tokens)
}

struct Parser<'a> {
    expression: &'a str,
    tokens: Vec<(usize, Token)>,
    position: usize,
    depth: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position).map(|(_, t)| t)
    }

    fn eat_op(&mut self, ops: &[&'static str]) -> Option<&'static str> {
        match self.peek() {
            Some(Token::Op(op)) if ops.contains(op) => {
                let op = *op;
                self.position += 1;
                Some(op)
            }
            _ => None,
        }
    }

    fn unexpected(&self, position: usize, token: &Token) -> CalculatorError {
        CalculatorError::Syntax {
            expression: self.expression.to_string(),
            position,
            found: token.to_string(),
        }
    }

    fn unexpected_here(&self) -> CalculatorError {
        match self.tokens.get(self.position) {
            Some((position, token)) => self.unexpected(*position, token),
            None => CalculatorError::Syntax {
                expression: self.expression.to_string(),
                position: self.expression.len(),
                found: "end of expression".to_string(),
            },
        }
    }

    fn expect(&mut self, token: Token) -> Result<(), CalculatorError> {
        if self.peek() == Some(&token) {
            self.position += 1;
            Ok(())
        } else {
            Err(self.unexpected_here())
        }
    }

    /// expression := term (("+" | "-") term)*
    fn expression(&mut self) -> Result<f64, CalculatorError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(CalculatorError::TooDeep);
        }
        let mut value = self.term()?;
        while let Some(op) = self.eat_op(&["+", "-"]) {
            let rhs = self.term()?;
            value = if op == "+" { value + rhs } else { value - rhs };
        }
        self.depth -= 1;
        Ok(value)
    }

    /// term := unary (("*" | "/" | "//" | "%") unary)*
    fn term(&mut self) -> Result<f64, CalculatorError> {
        let mut value = self.unary()?;
        while let Some(op) = self.eat_op(&["*", "/", "//", "%"]) {
            let rhs = self.unary()?;
            if op != "*" && rhs == 0.0 {
                return Err(CalculatorError::DivisionByZero);
            }
            value = match op {
                "*" => value * rhs,
                "/" => value / rhs,
                "//" => (value / rhs).floor(),
                // Python semantics: the result has the sign of the divisor.
                _ => value - rhs * (value / rhs).floor(),
            };
        }
        Ok(value)
    }

    /// unary := ("+" | "-") unary | power. `-2^2` is -(2^2), as in maths and Python.
    fn unary(&mut self) -> Result<f64, CalculatorError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(CalculatorError::TooDeep);
        }
        let value = match self.eat_op(&["+", "-"]) {
            Some("-") => -self.unary()?,
            Some(_) => self.unary()?,
            None => self.power()?,
        };
        self.depth -= 1;
        Ok(value)
    }

    /// power := primary ("^" unary)?   (right associative: 2^3^2 = 2^9)
    fn power(&mut self) -> Result<f64, CalculatorError> {
        let base = self.primary()?;
        if self.eat_op(&["^"]).is_some() {
            let exponent = self.unary()?;
            return Ok(base.powf(exponent));
        }
        Ok(base)
    }

    /// primary := number | name | name "(" args ")" | "(" expression ")"
    fn primary(&mut self) -> Result<f64, CalculatorError> {
        let Some((_, token)) = self.tokens.get(self.position).cloned() else {
            return Err(self.unexpected_here());
        };
        match token {
            Token::Number(n) => {
                self.position += 1;
                Ok(n)
            }
            Token::Open => {
                self.position += 1;
                let value = self.expression()?;
                self.expect(Token::Close)?;
                Ok(value)
            }
            Token::Name(name) => {
                self.position += 1;
                if self.peek() == Some(&Token::Open) {
                    self.position += 1;
                    let mut args = Vec::new();
                    if self.peek() != Some(&Token::Close) {
                        args.push(self.expression()?);
                        while self.peek() == Some(&Token::Comma) {
                            self.position += 1;
                            args.push(self.expression()?);
                        }
                    }
                    self.expect(Token::Close)?;
                    call(&name, &args)
                } else {
                    constant(&name)
                }
            }
            _ => Err(self.unexpected_here()),
        }
    }
}

fn constant(name: &str) -> Result<f64, CalculatorError> {
    match name {
        "pi" => Ok(std::f64::consts::PI),
        "e" => Ok(std::f64::consts::E),
        "tau" => Ok(std::f64::consts::TAU),
        _ => Err(CalculatorError::UnknownName(name.to_string())),
    }
}

fn call(name: &str, args: &[f64]) -> Result<f64, CalculatorError> {
    let arity = |expected: &'static str| CalculatorError::Arity {
        function: name.to_string(),
        expected,
        got: args.len(),
    };
    let one = |f: fn(f64) -> f64| match args {
        [x] => Ok(f(*x)),
        _ => Err(arity("1")),
    };
    let two = |f: fn(f64, f64) -> f64| match args {
        [x, y] => Ok(f(*x, *y)),
        _ => Err(arity("2")),
    };
    match name {
        "sqrt" => one(f64::sqrt),
        "cbrt" => one(f64::cbrt),
        "abs" => one(f64::abs),
        "exp" => one(f64::exp),
        "ln" => one(f64::ln),
        "log" => match args {
            [x] => Ok(x.ln()),
            [x, base] => Ok(x.log(*base)),
            _ => Err(arity("1 or 2")),
        },
        "log10" => one(f64::log10),
        "log2" => one(f64::log2),
        "sin" => one(f64::sin),
        "cos" => one(f64::cos),
        "tan" => one(f64::tan),
        "asin" => one(f64::asin),
        "acos" => one(f64::acos),
        "atan" => one(f64::atan),
        "atan2" => two(f64::atan2),
        "sinh" => one(f64::sinh),
        "cosh" => one(f64::cosh),
        "tanh" => one(f64::tanh),
        "floor" => one(f64::floor),
        "ceil" => one(f64::ceil),
        // Half away from zero, like most calculators.
        "round" => match args {
            [x] => Ok(x.round()),
            [x, digits] => {
                let scale = 10f64.powi(*digits as i32);
                Ok((x * scale).round() / scale)
            }
            _ => Err(arity("1 or 2")),
        },
        "trunc" => one(f64::trunc),
        "pow" => two(f64::powf),
        "hypot" => two(f64::hypot),
        "min" | "max" if args.is_empty() => Err(arity("at least 1")),
        "min" => Ok(args.iter().copied().fold(f64::INFINITY, f64::min)),
        "max" => Ok(args.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        "factorial" => match args {
            [n] if *n >= 0.0 && n.fract() == 0.0 && *n <= 170.0 => {
                Ok((1..=*n as u32).map(f64::from).product())
            }
            [_] => Err(CalculatorError::NotFinite),
            _ => Err(arity("1")),
        },
        _ => Err(CalculatorError::UnknownName(name.to_string())),
    }
}

/// A tool that evaluates math expressions, so agents don't do arithmetic in their heads.
///
/// The input is an expression such as `(1234.5 * 3) / sqrt(2)`; see
/// [`evaluate_expression`] for the syntax. Invalid expressions come back as tool errors
/// that say what's wrong, so the model can fix the expression and retry.
#[derive(Debug, Clone, Default)]
pub struct Calculator;

impl Calculator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> String {
        "calculator".to_string()
    }

    fn description(&self) -> String {
        "Evaluates a math expression and returns the result. Use it for any arithmetic. \
         Supports + - * / % ^ (power), parentheses, the constants pi and e, and functions \
         like sqrt, abs, ln, log10, sin, cos, round, min, max and factorial. \
         Example input: (1234.5 * 3) / sqrt(2)"
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. (1234.5 * 3) / sqrt(2)"
                }
            },
            "required": ["input"]
        })
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        // The default `parse_input` hands over a string; anything else is tried as text.
        let expression = input
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| input.to_string());
        let expression = expression.trim().trim_matches('`').trim();
        Ok(format_number(evaluate_expression(expression)?))
    }
}

#[cfg(test)]
mod tests;
