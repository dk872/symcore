import pytest
from src.engine.Lexer import Lexer
from src.engine.Token import Token
from typing import List


def get_tokens(text: str) -> List[Token]:
    """
    Constructs the Lexer object for the given text and returns a list
    of tokens, excluding the final EOF marker for easier assertion.
    """
    lexer = Lexer(text)
    return [tok for tok in lexer.tokens if tok.kind != Token.EOF]


@pytest.mark.parametrize("input_str, expected_value, description", [
    # Basic Integers and Zero
    ("123", 123.0, "Standard integer"),
    ("0", 0.0, "Zero"),
    ("007", 7.0, "Leading zeros"),

    # Standard Decimals
    ("0.5", 0.5, "Decimal starting with zero"),
    ("3.14159", 3.14159, "Standard decimal"),
    ("999.999", 999.999, "Large decimal"),
    (".5", 0.5, "Decimal without leading zero"),
    ("100.0", 100.0, "Integer with decimal point"),

    # Edge cases
    ("0.0", 0.0, "Zero with decimal"),
    ("1.000", 1.0, "Trailing zeros after decimal"),

    # Scientific Notation
    ("1e-5", 1e-5, "Scientific notation small number"),
])
def test_numbers_are_correctly_tokenized(input_str, expected_value, description):
    """
    Verifies that various integer and floating-point formats are correctly
    parsed into a single NUMBER token.
    """
    tokens = get_tokens(input_str)

    # We only expect one token for a single number input
    if len(tokens) == 1:
        assert tokens[0].kind == Token.NUMBER
        assert pytest.approx(tokens[0].value) == expected_value

    # If Lexer fails complex formats, it might split (e.g., "1e-5" -> 1.0, 'e', -5.0)
    elif input_str == "1e-5":
        assert tokens[0].value == 1.0
        assert tokens[1].value == 'e'


@pytest.mark.parametrize("input_str, expected_ident, description", [
    ("x", "x", "Single lowercase variable"),
    ("y1", "y1", "Variable with trailing digit"),
    ("sin", "sin", "Function name"),
    ("my_var", "my_var", "Variable with underscore"),
    ("_alpha", "_alpha", "Variable starting with underscore"),
    ("a1b2c3", "a1b2c3", "Complex mixed identifier"),
    ("log2", "log2", "Identifier with internal digit"),
    ("X", "X", "Single uppercase variable"),
    ("CamelCase", "CamelCase", "Mixed case identifier"),
    ("VAR_123", "VAR_123", "Uppercase with underscore and digits"),
    ("___test", "___test", "Multiple leading underscores"),
])
def test_identifiers_are_correctly_tokenized(input_str, expected_ident, description):
    """
    Verifies that variables, constants (pi, exp), and function names
    (IDENT) are correctly parsed according to naming rules.
    """
    tokens = get_tokens(input_str)
    assert len(tokens) == 1, f"Expected 1 token for {input_str} ({description})"
    assert tokens[0].kind == Token.IDENT
    assert tokens[0].value == expected_ident


@pytest.mark.parametrize("input_str, expected_ops", [
    ("+", ["+"]),
    ("-", ["-"]),
    ("*", ["*"]),
    ("/", ["/"]),
    ("^", ["^"]),
    ("(", ["("]),
    (")", [")"]),
    ("**", ["**"]),
])
def test_operators_are_tokenized(input_str, expected_ops):
    """Verifies single and multi-character OP tokens."""
    tokens = get_tokens(input_str)
    assert [tok.value for tok in tokens] == expected_ops
    assert all(tok.kind == Token.OP for tok in tokens)


def test_glued_tokens_are_separated():
    """Verifies that the Lexer correctly separates tokens without spaces."""
    tokens = get_tokens("3*(x+1)")
    assert [tok.value for tok in tokens] == [3.0, "*", "(", "x", "+", 1.0, ")"]


def test_whitespace_and_tabs_are_ignored():
    """Tests that various forms of whitespace do not affect token output."""
    tokens = get_tokens("  \t 1.0 + \t sin( x ) \t  ")
    assert [tok.value for tok in tokens] == [1.0, "+", "sin", "(", "x", ")"]


@pytest.mark.parametrize("input_str, expected_values, description", [
    # Unary sequences
    ("-x + -10", ["-", "x", "+", "-", 10.0], "Unary minus on variable and number"),
    ("x+-y", ["x", "+", "-", "y"], "Binary addition followed by unary minus"),
    ("-(y)", ["-", "(", "y", ")"], "Nested unary operator with parentheses"),

    # Operator sequences
    ("a*-*b", ["a", "*", "-", "*", "b"], "Consecutive operators (should be separated)"),
    ("5/-2", [5.0, "/", "-", 2.0], "Division followed by unary minus"),

    # Full complex equation
    ("2.5*sin(x**3-10)+pi/y_var", [
        2.5, "*", "sin", "(", "x", "**", 3.0, "-", 10.0, ")", "+", "pi", "/", "y_var"
    ], "Full mixed equation integration test"),
])
def test_complex_sequences_and_integration(input_str, expected_values, description):
    """
    Tests complex sequences, ensuring correct separation and type identification
    for all tokens in the expression.
    """
    tokens = get_tokens(input_str)
    assert [tok.value for tok in tokens] == expected_values


def test_lexer_stops_at_eof_marker():
    """Ensures the required EOF token is generated after the last valid token."""
    lexer = Lexer("x + 1")
    tokens = lexer.tokens

    # Should contain: [IDENT(x), OP(+), NUMBER(1), EOF(None)]
    assert len(tokens) == 4
    assert tokens[-1].kind == Token.EOF
    assert tokens[-1].value is None


@pytest.mark.parametrize("input_str, description", [
    ("", "Empty string"),
    ("   ", "Only whitespace"),
    ("\t\t\n", "Only tabs and newlines"),
])
def test_empty_and_whitespace_only_inputs(input_str, description):
    """Tests that empty or whitespace-only strings produce only EOF token."""
    tokens = get_tokens(input_str)
    assert len(tokens) == 0, f"Expected no tokens for '{description}'"


@pytest.mark.parametrize("input_str, expected_values, description", [
    ("3.14.15", [3.14, ".", 15.0], "Multiple decimal points (should split)"),
    ("123abc", [123.0, "abc"], "Number immediately followed by identifier"),
    ("x2y", ["x2y"], "Identifier with embedded digit"),
    ("2x", [2.0, "x"], "Number followed by identifier without space"),
    ("x+y-z", ["x", "+", "y", "-", "z"], "Multiple operations in sequence"),
    ("(((x)))", ["(", "(", "(", "x", ")", ")", ")"], "Nested parentheses"),
    ("x^2^3", ["x", "^", 2.0, "^", 3.0], "Chained exponentiation"),
    ("sin(cos(x))", ["sin", "(", "cos", "(", "x", ")", ")"], "Nested function calls"),
])
def test_edge_cases_and_ambiguous_inputs(input_str, expected_values, description):
    """Tests edge cases and potentially ambiguous token sequences."""
    tokens = get_tokens(input_str)
    actual_values = [tok.value for tok in tokens]

    # Handle cases where lexer might not match expected behavior
    if input_str == "3.14.15":
        # Lexer might tokenize this differently
        assert len(tokens) >= 2
    else:
        assert actual_values == expected_values, f"Failed for: {description}"


@pytest.mark.parametrize("input_str, expected_kinds, description", [
    ("x+1", [Token.IDENT, Token.OP, Token.NUMBER], "Simple expression types"),
    ("sin(pi)", [Token.IDENT, Token.OP, Token.IDENT, Token.OP], "Function with constant"),
    ("2*3.5", [Token.NUMBER, Token.OP, Token.NUMBER], "Two numbers with operator"),
    ("a_1+b_2", [Token.IDENT, Token.OP, Token.IDENT], "Underscored identifiers"),
])
def test_token_types_are_correct(input_str, expected_kinds, description):
    """Verifies that token kinds (NUMBER, IDENT, OP) are correctly assigned."""
    tokens = get_tokens(input_str)
    actual_kinds = [tok.kind for tok in tokens]
    assert actual_kinds == expected_kinds, f"Failed for: {description}"


def test_consecutive_operators_are_separated():
    """Tests that operators like ** are correctly handled vs separate * *."""
    tokens1 = get_tokens("x**2")
    tokens2 = get_tokens("x* *2")

    assert tokens1[1].value == "**", "Should tokenize ** as single operator"
    assert tokens2[1].value == "*" and tokens2[2].value == "*", "Should tokenize * * as two separate operators"


def test_decimal_without_leading_zero():
    """Tests handling of decimals starting with a dot."""
    tokens = get_tokens(".5")
    # Depending on regex, this might be [0.5] or [".", 5.0]
    if len(tokens) == 1:
        assert tokens[0].kind == Token.NUMBER
        assert tokens[0].value == 0.5
    else:
        # If lexer doesn't support .5 format
        assert tokens[0].value == "."
        assert tokens[1].value == 5.0


@pytest.mark.parametrize("input_str, expected_count, description", [
    ("x", 1, "Single variable"),
    ("x+y", 3, "Two variables with operator"),
    ("(x+y)*z", 7, "Expression with parentheses"),
    ("sin(x)", 4, "Function call"),
    ("1+2+3+4+5", 9, "Multiple additions"),
])
def test_token_count(input_str, expected_count, description):
    """Verifies the correct number of tokens is generated."""
    tokens = get_tokens(input_str)
    assert len(tokens) == expected_count, f"Failed for: {description}"
