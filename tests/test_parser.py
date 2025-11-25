import pytest
import math
from src.engine.Parser import Parser


def parse_and_get_repr(expression_string: str) -> str:
    """
    Parses the input string and returns the unambiguous string representation (repr)
    of the root AST node. This allows for clear structure validation.
    """
    parser = Parser(expression_string)
    root_node = parser.parse()
    return repr(root_node)


def test_single_literal_integer():
    """Verifies parsing of a simple integer literal."""
    assert parse_and_get_repr("123") == "Literal(123)"


def test_single_decimal_literal():
    """Verifies parsing of a simple floating-point literal."""
    assert parse_and_get_repr("3.14") == "Literal(3.14)"


def test_single_variable():
    """Verifies parsing of a single variable."""
    assert parse_and_get_repr("x_var") == "Variable('x_var')"


def test_pi_constant():
    """Verifies parsing of the 'pi' constant, confirming it's a Literal."""
    assert parse_and_get_repr("pi") == f"Literal({math.pi})"


def test_nested_parentheses_as_primary():
    """Verifies that nested parentheses are treated as a single primary node."""
    expected = "BinaryOperator('*', BinaryOperator('+', Variable('a'), Variable('b')), Literal(2))"
    assert parse_and_get_repr("(a + b) * 2") == expected


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("+x", "UnaryOperator('+', Variable('x'))", "Unary plus"),
    ("-5", "UnaryOperator('-', Literal(5))", "Unary minus on number"),
    ("sin(y)", "UnaryOperator('sin', Variable('y'))", "Simple function call"),
    ("ln(10.5)", "UnaryOperator('ln', Literal(10.5))", "Logarithm on decimal"),
    ("sqrt(z_var)", "UnaryOperator('sqrt', Variable('z_var'))", "Square root function"),
    ("-(-y)", "UnaryOperator('-', UnaryOperator('-', Variable('y')))", "Nested unary operators"),
    ("sin(-x)", "UnaryOperator('sin', UnaryOperator('-', Variable('x')))", "Function with unary argument"),
    ("tan(a / b)", "UnaryOperator('tan', BinaryOperator('/', Variable('a'), Variable('b')))",
     "Function with complex argument"),
    ("cos(pi)", f"UnaryOperator('cos', Literal({math.pi}))", "Function on constant"),
    ("log(x)", "UnaryOperator('log', Variable('x'))", "Base-10 logarithm"),
    ("exp(2)", "UnaryOperator('exp', Literal(2))", "Exponential function"),
    ("cot(a)", "UnaryOperator('cot', Variable('a'))", "Cotangent function"),
])
def test_unary_operators_and_functions(input_str, expected_repr, description):
    """Verifies parsing of unary operators (+, -) and function calls."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_structure, description", [
    # Hierarchy: Add vs. Mul
    ("a + b * c",
     "BinaryOperator('+', Variable('a'), BinaryOperator('*', Variable('b'), Variable('c')))",
     "Mul binds before Add"),

    # Hierarchy: Mul vs. Div (Same level)
    ("a * b / c",
     "BinaryOperator('/', BinaryOperator('*', Variable('a'), Variable('b')), Variable('c'))",
     "Mul and Div (Left Associative)"),

    # Hierarchy: Pow vs. Mul
    ("2 * x ^ 3",
     "BinaryOperator('*', Literal(2), BinaryOperator('^', Variable('x'), Literal(3)))",
     "Pow binds before Mul"),

    # Full hierarchy check
    ("a - b / c ^ 2 + d",
     "BinaryOperator('+', BinaryOperator('-', Variable('a'), BinaryOperator('/', Variable('b'), BinaryOperator('^', "
     "Variable('c'), Literal(2)))), Variable('d'))",
     "Pow, Div, Sub, Add"),

    # Additional precedence tests
    ("x + y - z",
     "BinaryOperator('-', BinaryOperator('+', Variable('x'), Variable('y')), Variable('z'))",
     "Add and Sub (Left Associative)"),

    ("a / b * c",
     "BinaryOperator('*', BinaryOperator('/', Variable('a'), Variable('b')), Variable('c'))",
     "Div and Mul (Left Associative)"),

    ("x ^ y * z",
     "BinaryOperator('*', BinaryOperator('^', Variable('x'), Variable('y')), Variable('z'))",
     "Pow binds tighter than Mul"),

    ("a + b ^ c",
     "BinaryOperator('+', Variable('a'), BinaryOperator('^', Variable('b'), Variable('c')))",
     "Pow binds tighter than Add"),
])
def test_operator_precedence_hierarchy(input_str, expected_structure, description):
    """Verifies the correct precedence order: ^ > * / > + -."""
    assert parse_and_get_repr(input_str) == expected_structure, description


@pytest.mark.parametrize("input_str, expected_structure, description", [
    # Left Associativity (+, -)
    ("a - b + c - d",
     "BinaryOperator('-', BinaryOperator('+', BinaryOperator('-', Variable('a'), Variable('b')), Variable('c')), "
     "Variable('d'))",
     "L-R: Complex Add/Sub"),

    ("a / b * c / d",
     "BinaryOperator('/', BinaryOperator('*', BinaryOperator('/', Variable('a'), Variable('b')), Variable('c')), "
     "Variable('d'))",
     "L-R: Complex Mul/Div"),

    # Right Associativity (^)
    ("2 ^ x ^ y",
     "BinaryOperator('^', Literal(2), BinaryOperator('^', Variable('x'), Variable('y')))",
     "R-L: Nested Power"),

    ("a + b + c",
     "BinaryOperator('+', BinaryOperator('+', Variable('a'), Variable('b')), Variable('c'))",
     "L-R: Multiple additions"),

    ("a * b * c",
     "BinaryOperator('*', BinaryOperator('*', Variable('a'), Variable('b')), Variable('c'))",
     "L-R: Multiple multiplications"),

    ("a ^ b ^ c ^ d",
     "BinaryOperator('^', Variable('a'), BinaryOperator('^', Variable('b'), BinaryOperator('^', Variable('c'), "
     "Variable('d'))))",
     "R-L: Multiple powers"),
])
def test_associativity_rules(input_str, expected_structure, description):
    """Verifies that operators group according to their associativity (L-R for +/*, R-L for ^)."""
    assert parse_and_get_repr(input_str) == expected_structure, description


def test_parentheses_and_mixed_unary():
    """
    Checks the structurally correct AST: (-3) * (a + 4),
    as the parser applies unary minus first.
    """
    expected = "BinaryOperator('*', UnaryOperator('-', Literal(3)), BinaryOperator('+', Variable('a'), Literal(4)))"
    assert parse_and_get_repr("-3 * (a + 4)") == expected


def test_function_on_complex_expression():
    """
    Checks the structurally correct AST based on Left Associativity:
    sqrt(((a*b) + c) - 1).
    """
    expected = ("UnaryOperator('sqrt', BinaryOperator('-', BinaryOperator('+', BinaryOperator('*', Variable('a'), "
                "Variable('b')), Variable('c')), Literal(1)))")
    assert parse_and_get_repr("sqrt(a*b + c - 1)") == expected


def test_deep_function_nesting():
    """Verifies correct parsing of deeply nested expressions involving all operators."""
    expected = ("BinaryOperator('/', BinaryOperator('^', BinaryOperator('+', Variable('x'), Literal(1)), Literal(2)), "
                "BinaryOperator('-', Variable('y'), Literal(3)))")
    assert parse_and_get_repr("(x + 1)^2 / (y - 3)") == expected


def test_function_nesting():
    """Verifies a function call that includes parentheses and follows precedence."""
    expected = ("BinaryOperator('*', UnaryOperator('sin', BinaryOperator('+', Variable('x'), Literal(1))), "
                "Variable('y'))")
    assert parse_and_get_repr("sin(x + 1) * y") == expected


def test_mixed_unary_and_parentheses_simple():
    """Verifies that a function call can be treated as a single primary operand."""
    expected = "BinaryOperator('*', UnaryOperator('cos', Variable('x')), Literal(3))"
    assert parse_and_get_repr("cos(x) * 3") == expected


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("-(x + y)", "UnaryOperator('-', BinaryOperator('+', Variable('x'), Variable('y')))",
     "Unary minus on parenthesized expression"),

    ("+sin(x)", "UnaryOperator('+', UnaryOperator('sin', Variable('x')))",
     "Unary plus on function call"),

    ("-sin(x)", "UnaryOperator('-', UnaryOperator('sin', Variable('x')))",
     "Unary minus on function call"),

    ("-(a * b)", "UnaryOperator('-', BinaryOperator('*', Variable('a'), Variable('b')))",
     "Unary minus on multiplication"),

    ("+(x)", "UnaryOperator('+', Variable('x'))",
     "Unary plus on parenthesized variable"),
])
def test_unary_with_complex_operands(input_str, expected_repr, description):
    """Verifies unary operators applied to complex expressions."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("(a)", "Variable('a')", "Single variable in parentheses"),

    ("((x))", "Variable('x')", "Double nested parentheses"),

    ("(((5)))", "Literal(5)", "Triple nested parentheses"),

    ("((a + b))", "BinaryOperator('+', Variable('a'), Variable('b'))",
     "Nested parentheses around expression"),

    ("(a) + (b)", "BinaryOperator('+', Variable('a'), Variable('b'))",
     "Multiple parenthesized primaries"),

    ("(a + b) * (c + d)",
     "BinaryOperator('*', BinaryOperator('+', Variable('a'), Variable('b')), BinaryOperator('+', Variable('c'), "
     "Variable('d')))",
     "Two parenthesized expressions multiplied"),
])
def test_parentheses_variations(input_str, expected_repr, description):
    """Verifies various parentheses usage patterns."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("sin(cos(x))", "UnaryOperator('sin', UnaryOperator('cos', Variable('x')))",
     "Nested trigonometric functions"),

    ("ln(exp(y))", "UnaryOperator('ln', UnaryOperator('exp', Variable('y')))",
     "Nested exponential and logarithm"),

    ("sqrt(sqrt(z))", "UnaryOperator('sqrt', UnaryOperator('sqrt', Variable('z')))",
     "Nested square roots"),

    ("sin(x + cos(y))",
     "UnaryOperator('sin', BinaryOperator('+', Variable('x'), UnaryOperator('cos', Variable('y'))))",
     "Function with expression containing another function"),

    ("exp(ln(x) * 2)",
     "UnaryOperator('exp', BinaryOperator('*', UnaryOperator('ln', Variable('x')), Literal(2)))",
     "Complex nested function expression"),
])
def test_nested_functions(input_str, expected_repr, description):
    """Verifies parsing of nested function calls."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("x ^ 2 ^ 3", "BinaryOperator('^', Variable('x'), BinaryOperator('^', Literal(2), Literal(3)))",
     "Right-associative power chain"),

    ("2 ^ 3 ^ 4", "BinaryOperator('^', Literal(2), BinaryOperator('^', Literal(3), Literal(4)))",
     "Right-associative numeric powers"),

    ("a ^ (b + c)",
     "BinaryOperator('^', Variable('a'), BinaryOperator('+', Variable('b'), Variable('c')))",
     "Power with expression as exponent"),

    ("(a + b) ^ c",
     "BinaryOperator('^', BinaryOperator('+', Variable('a'), Variable('b')), Variable('c'))",
     "Power with expression as base"),

    ("x ^ -2", "BinaryOperator('^', Variable('x'), UnaryOperator('-', Literal(2)))",
     "Power with negative exponent"),

    ("2 ^ -3", "BinaryOperator('^', Literal(2), UnaryOperator('-', Literal(3)))",
     "Numeric power with negative exponent"),
])
def test_power_operator_special_cases(input_str, expected_repr, description):
    """Verifies special cases of power operator parsing."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("a - b - c", "BinaryOperator('-', BinaryOperator('-', Variable('a'), Variable('b')), Variable('c'))",
     "Chain of subtractions (left-associative)"),

    ("a / b / c", "BinaryOperator('/', BinaryOperator('/', Variable('a'), Variable('b')), Variable('c'))",
     "Chain of divisions (left-associative)"),

    ("a + b - c + d",
     "BinaryOperator('+', BinaryOperator('-', BinaryOperator('+', Variable('a'), Variable('b')), Variable('c')), "
     "Variable('d'))",
     "Mixed additions and subtractions"),

    ("a * b / c * d",
     "BinaryOperator('*', BinaryOperator('/', BinaryOperator('*', Variable('a'), Variable('b')), Variable('c')), "
     "Variable('d'))",
     "Mixed multiplications and divisions"),
])
def test_left_associative_chains(input_str, expected_repr, description):
    """Verifies left-associativity for chains of same-precedence operators."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("0", "Literal(0)", "Zero literal"),
    ("0.0", "Literal(0)", "Zero as float converted to int"),
    ("1.0", "Literal(1)", "One as float converted to int"),
    ("123.456", "Literal(123.456)", "Non-integer float"),
    ("0 + x", "BinaryOperator('+', Literal(0), Variable('x'))", "Zero in expression"),
    ("x * 1", "BinaryOperator('*', Variable('x'), Literal(1))", "Multiplication by one"),
])
def test_literal_edge_cases(input_str, expected_repr, description):
    """Verifies parsing of edge case literals."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("a", "Variable('a')", "Single letter variable"),
    ("xyz", "Variable('xyz')", "Multi-letter variable"),
    ("x1", "Variable('x1')", "Variable with number"),
    ("_var", "Variable('_var')", "Variable starting with underscore"),
    ("var_name_123", "Variable('var_name_123')", "Complex variable name"),
])
def test_variable_naming_patterns(input_str, expected_repr, description):
    """Verifies various valid variable naming patterns."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("x+y", "BinaryOperator('+', Variable('x'), Variable('y'))", "No spaces"),
    ("x + y", "BinaryOperator('+', Variable('x'), Variable('y'))", "Spaces around operator"),
    ("  x  +  y  ", "BinaryOperator('+', Variable('x'), Variable('y'))", "Extra spaces"),
    ("sin( x )", "UnaryOperator('sin', Variable('x'))", "Spaces in function call"),
    ("( a + b ) * c",
     "BinaryOperator('*', BinaryOperator('+', Variable('a'), Variable('b')), Variable('c'))",
     "Spaces with parentheses"),
])
def test_whitespace_handling(input_str, expected_repr, description):
    """Verifies that whitespace is properly ignored."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("x**y", "BinaryOperator('^', Variable('x'), Variable('y'))", "Double asterisk as power"),
    ("2**3", "BinaryOperator('^', Literal(2), Literal(3))", "Numeric double asterisk"),
    ("a ** b ** c",
     "BinaryOperator('^', Variable('a'), BinaryOperator('^', Variable('b'), Variable('c')))",
     "Double asterisk chain (right-associative)"),
])
def test_double_asterisk_operator(input_str, expected_repr, description):
    """Verifies that ** is correctly parsed as power operator."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("sin(x) + cos(y)",
     "BinaryOperator('+', UnaryOperator('sin', Variable('x')), UnaryOperator('cos', Variable('y')))",
     "Two functions in addition"),

    ("exp(x) * ln(y)",
     "BinaryOperator('*', UnaryOperator('exp', Variable('x')), UnaryOperator('ln', Variable('y')))",
     "Two functions in multiplication"),

    ("sqrt(x) / tan(y)",
     "BinaryOperator('/', UnaryOperator('sqrt', Variable('x')), UnaryOperator('tan', Variable('y')))",
     "Two functions in division"),

    ("log(x) ^ cot(y)",
     "BinaryOperator('^', UnaryOperator('log', Variable('x')), UnaryOperator('cot', Variable('y')))",
     "Two functions in power"),
])
def test_multiple_functions_in_expression(input_str, expected_repr, description):
    """Verifies expressions with multiple function calls."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, expected_repr, description", [
    ("x + + y", "BinaryOperator('+', Variable('x'), UnaryOperator('+', Variable('y')))",
     "Plus followed by unary plus"),

    ("x - - y", "BinaryOperator('-', Variable('x'), UnaryOperator('-', Variable('y')))",
     "Minus followed by unary minus"),

    ("x * -y", "BinaryOperator('*', Variable('x'), UnaryOperator('-', Variable('y')))",
     "Multiplication with unary minus"),
])
def test_binary_followed_by_unary(input_str, expected_repr, description):
    """Verifies that binary operators can be followed by unary operators."""
    assert parse_and_get_repr(input_str) == expected_repr, description


@pytest.mark.parametrize("input_str, error_message_part", [
    # Trailing Operator
    ("a + 5 *", "Unexpected token in primary: Token(EOF, None)"),

    # Unclosed Parenthesis
    ("a * (b + 5", "Unexpected token Token(EOF, None), expected OP )"),

    # Trailing Parenthesis
    ("x + y )", "Unexpected input after end of expression"),

    # Double Operator
    ("5 ^ ^ x", "Unexpected token in primary: Token(OP, ^)"),

    # Function with Empty Argument
    ("sin()", "Unexpected token in primary: Token(OP, ))"),

    # Missing closing parenthesis for function
    ("cos(x", "Unexpected token Token(EOF, None), expected OP )"),

    # Invalid start
    ("* x", "Unexpected token in primary: Token(OP, *)"),

    # Empty expression
    ("", "Unexpected token in primary: Token(EOF, None)"),

    # Just operator that's not unary
    ("*", "Unexpected token in primary: Token(OP, *)"),

    # Mismatched parentheses
    ("(a + b))", "Unexpected input after end of expression"),

    # Invalid function name usage
    ("sin x", "Unexpected input after end of expression"),

    # Division without operand
    ("x / / y", "Unexpected token in primary: Token(OP, /)"),
])
def test_syntax_errors_are_raised_correctly(input_str, error_message_part):
    """Verifies that malformed expressions raise ValueError with descriptive messages."""

    with pytest.raises(ValueError) as excinfo:
        Parser(input_str).parse()

    # We check if the actual error message contains the expected part
    assert error_message_part in str(
        excinfo.value), f"Input '{input_str}' raised an unexpected error: {str(excinfo.value)}"
