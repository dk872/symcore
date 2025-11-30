import math
import pytest
from src.engine.Expression import parse

PI_VALUE = math.pi


def get_simplified_string(expression_str: str) -> str:
    """Parses, simplifies, and returns the string representation of an expression."""
    expr = parse(expression_str)
    simplified_expr = expr.simplify()
    return simplified_expr.to_string()


def test_add_zero_left():
    """Tests simplification of adding zero on the left side (0 + x = x)."""
    result = get_simplified_string("0 + x")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_add_zero_right():
    """Tests simplification of adding zero on the right side (x + 0 = x)."""
    result = get_simplified_string("x + 0")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_subtract_zero():
    """Tests simplification of subtracting zero (x - 0 = x)."""
    result = get_simplified_string("x - 0")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_subtract_from_itself():
    """Tests simplification of subtracting a term from itself (x - x = 0)."""
    result = get_simplified_string("x - x")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_by_zero_left():
    """Tests simplification of multiplying by zero on the left (0 * x = 0)."""
    result = get_simplified_string("0 * x")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_by_zero_right():
    """Tests simplification of multiplying by zero on the right (x * 0 = 0)."""
    result = get_simplified_string("x * 0")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_complex_by_zero():
    """Tests simplification of multiplying a complex term by zero (0 * (a + b) = 0)."""
    result = get_simplified_string("0 * (a + b)")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_divide_zero():
    """Tests simplification of dividing zero by a variable (0 / x = 0)."""
    result = get_simplified_string("0 / x")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_divide_zero_by_constant():
    """Tests simplification of dividing zero by a constant (0 / 5 = 0)."""
    result = get_simplified_string("0 / 5")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_by_one_left():
    """Tests simplification of multiplying by one on the left (1 * x = x)."""
    result = get_simplified_string("1 * x")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_by_one_right():
    """Tests simplification of multiplying by one on the right (x * 1 = x)."""
    result = get_simplified_string("x * 1")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_divide_by_one():
    """Tests simplification of dividing by one (x / 1 = x)."""
    result = get_simplified_string("x / 1")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_divide_by_itself():
    """Tests simplification of dividing a term by itself (x / x = 1)."""
    result = get_simplified_string("x / x")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_power_to_zero():
    """Tests simplification of raising a term to the power of zero (x^0 = 1)."""
    result = get_simplified_string("x^0")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_power_to_one():
    """Tests simplification of raising a term to the power of one (x^1 = x)."""
    result = get_simplified_string("x^1")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_zero_to_power():
    """Tests simplification of raising zero to a power (0^5 = 0)."""
    result = get_simplified_string("0^5")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_one_to_power():
    """Tests simplification of raising one to a power (1^100 = 1)."""
    result = get_simplified_string("1^100")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_addition():
    """Tests constant folding for addition (3 + 5)."""
    result = get_simplified_string("3 + 5")
    expected_options = {"8"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_subtraction():
    """Tests constant folding for subtraction (10 - 3)."""
    result = get_simplified_string("10 - 3")
    expected_options = {"7"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_multiplication():
    """Tests constant folding for multiplication (4 * 7)."""
    result = get_simplified_string("4 * 7")
    expected_options = {"28"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_division():
    """Tests constant folding for division (15 / 3)."""
    result = get_simplified_string("15 / 3")
    expected_options = {"5"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_power():
    """Tests constant folding for exponentiation (2^3)."""
    result = get_simplified_string("2^3")
    expected_options = {"8"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_folding_complex():
    """Tests constant folding with precedence (5 + 3 * 2)."""
    result = get_simplified_string("5 + 3 * 2")
    expected_options = {"11"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_folding_with_parentheses():
    """Tests constant folding respecting parentheses ((3 - 1) / 4)."""
    result = get_simplified_string("(3 - 1) / 4")
    expected_options = {"0.5"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_folding_nested():
    """Tests constant folding for nested operations (2^4 / 8 + 3)."""
    result = get_simplified_string("2^4 / 8 + 3")
    expected_options = {"5"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_folding_trig():
    """Tests constant folding for trigonometric functions (sin(pi/6) + cos(pi/3))."""
    result = get_simplified_string(f"sin({PI_VALUE} / 6) + cos({PI_VALUE} / 3)")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_sqrt():
    """Tests constant folding for square root (sqrt(16))."""
    result = get_simplified_string("sqrt(16)")
    expected_options = {"4"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_constant_exp_ln():
    """Tests simplification of exp(ln(T)) = T for constants."""
    result = get_simplified_string("exp(ln(5))")
    expected_options = {"5"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_same_variable():
    """Tests combining identical variable terms (x + x = 2x)."""
    result = get_simplified_string("x + x")
    expected_options = {"2x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_with_coefficients():
    """Tests combining terms with coefficients (3x + 2x = 5x)."""
    result = get_simplified_string("3 * x + 2 * x")
    expected_options = {"5x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_subtract_like_terms():
    """Tests combining terms with subtraction (3x - x = 2x)."""
    result = get_simplified_string("3 * x - x")
    expected_options = {"2x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_multiple_terms():
    """Tests combining multiple variable and constant terms (3x + 5 + x + 1)."""
    result = get_simplified_string("3 * x + 5 + x + 1")
    expected_options = {"4x + 6", "6 + 4x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_cancel_terms():
    """Tests combining terms that cancel out (a + b - a = b)."""
    result = get_simplified_string("a + b - a")
    expected_options = {"b"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_radicals():
    """Tests combining identical radical terms (sqrt(x) + sqrt(x) = 2sqrt(x))."""
    result = get_simplified_string("sqrt(x) + sqrt(x)")
    expected_options = {"2sqrt(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_powers():
    """Tests combining identical power terms (x^2 + x^2 = 2x^2)."""
    result = get_simplified_string("x^2 + x^2")
    expected_options = {"2x ^ 2", "2 * x ^ 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_different_bases():
    """Tests combining different variable bases (x + y + x = 2x + y)."""
    result = get_simplified_string("x + y + x")
    expected_options = {"2x + y", "y + 2x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combine_constants_and_variables():
    """Tests combining constants and variables with sorting (5 + x + 3 + 2x)."""
    result = get_simplified_string("5 + x + 3 + 2*x")
    expected_options = {"3x + 8", "8 + 3x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_same_base():
    """Tests multiplication rule: x^a * x^b = x^(a+b) (x^2 * x^3 = x^5)."""
    result = get_simplified_string("x^2 * x^3")
    expected_options = {"x ^ 5"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_same_base_complex():
    """Tests multiplication rule with base variable (a * a^2 * a^3 = a^6)."""
    result = get_simplified_string("a * a^2 * a^3")
    expected_options = {"a ^ 6"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_divide_same_base():
    """Tests division rule: x^a / x^b = x^(a-b) (x^5 / x^2 = x^3)."""
    result = get_simplified_string("x^5 / x^2")
    expected_options = {"x ^ 3"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_divide_same_base_variables():
    """Tests division rule with variable exponents (a^x / a^y = a^(x-y))."""
    result = get_simplified_string("a^x / a^y")
    expected_options = {"a ^ (x - y)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_power_of_power():
    """Tests power rule: (x^a)^b = x^(a*b) ((x^2)^3 = x^6)."""
    result = get_simplified_string("(x^2)^3")
    expected_options = {"x ^ 6"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_nested_power():
    """Tests power rule for nested powers (((y)^2)^3 = y^6)."""
    result = get_simplified_string("((y)^2)^3")
    expected_options = {"y ^ 6"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_power_with_base():
    """Tests multiplication rule: x * x^a = x^(a+1) (x * x^2 = x^3)."""
    result = get_simplified_string("x * x^2")
    expected_options = {"x ^ 3"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_divide_to_negative_power():
    """Tests division resulting in a negative exponent (x^2 / x^5 = 1/x^3)."""
    result = get_simplified_string("x^2 / x^5")
    expected_options = {"1 / (x ^ 3)", "x ^ -3"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_sqrt_times_sqrt():
    """Tests simplification of sqrt(x) * sqrt(x) = x."""
    result = get_simplified_string("sqrt(x) * sqrt(x)")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_sqrt_of_power():
    """Tests simplification of sqrt(x^6) = x^3."""
    result = get_simplified_string("sqrt(x^6)")
    expected_options = {"x ^ 3"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_sqrt_of_square():
    """Tests simplification of sqrt(x^2) = x (assuming positive domain)."""
    result = get_simplified_string("sqrt(x^2)")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_sqrt_of_product():
    """Tests simplification of sqrt(4 * x^2) = 2x."""
    result = get_simplified_string("sqrt(4 * x^2)")
    expected_options = {"2x", "2 * x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_sqrt_of_constant_product():
    """Tests simplification of sqrt(9 * x) = 3sqrt(x)."""
    result = get_simplified_string("sqrt(9 * x)")
    expected_options = {"3sqrt(x)", "3 * sqrt(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_power_sqrt_squared():
    """Tests simplification of sqrt(x)^2 = x."""
    result = get_simplified_string("sqrt(x)^2")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_distribute_number():
    """Tests distributive property: 2 * (x + 3) = 2x + 6."""
    result = get_simplified_string("2 * (x + 3)")
    expected_options = {"2x + 6", "6 + 2x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_distribute_with_cancellation():
    """Tests distributive property with factor cancellation: x * (1/x - y) = 1 - xy."""
    result = get_simplified_string("x * (1 / x - y)")
    expected_options = {"1 - x * y", "1 + -x * y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_distribute_and_cancel():
    """Tests full cancellation after distribution: 3(x + 1) - 3x = 3."""
    result = get_simplified_string("3 * (x + 1) - 3 * x")
    expected_options = {"3"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_simplify_fraction_same_factor():
    """Tests simplification of fractions with common factors: (2 * x) / 2 = x."""
    result = get_simplified_string("(2 * x) / 2")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_simplify_fraction_powers():
    """Tests simplification of fractions with powers: (x^3 * y^2) / (x * y) = x^2 * y."""
    result = get_simplified_string("(x^3 * y^2) / (x * y)")
    expected_options = {"x ^ 2 * y", "y * x ^ 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_simplify_fraction_cancel():
    """Tests simplification of fractions: (a * b) / a = b."""
    result = get_simplified_string("(a * b) / a")
    expected_options = {"b"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_add_fractions_same_denominator():
    """Tests addition of fractions with common denominator: x/y + z/y = (x+z)/y."""
    result = get_simplified_string("x / y + z / y")
    expected_options = {"(x + z) / y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_multiply_fractions():
    """Tests multiplication of fractions with cancellation: (a / b) * (b / c) = a/c."""
    result = get_simplified_string("(a / b) * (b / c)")
    expected_options = {"a / c"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_divide_fractions():
    """Tests division of fractions: (a / b) / (c / d) = ad/bc."""
    result = get_simplified_string("(a / b) / (c / d)")
    expected_options = {"(a * d) / (b * c)", "a * d / (b * c)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_ln_of_exp():
    """Tests simplification of ln(exp(x)) = x."""
    result = get_simplified_string("ln(exp(x))")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_exp_of_ln():
    """Tests simplification of exp(ln(y)) = y."""
    result = get_simplified_string("exp(ln(y))")
    expected_options = {"y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_ln_of_power():
    """Tests simplification of ln(x^2) = 2ln(x)."""
    result = get_simplified_string("ln(x ^ 2)")
    expected_options = {"2ln(x)", "2 * ln(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_exp_of_product_with_ln():
    """Tests simplification of exp(3 * ln(x)) = x^3."""
    result = get_simplified_string("exp(3 * ln(x))")
    expected_options = {"x ^ 3"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_exp_of_sum_with_ln():
    """Tests simplification of exp((a + 1) * ln(x)) = x^(1+a)."""
    result = get_simplified_string("exp((a + 1) * ln(x))")
    expected_options = {"x ^ (1 + a)", "x ^ (a + 1)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_ln_of_one():
    """Tests simplification of ln(1) = 0."""
    result = get_simplified_string("ln(1)")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_exp_of_zero():
    """Tests simplification of exp(0) = 1."""
    result = get_simplified_string("exp(0)")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_sin_of_zero():
    """Tests simplification of sin(0) = 0."""
    result = get_simplified_string("sin(0)")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_cos_of_zero():
    """Tests simplification of cos(0) = 1."""
    result = get_simplified_string("cos(0)")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_sin_squared_plus_cos_squared():
    """Tests simplification of sin(x)^2 + cos(x)^2 = 1."""
    result = get_simplified_string("sin(x)^2 + cos(x)^2")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_tan_as_sin_over_cos():
    """Tests simplification of sin(x) / cos(x) = tan(x)."""
    result = get_simplified_string("sin(x) / cos(x)")
    expected_options = {"tan(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_cot_as_cos_over_sin():
    """Tests simplification of cos(x) / sin(x) = cot(x)."""
    result = get_simplified_string("cos(x) / sin(x)")
    expected_options = {"cot(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_nested_addition():
    """Tests associativity of addition and flattening: (x + y) + (z + w)."""
    result = get_simplified_string("(x + y) + (z + w)")
    expected_options = {"x + y + z + w", "w + x + y + z"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_nested_multiplication():
    """Tests associativity of multiplication: (2 * x) * (3 * y) = 6xy."""
    result = get_simplified_string("(2 * x) * (3 * y)")
    expected_options = {"6x * y", "6 * x * y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_nested_powers():
    """Tests power of a power: ((x^2)^2)^2 = x^8."""
    result = get_simplified_string("((x^2)^2)^2")
    expected_options = {"x ^ 8"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_double_negative():
    """Tests double negation: -(-x) = x."""
    result = get_simplified_string("-(-x)")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_negative_times_negative():
    """Tests multiplication of two negative terms: (-a) * (-b) = ab."""
    result = get_simplified_string("(-a) * (-b)")
    expected_options = {"a * b", "a*b"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_subtract_negative():
    """Tests subtraction of a negative term: x - (-y) = x + y."""
    result = get_simplified_string("x - (-y)")
    expected_options = {"x + y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_nested_sign_handling():
    """Tests complex sign distribution: a - (b - (c - d))."""
    result = get_simplified_string("a - (b - (c - d))")
    expected_options = {"a + c - b - d"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_factoring_simple():
    """Tests factoring out a common numeric factor: 2x + 2y = 2(x + y)."""
    result = get_simplified_string("2*x + 2*y")
    expected_options = {"2(x + y)", "2x + 2y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_combining_fractions_and_powers():
    """Tests combining fractions resulting from power simplification: x^2/x + x^3/x^2 = 2x."""
    result = get_simplified_string("x^2 / x + x^3 / x^2")
    expected_options = {"2x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_simplify_already_simple():
    """Tests simplification of an already simple variable (x = x)."""
    result = get_simplified_string("x")
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_simplify_constant():
    """Tests simplification of a constant (42 = 42)."""
    result = get_simplified_string("42")
    expected_options = {"42"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_simplify_negative_constant():
    """Tests simplification of a negative constant (-5 = -5)."""
    result = get_simplified_string("-5")
    expected_options = {"-5"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_very_large_number():
    """Tests constant folding for very large numbers (1M + 1M)."""
    result = get_simplified_string("1000000 + 1000000")
    expected_options = {"2000000"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_very_small_fraction():
    """Tests constant folding for very small fractions (1 / 1M)."""
    result = get_simplified_string("1 / 1000000")
    expected_options = {"0.000001"}
    assert result in expected_options or "1e-06" in result, \
        f"Expected one of {expected_options} or scientific notation, got: {result}"


def test_mixed_operations():
    """Tests complex combining with cancellation (x + 2x - x = 2x)."""
    result = get_simplified_string("x + x * 2 - x")
    expected_options = {"2x", "2 * x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_zero_times_infinity_form():
    """Tests division by zero scenario that results in ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        get_simplified_string("0 * (1 / 0)")


def test_commutative_property():
    """Tests if the simplified representation respects commutativity (x + y vs y + x)."""
    result1 = get_simplified_string("x + y")
    result2 = get_simplified_string("y + x")
    assert result1 == result2, f"Expected simplified forms to be identical, got: {result1} and {result2}"


def test_associative_property():
    """Tests if the simplified representation respects associativity ((x + y) + z vs x + (y + z))."""
    result1 = get_simplified_string("(x + y) + z")
    result2 = get_simplified_string("x + (y + z)")
    assert result1 == result2, f"Expected simplified forms to be identical, got: {result1} and {result2}"


def test_ln_negative():
    """Test that ln(-1) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string("ln(-1)")
    assert "Domain error" in str(excinfo.value)
    assert "ln" in str(excinfo.value)


def test_ln_zero():
    """Test that ln(0) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string("ln(0)")
    assert "Domain error" in str(excinfo.value)
    assert "ln" in str(excinfo.value)


def test_log_negative():
    """Test that log(-5) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string("log(-5)")
    assert "Domain error" in str(excinfo.value)
    assert "log" in str(excinfo.value)


def test_log_zero():
    """Test that log(0) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string("log(0)")
    assert "Domain error" in str(excinfo.value)
    assert "log" in str(excinfo.value)


def test_sqrt_negative():
    """Test that sqrt(-3) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string("sqrt(-3)")
    assert "Domain error" in str(excinfo.value)
    assert "sqrt" in str(excinfo.value)


def test_tan_pi_over_2():
    """Test that tan(pi/2) raises domain error (asymptote)."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string(f"tan({PI_VALUE} / 2)")
    assert "Domain error" in str(excinfo.value)
    assert "tan" in str(excinfo.value)


def test_tan_3pi_over_2():
    """Test that tan(3*pi/2) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string(f"tan(3 * {PI_VALUE} / 2)")
    assert "Domain error" in str(excinfo.value)
    assert "tan" in str(excinfo.value)


def test_cot_zero():
    """Test that cot(0) raises domain error (asymptote)."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string("cot(0)")
    assert "Domain error" in str(excinfo.value)
    assert "cot" in str(excinfo.value)


def test_cot_pi():
    """Test that cot(pi) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        get_simplified_string(f"cot({PI_VALUE})")
    assert "Domain error" in str(excinfo.value)
    assert "cot" in str(excinfo.value)


def test_division_by_zero_constant():
    """Test that 2/0 raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError) as excinfo:
        get_simplified_string("2 / 0")
    assert "division by zero" in str(excinfo.value).lower()
