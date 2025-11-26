import math
from typing import Dict, Union
import pytest
from src.engine.Expression import parse

PI_VALUE = math.pi
E_VALUE = math.e
FLOAT_TOLERANCE = 1e-12


def evaluate_expr(expression_str: str, values: Dict[str, Union[int, float]] = None) -> Union[int, float]:
    """Parses expression and numerically evaluates it with given values."""
    expr = parse(expression_str)
    return expr.evaluate(values)


def substitute_expr(expression_str: str, sub_map: Dict[str, str]) -> str:
    """Parses expression, performs symbolic substitution and returns string representation."""
    expr = parse(expression_str)
    substituted_expr = expr.substitute(sub_map)
    return substituted_expr.to_string()


def test_addition():
    """Test basic addition."""
    result = evaluate_expr("2 + 3")
    expected = 5
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_subtraction():
    """Test basic subtraction."""
    result = evaluate_expr("10 - 4")
    expected = 6
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_multiplication():
    """Test basic multiplication."""
    result = evaluate_expr("5 * 6")
    expected = 30
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_division():
    """Test basic division."""
    result = evaluate_expr("15 / 3")
    expected = 5.0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_power():
    """Test power operation."""
    result = evaluate_expr("2 ^ 3")
    expected = 8
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_operator_precedence():
    """Test operator precedence: 2 + 3 * 4 = 14."""
    result = evaluate_expr("2 + 3 * 4")
    expected = 14
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_parentheses_precedence():
    """Test parentheses override precedence: (5 - 1) / 2 = 2."""
    result = evaluate_expr("(5 - 1) / 2")
    expected = 2.0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_complex_expression():
    """Test complex expression: (3 + 5) * 2 ^ 2 - 1."""
    result = evaluate_expr("(3 + 5) * 2 ^ 2 - 1")
    expected = 31
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_float_addition_precision():
    """Test that 1.5 + 0.5 = 2 (exact)."""
    result = evaluate_expr("1.5 + 0.5")
    expected = 2
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_sqrt_perfect_square():
    """Test that sqrt(4) = 2 (exact)."""
    result = evaluate_expr("sqrt(4)")
    expected = 2
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_power_fractional_exact():
    """Test that 4^(1/2) = 2.0."""
    result = evaluate_expr("4 ^ (1 / 2)")
    expected = 2.0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_exp_ln_annihilation():
    """Test that exp(ln(5)) = 5 (exact, no floating point error)."""
    result = evaluate_expr("exp(ln(5))")
    expected = 5
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_ln_exp_annihilation():
    """Test that ln(exp(3)) = 3."""
    result = evaluate_expr("ln(exp(3))")
    expected = 3
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_sqrt_square_annihilation():
    """Test that sqrt(9) * sqrt(9) = 9."""
    result = evaluate_expr("sqrt(9) * sqrt(9)")
    expected = 9
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_very_small_number():
    """Test evaluation of very small number."""
    result = evaluate_expr("0.000001 + 0.000002")
    expected = 0.000003
    assert abs(result - expected) < FLOAT_TOLERANCE


def test_very_large_number():
    """Test evaluation of very large number."""
    result = evaluate_expr("1000000 + 2000000")
    expected = 3000000
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_sin_zero():
    """Test sin(0) = 0."""
    result = evaluate_expr("sin(0)")
    expected = 0
    assert abs(result - expected) < FLOAT_TOLERANCE


def test_cos_zero():
    """Test cos(0) = 1."""
    result = evaluate_expr("cos(0)")
    expected = 1
    assert abs(result - expected) < FLOAT_TOLERANCE


def test_sin_pi_over_2():
    """Test sin(pi/2) = 1."""
    result = evaluate_expr(f"sin({PI_VALUE} / 2)")
    expected = 1.0
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_cos_pi():
    """Test cos(pi) = -1."""
    result = evaluate_expr(f"cos({PI_VALUE})")
    expected = -1.0
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_tan_pi_over_4():
    """Test tan(pi/4) = 1."""
    result = evaluate_expr(f"tan({PI_VALUE} / 4)")
    expected = 1.0
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_sin_pi():
    """Test sin(pi) ≈ 0 (within tolerance)."""
    result = evaluate_expr(f"sin({PI_VALUE})")
    assert abs(result) < FLOAT_TOLERANCE


def test_cos_pi_over_2():
    """Test cos(pi/2) ≈ 0 (within tolerance)."""
    result = evaluate_expr(f"cos({PI_VALUE} / 2)")
    assert abs(result) < FLOAT_TOLERANCE


def test_sin_30_degrees():
    """Test sin(pi/6) = 0.5."""
    result = evaluate_expr(f"sin({PI_VALUE} / 6)")
    expected = 0.5
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_cos_60_degrees():
    """Test cos(pi/3) = 0.5."""
    result = evaluate_expr(f"cos({PI_VALUE} / 3)")
    expected = 0.5
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_log_100():
    """Test log10(100) = 2."""
    result = evaluate_expr("log(100)")
    expected = 2
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_log_1000():
    """Test log10(1000) = 3."""
    result = evaluate_expr("log(1000)")
    expected = 3
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_ln_e():
    """Test ln(e) = 1."""
    result = evaluate_expr(f"ln({E_VALUE})")
    expected = 1
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_ln_1():
    """Test ln(1) = 0."""
    result = evaluate_expr("ln(1)")
    expected = 0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_exp_0():
    """Test exp(0) = 1."""
    result = evaluate_expr("exp(0)")
    expected = 1
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_exp_1():
    """Test exp(1) ≈ e."""
    result = evaluate_expr("exp(1)")
    expected = E_VALUE
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_exp_ln_identity():
    """Test exp(ln(10)) = 10."""
    result = evaluate_expr("exp(ln(10))")
    expected = 10
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_ln_exp_identity():
    """Test ln(exp(5)) = 5."""
    result = evaluate_expr("ln(exp(5))")
    expected = 5
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_sqrt_0():
    """Test sqrt(0) = 0."""
    result = evaluate_expr("sqrt(0)")
    expected = 0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_sqrt_1():
    """Test sqrt(1) = 1."""
    result = evaluate_expr("sqrt(1)")
    expected = 1
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_sqrt_16():
    """Test sqrt(16) = 4."""
    result = evaluate_expr("sqrt(16)")
    expected = 4
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_sqrt_2():
    """Test sqrt(2) ≈ 1.414."""
    result = evaluate_expr("sqrt(2)")
    expected = math.sqrt(2)
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_sqrt_fraction():
    """Test sqrt(0.25) = 0.5."""
    result = evaluate_expr("sqrt(0.25)")
    expected = 0.5
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_single_variable():
    """Test x + 5 where x = 10."""
    result = evaluate_expr("x + 5", {'x': 10})
    expected = 15
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_in_multiplication():
    """Test 2 * x where x = 7."""
    result = evaluate_expr("2 * x", {'x': 7})
    expected = 14
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_in_power():
    """Test x ^ 2 where x = 5."""
    result = evaluate_expr("x ^ 2", {'x': 5})
    expected = 25
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_in_division():
    """Test 100 / x where x = 4."""
    result = evaluate_expr("100 / x", {'x': 4})
    expected = 25.0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_negative_value():
    """Test x ^ 2 where x = -3."""
    result = evaluate_expr("x ^ 2", {'x': -3})
    expected = 9
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_two_variables():
    """Test x + y where x = 10, y = 5."""
    result = evaluate_expr("x + y", {'x': 10, 'y': 5})
    expected = 15
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_expression():
    """Test 2 * x - 1 where x = 4."""
    result = evaluate_expr("2 * x - 1", {'x': 4})
    expected = 7
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_power_and_addition():
    """Test x ^ 2 + y where x = 3, y = -5."""
    result = evaluate_expr("x ^ 2 + y", {'x': 3, 'y': -5})
    expected = 4
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_division_variables():
    """Test a / b where a = 1, b = 4."""
    result = evaluate_expr("a / b", {'a': 1, 'b': 4})
    expected = 0.25
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_with_trig():
    """Test x * sin(pi / 6) where x = 10."""
    result = evaluate_expr(f"x * sin({PI_VALUE} / 6)", {'x': 10})
    expected = 5.0
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_substitute_complex_expression():
    """Test (a + b) * (c - d) where a=2, b=3, c=5, d=1."""
    result = evaluate_expr("(a + b) * (c - d)", {'a': 2, 'b': 3, 'c': 5, 'd': 1})
    expected = 20
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_polynomial_identity():
    """Test x^2 - (x+1)(x-1) - 1.0 = 0.0 where x = 10."""
    result = evaluate_expr("x^2 - (x+1)*(x-1) - 1.0", {'x': 10})
    expected = 0.0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_trig_identity():
    """Test sin(x)^2 + cos(x)^2 = 1 where x = pi/4."""
    result = evaluate_expr("sin(x)^2 + cos(x)^2", {'x': PI_VALUE / 4})
    expected = 1.0
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_substitute_exp_identity():
    """Test exp(a) * exp(b) = exp(a+b) where a=1, b=2."""
    result = evaluate_expr("exp(a) * exp(b)", {'a': 1, 'b': 2})
    expected = evaluate_expr("exp(a + b)", {'a': 1, 'b': 2})
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_symbolic_substitute_variable():
    """Test x^2 + 1 where x = y."""
    result = substitute_expr("x^2 + 1", {'x': 'y'})
    expected_options = {"1 + y ^ 2", "y ^ 2 + 1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_symbolic_substitute_expression():
    """Test x^2 + 1 where x = y + 1."""
    result = substitute_expr("x^2 + 1", {'x': 'y + 1'})
    expected_options = {"1 + (y + 1) ^ 2", "(y + 1) ^ 2 + 1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_symbolic_substitute_multiple_vars():
    """Test a * b + c where a = x/2, c = z."""
    result = substitute_expr("a * b + c", {'a': 'x/2', 'c': 'z'})
    expected_options = {"x / 2 * b + z", "z + x / 2 * b"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_symbolic_substitute_no_simplification():
    """Test that substitution doesn't simplify algebraically."""
    result = substitute_expr("(a + b) * (a - b)", {'a': 'x * 2', 'b': '5'})
    expected_options = {"(x * 2 + 5) * (x * 2 - 5)", "(5 + x * 2) * (x * 2 - 5)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_symbolic_substitute_in_function():
    """Test sin(x) where x = 2*y."""
    result = substitute_expr("sin(x)", {'x': '2 * y'})
    expected_options = {"sin(2y)", "sin(2 * y)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_symbolic_substitute_nested():
    """Test (x + y)^2 where x = a, y = b."""
    result = substitute_expr("(x + y)^2", {'x': 'a', 'y': 'b'})
    expected = "(a + b) ^ 2"
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_ln_negative():
    """Test that ln(-1) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("ln(-1)")
    assert "Domain error" in str(excinfo.value)
    assert "ln" in str(excinfo.value)


def test_ln_zero():
    """Test that ln(0) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("ln(0)")
    assert "Domain error" in str(excinfo.value)
    assert "ln" in str(excinfo.value)


def test_log_negative():
    """Test that log(-10) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("log(-10)")
    assert "Domain error" in str(excinfo.value)
    assert "log" in str(excinfo.value)


def test_log_zero():
    """Test that log(0) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("log(0)")
    assert "Domain error" in str(excinfo.value)
    assert "log" in str(excinfo.value)


def test_ln_expression_negative():
    """Test that ln(x-5) raises error when x=3."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("ln(x - 5)", {'x': 3})
    assert "Domain error" in str(excinfo.value)
    assert "ln" in str(excinfo.value)


def test_sqrt_negative():
    """Test that sqrt(-4) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("sqrt(-4)")
    assert "Domain error" in str(excinfo.value)
    assert "sqrt" in str(excinfo.value)


def test_sqrt_negative_variable():
    """Test that sqrt(x) raises error when x=-1."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("sqrt(x)", {'x': -1})
    assert "Domain error" in str(excinfo.value)
    assert "sqrt" in str(excinfo.value)


def test_sqrt_expression_negative():
    """Test that sqrt(x-10) raises error when x=5."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("sqrt(x - 10)", {'x': 5})
    assert "Domain error" in str(excinfo.value)
    assert "sqrt" in str(excinfo.value)


def test_tan_pi_over_2():
    """Test that tan(pi/2) raises domain error (asymptote)."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr(f"tan({PI_VALUE} / 2)")
    assert "Domain error" in str(excinfo.value)
    assert "tan" in str(excinfo.value)


def test_tan_3pi_over_2():
    """Test that tan(3*pi/2) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr(f"tan(3 * {PI_VALUE} / 2)")
    assert "Domain error" in str(excinfo.value)
    assert "tan" in str(excinfo.value)


def test_cot_zero():
    """Test that cot(0) raises domain error (asymptote)."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("cot(0)")
    assert "Domain error" in str(excinfo.value)
    assert "cot" in str(excinfo.value)


def test_cot_pi():
    """Test that cot(pi) raises domain error."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr(f"cot({PI_VALUE})")
    assert "Domain error" in str(excinfo.value)
    assert "cot" in str(excinfo.value)


def test_division_by_zero_constant():
    """Test that 5/0 raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError) as excinfo:
        evaluate_expr("5 / 0")
    assert "division by zero" in str(excinfo.value).lower()


def test_division_by_zero_variable():
    """Test that x/0 raises error."""
    with pytest.raises(ZeroDivisionError) as excinfo:
        evaluate_expr("x / 0", {'x': 10})
    assert "division by zero" in str(excinfo.value).lower()


def test_division_by_zero_expression():
    """Test that 1/(x-5) raises error when x=5."""
    with pytest.raises(ZeroDivisionError) as excinfo:
        evaluate_expr("1 / (x - 5)", {'x': 5})
    assert "division by zero" in str(excinfo.value).lower()


def test_unbound_variable_single():
    """Test that x + 1 raises error when x is not provided."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("x + 1")
    assert "Unbound variable x" in str(excinfo.value)


def test_unbound_variable_multiple():
    """Test that x + y raises error when only x is provided."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("x + y", {'x': 5})
    assert "Unbound variable y" in str(excinfo.value)


def test_unbound_variable_in_function():
    """Test that sin(x) raises error when x is not provided."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("sin(x)")
    assert "Unbound variable x" in str(excinfo.value)


def test_unbound_variable_complex():
    """Test that (a + b) * c raises error when not all vars provided."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_expr("(a + b) * c", {'a': 1, 'b': 2})
    assert "Unbound variable c" in str(excinfo.value)


def test_evaluate_constant_only():
    """Test that evaluating pure constant works."""
    result = evaluate_expr("42")
    expected = 42
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_evaluate_pi():
    """Test that evaluating pi works."""
    result = evaluate_expr("pi")
    expected = PI_VALUE
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_evaluate_negative_constant():
    """Test that evaluating negative constant works."""
    result = evaluate_expr("-5")
    expected = -5
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_evaluate_zero():
    """Test that evaluating 0 works."""
    result = evaluate_expr("0")
    expected = 0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_zero():
    """Test substitution with zero."""
    result = evaluate_expr("x * 5", {'x': 0})
    expected = 0
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_substitute_negative():
    """Test substitution with negative value."""
    result = evaluate_expr("x + 10", {'x': -5})
    expected = 5
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_very_nested_expression():
    """Test deeply nested expression."""
    result = evaluate_expr("((((1 + 1) * 2) + 3) * 2)")
    expected = 14
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_mixed_operations():
    """Test expression with all basic operations."""
    result = evaluate_expr("2 + 3 * 4 - 5 / 5 ^ 2")
    expected = 2 + 3 * 4 - 5 / (5 ** 2)
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_fractional_division():
    """Test 1/3 + 1/6 = 1/2."""
    result = evaluate_expr("1/3 + 1/6")
    expected = 0.5
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_decimal_precision():
    """Test 0.1 + 0.2 (classic floating point issue)."""
    result = evaluate_expr("0.1 + 0.2")
    expected = 0.3
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_repeating_decimal():
    """Test 1/7 evaluation."""
    result = evaluate_expr("1 / 7")
    expected = 1.0 / 7.0
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_negative_base_even_power():
    """Test (-2)^2 = 4."""
    result = evaluate_expr("(-2) ^ 2")
    expected = 4
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_negative_base_odd_power():
    """Test (-2)^3 = -8."""
    result = evaluate_expr("(-2) ^ 3")
    expected = -8
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_fractional_power():
    """Test 8^(1/3) = 2."""
    result = evaluate_expr("8 ^ (1/3)")
    expected = 2.0
    assert pytest.approx(result) == expected, f"Expected: {expected}, got: {result}"


def test_negative_power():
    """Test 2^(-3) = 0.125."""
    result = evaluate_expr("2 ^ (-3)")
    expected = 0.125
    assert result == expected, f"Expected: {expected}, got: {result}"


def test_zero_power():
    """Test 5^0 = 1."""
    result = evaluate_expr("5 ^ 0")
    expected = 1
    assert result == expected, f"Expected: {expected}, got: {result}"
