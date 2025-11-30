from src.engine.Expression import parse


def get_derived_string(expression_str: str, variable: str = 'x') -> str:
    """Parses expression, differentiates it and returns simplified derivative string."""
    expr = parse(expression_str)
    derived_expr = expr.diff(variable)
    return derived_expr.to_string()


def test_derivative_of_constant():
    """Tests that the derivative of a constant (Literal) should be 0."""
    result = get_derived_string("5")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"

    result = get_derived_string("pi")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"

    result = get_derived_string("sin(1)")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_variable():
    """Tests that the derivative of x with respect to x should be 1."""
    result = get_derived_string("x")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_variable_by_different_variable():
    """Tests that the derivative of y with respect to x should be 0 (partial derivative)."""
    result = get_derived_string("y", variable='x')
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_sum_and_difference():
    """Tests the sum/difference rule: (x^2 + 3x - 5)' = 2x + 3."""
    result = get_derived_string("x^2 + 3*x - 5")
    expected_options = {"2x + 3"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_multiple_terms():
    """Tests the derivative of a polynomial with multiple terms."""
    result = get_derived_string("x^3 + x^2 + x + 1")
    expected_options = {"3x^2 + 2x + 1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_constant_multiple():
    """Tests the derivative with constant coefficients: (ax)' = a."""
    result = get_derived_string("7*x")
    expected_options = {"7"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"

    result = get_derived_string("pi*x")
    expected_options = {"3.141592653589793"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_product_rule_simple():
    """Tests simple product rule: (x*sin(x))' = sin(x) + x*cos(x)."""
    result = get_derived_string("x * sin(x)")
    expected_options = {"cos(x) * x + sin(x)", "sin(x) + cos(x) * x", "sin(x) + x * cos(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_product_rule_with_exponent():
    """Tests product rule with exponent: (x^2 * exp(x))'."""
    result = get_derived_string("x^2 * exp(x)")
    expected_options = {"exp(x) * (2x + x ^ 2)", "2 * exp(x) * x + exp(x) * x ^ 2", "exp(x) * x ^ 2 + 2exp(x) * x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_product_rule_with_constants():
    """Tests product rule with constant: (5x^3)' = 15x^2."""
    result = get_derived_string("5 * x^3")
    expected_options = {"15x ^ 2", "15 * x ^ 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_product_rule_three_factors():
    """Tests product rule with three factors: (x * y * z)' with respect to x."""
    result = get_derived_string("x * y * z", variable='x')
    expected_options = {"y * z"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_product_rule_polynomial_times_trig():
    """Tests product of polynomial and trigonometric function."""
    result = get_derived_string("(x^2 + 1) * sin(x)")
    expected_options = {"(1 + x ^ 2) * cos(x) + 2 * sin(x) * x", "2sin(x) * x + (1 + x ^ 2) * cos(x)",
                        "(x ^ 2 + 1) * cos(x) + 2x * sin(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_product_rule_two_polynomials():
    """Tests product of two polynomials: ((x+1)*(x-1))' = (x^2-1)' = 2x."""
    result = get_derived_string("(x + 1) * (x - 1)")
    expected_options = {"2x", "2 * x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_quotient_rule_simple():
    """Tests simple quotient rule: (x/y)' with respect to x."""
    result = get_derived_string("x / y", variable='x')
    expected_options = {"1 / y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_quotient_rule_complex():
    """Tests complex quotient rule: (x / sin(x))'."""
    result = get_derived_string("x / sin(x)")
    expected_options = {"(-cos(x) * x + sin(x)) / ((sin(x)) ^ 2)", "(sin(x) - cos(x) * x) / ((sin(x)) ^ 2)",
                        "(sin(x) - x * cos(x)) / (sin(x) ^ 2)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_quotient_rule_with_reciprocal():
    """Tests derivative of 1/x should be -1/x^2."""
    result = get_derived_string("1 / x")
    expected_options = {"-1 / (x ^ 2)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_quotient_rule_trig_over_trig():
    """Tests quotient of trigonometric functions: (sin(x)/cos(x))' = 1/cos^2(x)."""
    result = get_derived_string("sin(x) / cos(x)")
    expected_options = {"1 / ((cos(x)) ^ 2)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_quotient_rule_constant_numerator():
    """Tests quotient with constant numerator: (5/x^2)'."""
    result = get_derived_string("5 / x^2")
    expected_options = {"-10 / (x ^ 3)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_chain_rule_power_of_function():
    """Tests chain rule for power: (sin(x)^2)' = 2*sin(x)*cos(x)."""
    result = get_derived_string("sin(x)^2")
    expected_options = {"2cos(x) * sin(x)", "2sin(x) * cos(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_chain_rule_complex_exponent():
    """Tests chain rule for exponential: (exp(x^3))' = 3x^2 * exp(x^3)."""
    result = get_derived_string("exp(x^3)")
    expected_options = {"3exp(x ^ 3) * x ^ 2", "3x ^ 2 * exp(x ^ 3)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_chain_rule_nested_functions():
    """Tests chain rule for nested function: (ln(cos(x)))' = -tan(x)."""
    result = get_derived_string("ln(cos(x))")
    expected_options = {"-sin(x) / cos(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_chain_rule_with_simplified_argument():
    """Tests simplified argument after chain rule: cos(3 - 2x)' = 2sin(3 - 2x)."""
    result = get_derived_string("cos(3 - 2*x)")
    expected_options = {"2sin(-2x + 3)", "2sin(3 - 2x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_chain_rule_nested_trig():
    """Tests nested trigonometric functions: sin(cos(x))'."""
    result = get_derived_string("sin(cos(x))")
    expected_options = {"-cos(cos(x)) * sin(x)", "-sin(x) * cos(cos(x))"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_chain_rule_exp_of_sin():
    """Tests chain rule: (exp(sin(x)))' = cos(x)*exp(sin(x))."""
    result = get_derived_string("exp(sin(x))")
    expected_options = {"cos(x) * exp(sin(x))", "exp(sin(x)) * cos(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_chain_rule_ln_of_polynomial():
    """Tests chain rule: (ln(x^2 + 1))' = 2x/(x^2+1)."""
    result = get_derived_string("ln(x^2 + 1)")
    expected_options = {"2x / (1 + x ^ 2)", "2x / (x ^ 2 + 1)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_chain_rule_sqrt_of_expression():
    """Tests chain rule: (sqrt(x^2 + 1))' = x/sqrt(x^2+1)."""
    result = get_derived_string("sqrt(x^2 + 1)")
    expected_options = {"x / sqrt(1 + x ^ 2)", "x / sqrt(x ^ 2 + 1)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_power_rule_simple():
    """Tests power rule: (x^n)' = n*x^(n-1)."""
    result = get_derived_string("x^5")
    expected_options = {"5x ^ 4", "5 * x ^ 4"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"

    result = get_derived_string("x^3")
    expected_options = {"3x ^ 2", "3 * x ^ 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_power_rule_negative_exponent():
    """Tests power rule with negative exponent: (x^-2)' = -2*x^-3."""
    result = get_derived_string("x^(-2)")
    expected_options = {"-2 / (x ^ 3)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_power_rule_fractional_exponent():
    """Tests power rule with fractional exponent: (x^0.5)' = 0.5*x^-0.5."""
    result = get_derived_string("x^0.5")
    expected_options = {"0.5 * (x ^ -0.5)", "0.5 / (x ^ 0.5)", "0.5 / sqrt(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_power_rule_with_chain():
    """Tests power rule with chain: ((2x)^3)' = 6*(2x)^2."""
    result = get_derived_string("(2*x)^3")
    expected_options = {"6(2x) ^ 2", "24x ^ 2", "6 * (2x) ^ 2", "24 * x ^ 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_sin():
    """Tests derivative of sin(x) should be cos(x)."""
    result = get_derived_string("sin(x)")
    expected_options = {"cos(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_cos():
    """Tests derivative of cos(x) should be -sin(x)."""
    result = get_derived_string("cos(x)")
    expected_options = {"-sin(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_tan():
    """Tests derivative of tan(x) should be 1/cos(x)^2."""
    result = get_derived_string("tan(x)")
    expected_options = {"1 / ((cos(x)) ^ 2)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_cot():
    """Tests derivative of cot(x) should be -1/sin(x)^2."""
    result = get_derived_string("cot(x)")
    expected_options = {"-1 / ((sin(x)) ^ 2)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_cot_complex_argument():
    """Tests derivative of cot(x^2) with chain rule."""
    result = get_derived_string("cot(x^2)")
    expected_options = {"-2x / ((sin(x ^ 2)) ^ 2)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_sqrt():
    """Tests derivative of sqrt(x) should be 0.5 / sqrt(x)."""
    result = get_derived_string("sqrt(x)")
    expected_options = {"0.5 / sqrt(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_exp():
    """Tests derivative of exp(x) should be exp(x)."""
    result = get_derived_string("exp(x)")
    expected_options = {"exp(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_ln():
    """Tests derivative of ln(x) should be 1/x."""
    result = get_derived_string("ln(x)")
    expected_options = {"1 / x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_log():
    """Tests derivative of log(x) (base 10 logarithm)."""
    result = get_derived_string("log(x)")
    expected_options = {"0.43429448190325176 / x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_sin_with_coefficient():
    """Tests derivative of a*sin(bx) should be ab*cos(bx)."""
    result = get_derived_string("3*sin(2*x)")
    expected_options = {"6cos(2x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_variable_base_constant_exponent():
    """Tests derivative of x^n where n is constant: (x^3)' = 3x^2."""
    result = get_derived_string("x^3")
    expected_options = {"3x ^ 2", "3 * x ^ 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_constant_base_variable_exponent():
    """Tests derivative of a^x where a is constant: (2^x)' = 2^x * ln(2)."""
    result = get_derived_string("2^x")
    expected_options = {"0.6931471805599453 * 2 ^ x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_partial_derivative_product():
    """Tests partial derivative (x*y)' with respect to x."""
    result = get_derived_string("x * y", variable='x')
    expected_options = {"y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_partial_derivative_of_power():
    """Tests partial derivative (x^y)' with respect to x (y treated as constant n)."""
    result = get_derived_string("x^y", variable='x')
    expected_options = {"x ^ y * y / x", "y * x ^ (y - 1)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_partial_derivative_of_power_by_y():
    """Tests partial derivative (x^y)' with respect to y (logarithmic differentiation)."""
    result = get_derived_string("x^y", variable='y')
    expected_options = {"ln(x) * x ^ y", "x ^ y * ln(x)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_partial_derivative_complex_expression():
    """Tests partial derivative of x^2*y^3 with respect to x."""
    result = get_derived_string("x^2 * y^3", variable='x')
    expected_options = {"2x * y ^ 3", "2 * x * y ^ 3", "y ^ 3 * 2x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_partial_derivative_with_constant_variable():
    """Tests partial derivative where variable doesn't appear."""
    result = get_derived_string("y^2 + 2*y + 1", variable='x')
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_complex_expression_product_quotient():
    """Tests complex expression with product and quotient."""
    result = get_derived_string("(x^2 * sin(x)) / cos(x)")
    expected_options = {"((cos(x)) ^ 2 * x ^ 2 + (sin(x)) ^ 2 * x ^ 2 + 2 * cos(x) * sin(x) * x) / ((cos(x)) ^ 2)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_complex_nested_expression():
    """Tests deeply nested expression: exp(sin(x^2))'."""
    result = get_derived_string("exp(sin(x^2))")
    expected_options = {"2cos(x ^ 2) * exp(sin(x ^ 2)) * x", "2x * cos(x ^ 2) * exp(sin(x ^ 2))",
                        "exp(sin(x ^ 2)) * 2x * cos(x ^ 2)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_complex_polynomial_in_trig():
    """Tests trigonometric function of polynomial: sin(x^3 + 2x)'."""
    result = get_derived_string("sin(x ^ 3 + 2 * x)")
    expected_options = {"2cos(2x + x ^ 3) + 3 * cos(2x + x ^ 3) * x ^ 2",
                        "2cos(2x + x ^ 3) + 3cos(2x + x ^ 3) * x ^ 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_complex_quotient_with_chain():
    """Tests complex quotient with chain rule: (sin(x^2) / x)'."""
    result = get_derived_string("sin(x^2) / x")
    expected_options = {"(2 * cos(x ^ 2) * x ^ 2 - sin(x ^ 2)) / (x ^ 2)",
                        "(2cos(x ^ 2) * x ^ 2 - sin(x ^ 2)) / (x ^ 2)",
                        "(2x * cos(x ^ 2) - sin(x ^ 2) / x) / x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_complex_product_of_quotients():
    """Tests product of quotients: (x/y) * (y/z) with respect to x."""
    result = get_derived_string("(x/y) * (y/z)")
    expected_options = {"1 / z", "z ^ -1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_zero():
    """Tests derivative of 0 should be 0."""
    result = get_derived_string("0")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_x_plus_zero():
    """Tests derivative of x + 0 should be 1."""
    result = get_derived_string("x + 0")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_x_times_one():
    """Tests derivative of x * 1 should be 1."""
    result = get_derived_string("x * 1")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_x_to_power_one():
    """Tests derivative of x^1 should be 1."""
    result = get_derived_string("x^1")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_x_to_power_zero():
    """Tests derivative of x^0 (which is 1) should be 0."""
    result = get_derived_string("x^0")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_constant_times_variable_squared():
    """Tests derivative of c*x^2 where c is another variable."""
    result = get_derived_string("a * x^2", variable='x')
    expected_options = {"2a * x", "2 * a * x", "a * 2x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_unary_minus():
    """Tests derivative of -x should be -1."""
    result = get_derived_string("-x")
    expected_options = {"-1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_unary_plus():
    """Tests derivative of +x should be 1."""
    result = get_derived_string("+x")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_nested_unary():
    """Tests derivative of -(-x) should be 1."""
    result = get_derived_string("-(-x)")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_xy_by_x():
    """Tests derivative of x*y with respect to x (y constant)."""
    result = get_derived_string("x * y", variable='x')
    expected_options = {"y"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_xy_by_y():
    """Tests derivative of x*y with respect to y (x constant)."""
    result = get_derived_string("x * y", variable='y')
    expected_options = {"x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_x_plus_y_by_x():
    """Tests derivative of x+y with respect to x should be 1."""
    result = get_derived_string("x + y", variable='x')
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_sin_xy_by_x():
    """Tests derivative of sin(x*y) with respect to x (chain rule)."""
    result = get_derived_string("sin(x * y)", variable='x')
    expected_options = {"cos(x * y) * y", "y * cos(x * y)"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_logarithmic_diff_x_to_x():
    """Tests logarithmic differentiation: (x^x)' = x^x(ln(x) + 1)."""
    result = get_derived_string("x^x")
    expected_options = {"x ^ x * (1 + ln(x))", "x ^ x + x ^ x * ln(x)", "ln(x) * x ^ x + x ^ x"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_simplifies_to_constant():
    """Tests derivative that simplifies to a constant."""
    result = get_derived_string("3*x + 5")
    expected_options = {"3"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_simplifies_zero_terms():
    """Tests derivative where some terms become zero."""
    result = get_derived_string("x + y", variable='z')
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_cancels_terms():
    """Tests derivative where terms should cancel."""
    result = get_derived_string("x - x")
    expected_options = {"0"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_triple_nested_function():
    """Tests triple nested function: ln(sin(cos(x)))'."""
    result = get_derived_string("ln(sin(cos(x)))")
    expected_options = {"-cos(cos(x)) * sin(x) / sin(cos(x))", "(-sin(x) * cos(cos(x))) / sin(cos(x))"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_quadruple_composition():
    """Tests four-level composition: exp(ln(sin(cos(x))))'."""
    result = get_derived_string("exp(ln(sin(cos(x))))")
    expected_options = {"-cos(cos(x)) * sin(x)", "-sin(x) * cos(cos(x))"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_derivative_of_nested_fraction():
    """Tests derivative of nested fraction: (1/(1/x))'."""
    result = get_derived_string("1 / (1 / x)")
    expected_options = {"1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_long_sum():
    """Tests derivative of long sum of terms."""
    result = get_derived_string("x + x^2 + x^3 + x^4 + x^5")
    expected_options = {"5x^4 + 4x^3 + 3x^2 + 2x + 1"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_long_product():
    """Tests derivative of multiple products."""
    result = get_derived_string("x * x * x")
    expected_options = {"3x ^ 2", "3 * x ^ 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"


def test_alternating_operations():
    """Tests derivative with alternating operations."""
    result = get_derived_string("x^2 - 2*x + 1")
    expected_options = {"2x - 2"}
    assert result in expected_options, f"Expected one of {expected_options}, got: {result}"
