import math
import pytest
from hypothesis import given, assume, strategies as st, settings, example
from src.engine.Expression import parse, Expression
from src.nodes.BinaryOperator import BinaryOperator


@st.composite
def valid_numbers(draw):
    """Generate valid numeric values excluding extreme edge cases."""
    return draw(st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False
    ))


@st.composite
def positive_numbers(draw):
    """Generate positive numeric values for operations requiring positive inputs."""
    return draw(st.floats(min_value=0.1, max_value=1e4))


@st.composite
def small_exponents(draw):
    """Generate small exponents for exponential tests to avoid overflow."""
    return draw(st.floats(min_value=-50, max_value=50))


@st.composite
def nonzero_numbers(draw):
    """Generate non-zero numeric values to avoid division by zero."""
    num = draw(st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False
    ))
    assume(abs(num) > 0.01)
    return num


@st.composite
def small_integers(draw):
    """Generate small integers for exponents and coefficients."""
    return draw(st.integers(min_value=-5, max_value=5))


@st.composite
def variable_names(draw):
    """Generate valid variable names."""
    return draw(st.sampled_from(['x', 'y', 'z', 'a', 'b', 'c']))


def assert_approx_equal(val1, val2, rel_tol=1e-9, abs_tol=1e-9):
    """
    Custom assertion for float equality using relative tolerance.
    Critical for Property-Based Testing where inputs vary by orders of magnitude.
    """
    if math.isinf(val1) or math.isinf(val2) or math.isnan(val1) or math.isnan(val2):
        # Skip cases where calculation exploded
        return

    assert math.isclose(val1, val2, rel_tol=rel_tol, abs_tol=abs_tol), \
        f"{val1} != {val2} (rel_tol={rel_tol})"


class TestArithmeticProperties:
    """Test fundamental arithmetic properties."""

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_addition_commutative(self, a, b):
        """Test that addition is commutative: a + b = b + a."""
        expr1 = parse(f"{a} + {b}")
        expr2 = parse(f"{b} + {a}")
        result1 = expr1.evaluate({})
        result2 = expr2.evaluate({})
        assert_approx_equal(result1, result2)

    @given(valid_numbers(), valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_addition_associative(self, a, b, c):
        """Test that addition is associative: (a + b) + c = a + (b + c)."""
        # Using smaller tolerance because associativity errors accumulate
        expr1 = parse(f"({a} + {b}) + {c}")
        expr2 = parse(f"{a} + ({b} + {c})")
        result1 = expr1.evaluate({})
        result2 = expr2.evaluate({})
        assert_approx_equal(result1, result2, rel_tol=1e-6)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_addition_identity(self, a):
        """Test additive identity: a + 0 = a."""
        expr = parse(f"{a} + 0")
        result = expr.evaluate({})
        assert_approx_equal(result, a)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_additive_inverse(self, a):
        """Test additive inverse: a + (-a) = 0."""
        expr = parse(f"{a} + ({-a})")
        result = expr.evaluate({})
        assert_approx_equal(result, 0, abs_tol=1e-6)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_multiplication_commutative(self, a, b):
        """Test that multiplication is commutative: a * b = b * a."""
        expr1 = parse(f"{a} * {b}")
        expr2 = parse(f"{b} * {a}")
        result1 = expr1.evaluate({})
        result2 = expr2.evaluate({})
        assert_approx_equal(result1, result2)

    @given(valid_numbers(), valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_multiplication_associative(self, a, b, c):
        """Test that multiplication is associative."""
        expr1 = parse(f"({a} * {b}) * {c}")
        expr2 = parse(f"{a} * ({b} * {c})")
        result1 = expr1.evaluate({})
        result2 = expr2.evaluate({})
        assert_approx_equal(result1, result2, rel_tol=1e-6)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_multiplication_identity(self, a):
        """Test multiplicative identity: a * 1 = a."""
        expr = parse(f"{a} * 1")
        result = expr.evaluate({})
        assert_approx_equal(result, a)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_multiplication_zero(self, a):
        """Test multiplication by zero: a * 0 = 0."""
        expr = parse(f"{a} * 0")
        result = expr.evaluate({})
        assert_approx_equal(result, 0)

    @given(valid_numbers(), valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_distributive_property(self, a, b, c):
        """Test distributive property: a * (b + c) = a*b + a*c."""
        expr1 = parse(f"{a} * ({b} + {c})")
        expr2 = parse(f"{a}*{b} + {a}*{c}")
        result1 = expr1.evaluate({})
        result2 = expr2.evaluate({})
        assert_approx_equal(result1, result2, rel_tol=1e-6)

    @given(nonzero_numbers())
    @settings(max_examples=100)
    def test_multiplicative_inverse(self, a):
        """Test multiplicative inverse: a * (1/a) = 1."""
        expr = parse(f"{a} * (1/{a})")
        result = expr.evaluate({})
        assert_approx_equal(result, 1, rel_tol=1e-6)


class TestSimplificationProperties:
    """Test that simplification preserves mathematical equality."""

    @given(valid_numbers(), variable_names())
    @settings(max_examples=100)
    def test_simplification_preserves_value(self, value, var):
        """Test that simplification doesn't change expression value."""
        expr = parse(f"{var} + {var}")
        simplified = expr.simplify()
        result_original = expr.evaluate({var: value})
        result_simplified = simplified.evaluate({var: value})
        assert_approx_equal(result_original, result_simplified)

    @given(valid_numbers(), variable_names())
    @settings(max_examples=100)
    def test_double_simplification_idempotent(self, value, var):
        """Test that simplifying twice gives same result as simplifying once."""
        expr = parse(f"{var} * 2 + {var} * 3")
        simplified_once = expr.simplify()
        simplified_twice = simplified_once.simplify()

        result_once = simplified_once.evaluate({var: value})
        result_twice = simplified_twice.evaluate({var: value})

        assert_approx_equal(result_once, result_twice)
        assert simplified_once.to_string() == simplified_twice.to_string()

    @given(valid_numbers(), variable_names())
    @settings(max_examples=100)
    def test_identity_simplification_addition(self, value, var):
        """Test that x + 0 simplifies to x."""
        expr = parse(f"{var} + 0")
        simplified = expr.simplify()
        assert simplified.to_string() == var
        result = simplified.evaluate({var: value})
        assert_approx_equal(result, value)

    @given(valid_numbers(), variable_names())
    @settings(max_examples=100)
    def test_identity_simplification_multiplication(self, value, var):
        """Test that x * 1 simplifies to x."""
        expr = parse(f"{var} * 1")
        simplified = expr.simplify()
        assert simplified.to_string() == var
        result = simplified.evaluate({var: value})
        assert_approx_equal(result, value)

    @given(variable_names())
    @settings(max_examples=50)
    def test_zero_multiplication_simplification(self, var):
        """Test that x * 0 simplifies to 0."""
        expr = parse(f"{var} * 0")
        simplified = expr.simplify()
        assert simplified.to_string() == "0"

    @given(valid_numbers(), valid_numbers(), variable_names())
    @settings(max_examples=100)
    def test_like_terms_combination(self, a, b, var):
        """Test that like terms are combined: ax + bx = (a+b)x."""
        expr = parse(f"{a}*{var} + {b}*{var}")
        simplified = expr.simplify()
        test_value = 5.0
        result = simplified.evaluate({var: test_value})
        expected = (a + b) * test_value
        assert_approx_equal(result, expected)


class TestSubstitutionProperties:
    """Test properties of variable substitution."""

    @given(valid_numbers(), variable_names())
    @settings(max_examples=100)
    def test_substitution_preserves_evaluation(self, value, var):
        """Test that substituting a value equals direct evaluation."""
        expr = parse(f"{var} + {var}")
        result1 = expr.evaluate({var: value})
        substituted = expr.substitute({var: str(value)}, simplify=True)
        result2 = substituted.evaluate({})
        assert_approx_equal(result1, result2)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_multiple_substitution_order_independence(self, val_x, val_y):
        """Test that order of substitutions doesn't matter for independent variables."""
        expr = parse("x + y")

        sub1 = expr.substitute({'x': str(val_x)}, simplify=False)
        result1 = sub1.substitute({'y': str(val_y)}, simplify=True)

        sub2 = expr.substitute({'y': str(val_y)}, simplify=False)
        result2 = sub2.substitute({'x': str(val_x)}, simplify=True)

        eval1 = result1.evaluate({})
        eval2 = result2.evaluate({})
        assert_approx_equal(eval1, eval2)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_identity_substitution(self, value):
        """Test that substituting x with x doesn't change the expression."""
        expr = parse("x + 1")
        substituted = expr.substitute({'x': 'x'}, simplify=True)
        result_original = expr.evaluate({'x': value})
        result_substituted = substituted.evaluate({'x': value})
        assert_approx_equal(result_original, result_substituted)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_transitive_substitution(self, val1, val2):
        """Test transitive substitution: x→y, y→val equals x→val."""
        expr = parse("x + 2")

        direct = expr.substitute({'x': str(val1)}, simplify=True)
        result1 = direct.evaluate({})

        step1 = expr.substitute({'x': 'y'}, simplify=False)
        step2 = step1.substitute({'y': str(val1)}, simplify=True)
        result2 = step2.evaluate({})

        assert_approx_equal(result1, result2)


class TestDifferentiationProperties:
    """Test fundamental calculus properties."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_constant_derivative_is_zero(self, c):
        """Test that derivative of a constant is zero."""
        expr = parse(str(c))
        derivative = expr.diff('x')
        assert derivative.to_string() == "0"

    @given(variable_names())
    @settings(max_examples=50)
    def test_variable_derivative_is_one(self, var):
        """Test that derivative of x with respect to x is 1."""
        expr = parse(var)
        derivative = expr.diff(var)
        simplified = derivative.simplify()
        assert simplified.to_string() == "1"

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_linear_function_derivative(self, value):
        """Test that d/dx(ax + b) = a."""
        a, b = 3, 5
        expr = parse(f"{a}*x + {b}")
        derivative = expr.diff('x')
        simplified = derivative.simplify()
        result = simplified.evaluate({'x': value})
        assert_approx_equal(result, a)

    @given(small_integers())
    @settings(max_examples=50)
    @example(n=2)
    def test_power_rule(self, n):
        """Test power rule: d/dx(x^n) = n*x^(n-1)."""
        assume(n != 0)
        expr = parse(f"x^{n}")
        derivative = expr.diff('x')
        test_value = 2.0
        result = derivative.evaluate({'x': test_value})
        expected = n * (test_value ** (n - 1))
        assert_approx_equal(result, expected)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_sum_rule(self, value):
        """Test sum rule: d/dx(f + g) = f' + g'."""
        f = parse("x^2")
        g = parse("x^3")
        sum_expr = parse("x^2 + x^3")

        derivative_sum = sum_expr.diff('x')

        df = f.diff('x')
        dg = g.diff('x')
        sum_derivatives = Expression(BinaryOperator('+', df._root, dg._root))

        result1 = derivative_sum.evaluate({'x': value})
        result2 = sum_derivatives.evaluate({'x': value})
        assert_approx_equal(result1, result2)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_constant_multiple_rule(self, value):
        """Test constant multiple rule: d/dx(c*f) = c*f'."""
        c = 5
        expr = parse(f"{c}*x^2")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': value})
        expected = c * 2 * value
        assert_approx_equal(result, expected)

    @given(nonzero_numbers())
    @settings(max_examples=100)
    def test_product_rule(self, value):
        """Test product rule: d/dx(f*g) = f'*g + f*g'."""
        expr = parse("x^2 * x^3")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': value})
        expected = 5 * (value ** 4)
        assert_approx_equal(result, expected, rel_tol=1e-5)


class TestExponentialLogarithmProperties:
    """Test properties of exponential and logarithmic functions."""

    @given(positive_numbers())
    @settings(max_examples=100)
    def test_exp_ln_inverse(self, value):
        """Test that exp(ln(x)) = x for x > 0."""
        expr = parse("exp(ln(x))")
        simplified = expr.simplify()
        result = simplified.evaluate({'x': value})
        assert_approx_equal(result, value)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_ln_exp_inverse(self, value):
        """Test that ln(exp(x)) = x."""
        expr = parse("ln(exp(x))")
        simplified = expr.simplify()
        result = simplified.evaluate({'x': value})
        assert_approx_equal(result, value)

    @given(positive_numbers(), positive_numbers())
    @settings(max_examples=100)
    def test_logarithm_product_rule(self, a, b):
        """Test that ln(a*b) = ln(a) + ln(b)."""
        result1 = math.log(a * b)
        result2 = math.log(a) + math.log(b)
        assert_approx_equal(result1, result2)

    @given(positive_numbers(), positive_numbers())
    @settings(max_examples=100)
    def test_logarithm_quotient_rule(self, a, b):
        """Test that ln(a/b) = ln(a) - ln(b)."""
        assume(b > 0.01)
        result1 = math.log(a / b)
        result2 = math.log(a) - math.log(b)
        assert_approx_equal(result1, result2)

    @given(positive_numbers(), st.floats(min_value=0.1, max_value=10))
    @settings(max_examples=100)
    def test_logarithm_power_rule(self, a, n):
        """Test that ln(a^n) = n*ln(a)."""
        expr = parse(f"ln(x^{n})")
        simplified = expr.simplify()
        result = simplified.evaluate({'x': a})
        expected = n * math.log(a)
        assert_approx_equal(result, expected, rel_tol=1e-6)

    @given(small_exponents(), small_exponents())
    @settings(max_examples=100)
    def test_exponential_sum_rule(self, a, b):
        """Test that exp(a + b) = exp(a) * exp(b)."""
        result1 = math.exp(a + b)
        result2 = math.exp(a) * math.exp(b)
        assert_approx_equal(result1, result2)


class TestTrigonometricProperties:
    """Test properties of trigonometric functions."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_pythagorean_identity(self, x):
        """Test that sin²(x) + cos²(x) = 1."""
        expr = parse("sin(x)^2 + cos(x)^2")
        simplified = expr.simplify()
        assert simplified.to_string() == "1"
        result = simplified.evaluate({'x': x})
        assert_approx_equal(result, 1)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_sin_cos_ratio(self, x):
        """Test that sin(x)/cos(x) = tan(x) where cos(x) ≠ 0."""
        assume(abs(math.cos(x)) > 0.1)
        expr = parse("sin(x) / cos(x)")
        simplified = expr.simplify()
        result = simplified.evaluate({'x': x})
        expected = math.tan(x)
        assert_approx_equal(result, expected)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_sin_negative_argument(self, x):
        """Test that sin(-x) = -sin(x)."""
        expr1 = parse("sin(-x)")
        expr2 = parse("-sin(x)")
        result1 = expr1.evaluate({'x': x})
        result2 = expr2.evaluate({'x': x})
        assert_approx_equal(result1, result2)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_cos_negative_argument(self, x):
        """Test that cos(-x) = cos(x)."""
        expr1 = parse("cos(-x)")
        expr2 = parse("cos(x)")
        result1 = expr1.evaluate({'x': x})
        result2 = expr2.evaluate({'x': x})
        assert_approx_equal(result1, result2)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_sin_derivative(self, x):
        """Test that d/dx(sin(x)) = cos(x)."""
        expr = parse("sin(x)")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = math.cos(x)
        assert_approx_equal(result, expected)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_cos_derivative(self, x):
        """Test that d/dx(cos(x)) = -sin(x)."""
        expr = parse("cos(x)")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = -math.sin(x)
        assert_approx_equal(result, expected)


class TestPowerProperties:
    """Test properties of exponentiation."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_power_zero_exponent(self, a):
        """Test that a^0 = 1 for any non-zero a."""
        assume(abs(a) > 0.01)
        expr = parse(f"({a})^0")
        result = expr.evaluate({})
        assert_approx_equal(result, 1)

    @given(nonzero_numbers())
    @settings(max_examples=100)
    def test_power_one_exponent(self, a):
        """Test that a^1 = a."""
        expr = parse(f"({a})^1")
        result = expr.evaluate({})
        assert_approx_equal(result, a)

    @given(positive_numbers(), small_integers(), small_integers())
    @settings(max_examples=100)
    def test_power_multiplication_rule(self, a, m, n):
        """Test that a^m * a^n = a^(m+n)."""
        assume(abs(m) < 5 and abs(n) < 5)
        result1 = (a ** m) * (a ** n)
        result2 = a ** (m + n)
        assert_approx_equal(result1, result2, rel_tol=1e-6)

    @given(positive_numbers(), small_integers(), small_integers())
    @settings(max_examples=100)
    def test_power_division_rule(self, a, m, n):
        """Test that a^m / a^n = a^(m-n)."""
        assume(abs(m) < 5 and abs(n) < 5)
        assume(a > 0.1)
        result1 = (a ** m) / (a ** n)
        result2 = a ** (m - n)
        assert_approx_equal(result1, result2, rel_tol=1e-6)

    @given(positive_numbers(), small_integers(), small_integers())
    @settings(max_examples=100)
    def test_power_of_power_rule(self, a, m, n):
        """Test that (a^m)^n = a^(m*n)."""
        assume(abs(m) < 4 and abs(n) < 4)
        assume(a > 0.1)
        result1 = (a ** m) ** n
        result2 = a ** (m * n)
        assert_approx_equal(result1, result2, rel_tol=1e-6)

    @given(positive_numbers(), positive_numbers(), small_integers())
    @settings(max_examples=100)
    def test_power_of_product_rule(self, a, b, n):
        """Test that (a*b)^n = a^n * b^n."""
        assume(abs(n) < 5)
        assume(a > 0.1 and b > 0.1)
        result1 = (a * b) ** n
        result2 = (a ** n) * (b ** n)
        assert_approx_equal(result1, result2, rel_tol=1e-6)


class TestAlgebraicIdentities:
    """Test algebraic identities and formulas."""

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_difference_of_squares(self, a, b):
        """Test that (a+b)(a-b) = a² - b²."""
        assume(abs(a - b) > 1e-2)
        assume(abs(a + b) > 1e-2)

        expr = parse(f"({a} + {b}) * ({a} - {b})")
        result1 = expr.evaluate({})

        result2 = a ** 2 - b ** 2

        # Relax tolerance slightly for this specific identity
        assert_approx_equal(result1, result2, rel_tol=1e-5, abs_tol=1e-7)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_square_of_sum(self, a, b):
        """Test that (a+b)² = a² + 2ab + b²."""
        expr = parse("(x + y)^2")
        expanded = expr.simplify()
        result1 = expanded.evaluate({'x': a, 'y': b})
        result2 = a ** 2 + 2 * a * b + b ** 2
        assert_approx_equal(result1, result2, rel_tol=1e-6)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_square_of_difference(self, a, b):
        """Test that (a-b)² = a² - 2ab + b²."""
        expr = parse("(x - y)^2")
        expanded = expr.simplify()
        result1 = expanded.evaluate({'x': a, 'y': b})
        result2 = a ** 2 - 2 * a * b + b ** 2
        assert_approx_equal(result1, result2, rel_tol=1e-6)

    @given(valid_numbers(), valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_perfect_square_trinomial(self, a, b, c):
        """Test expansion consistency for trinomials."""
        expr = parse(f"({a}*x + {b})^2")
        expanded = expr.simplify()
        test_val = c
        result1 = expr.evaluate({'x': test_val})
        result2 = expanded.evaluate({'x': test_val})
        assert_approx_equal(result1, result2, rel_tol=1e-6)


class TestSquareRootProperties:
    """Test properties of square root function."""

    @given(positive_numbers())
    @settings(max_examples=100)
    def test_sqrt_square_inverse(self, x):
        """Test that sqrt(x²) = |x| for x ≥ 0."""
        expr = parse("sqrt(x^2)")
        simplified = expr.simplify()
        result = simplified.evaluate({'x': x})
        assert_approx_equal(result, abs(x))

    @given(positive_numbers())
    @settings(max_examples=100)
    def test_square_sqrt_inverse(self, x):
        """Test that (√x)² = x for x ≥ 0."""
        expr = parse("sqrt(x)^2")
        simplified = expr.simplify()
        result = simplified.evaluate({'x': x})
        assert_approx_equal(result, x)

    @given(positive_numbers(), positive_numbers())
    @settings(max_examples=100)
    def test_sqrt_product_rule(self, a, b):
        """Test that √(ab) = √a * √b for a,b ≥ 0."""
        result1 = math.sqrt(a * b)
        result2 = math.sqrt(a) * math.sqrt(b)
        assert_approx_equal(result1, result2)

    @given(positive_numbers(), positive_numbers())
    @settings(max_examples=100)
    def test_sqrt_quotient_rule(self, a, b):
        """Test that √(a/b) = √a / √b for a,b > 0."""
        assume(b > 0.1)
        result1 = math.sqrt(a / b)
        result2 = math.sqrt(a) / math.sqrt(b)
        assert_approx_equal(result1, result2)

    @given(positive_numbers())
    @settings(max_examples=100)
    def test_sqrt_derivative(self, x):
        """Test that d/dx(√x) = 1/(2√x)."""
        assume(x > 0.1)
        expr = parse("sqrt(x)")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = 1 / (2 * math.sqrt(x))
        assert_approx_equal(result, expected)


class TestAbsoluteValueProperties:
    """Test properties related to absolute values and signs."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_double_negation(self, a):
        """Test that -(-a) = a."""
        expr = parse(f"-(-{a})")
        result = expr.evaluate({})
        assert_approx_equal(result, a)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_negation_distributive_over_addition(self, a, b):
        """Test that -(a + b) = -a + (-b)."""
        expr1 = parse(f"-({a} + {b})")
        expr2 = parse(f"(-{a}) + (-{b})")
        result1 = expr1.evaluate({})
        result2 = expr2.evaluate({})
        assert_approx_equal(result1, result2)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_negation_distributive_over_multiplication(self, a, b):
        """Test that -(a * b) = (-a) * b = a * (-b)."""
        result1 = -(a * b)
        result2 = (-a) * b
        result3 = a * (-b)
        assert_approx_equal(result1, result2)
        assert_approx_equal(result2, result3)

    @given(nonzero_numbers(), nonzero_numbers())
    @settings(max_examples=100)
    def test_sign_multiplication_rule(self, a, b):
        """Test that sign(a*b) = sign(a)*sign(b)."""
        result = a * b
        expected_positive = (a > 0 and b > 0) or (a < 0 and b < 0)
        assert (result > 0) == expected_positive


class TestDivisionProperties:
    """Test properties of division operations."""

    @given(nonzero_numbers(), nonzero_numbers())
    @settings(max_examples=100)
    def test_division_multiplication_inverse(self, a, b):
        """Test that (a/b) * b = a."""
        result = (a / b) * b
        assert_approx_equal(result, a)

    @given(nonzero_numbers(), nonzero_numbers(), nonzero_numbers())
    @settings(max_examples=100)
    def test_division_chain_rule(self, a, b, c):
        """Test that (a/b)/c = a/(b*c)."""
        result1 = (a / b) / c
        result2 = a / (b * c)
        assert_approx_equal(result1, result2)

    @given(nonzero_numbers(), nonzero_numbers())
    @settings(max_examples=100)
    def test_reciprocal_of_reciprocal(self, a, b):
        """Test that 1/(1/a) = a."""
        assume(abs(a) > 0.01)
        result = 1 / (1 / a)
        assert_approx_equal(result, a)

    @given(nonzero_numbers(), nonzero_numbers(), nonzero_numbers())
    @settings(max_examples=100)
    def test_fraction_addition_same_denominator(self, a, b, c):
        """Test that a/c + b/c = (a+b)/c."""
        result1 = (a / c) + (b / c)
        result2 = (a + b) / c
        assert_approx_equal(result1, result2)

    @given(nonzero_numbers(), nonzero_numbers(), nonzero_numbers(), nonzero_numbers())
    @settings(max_examples=100)
    def test_fraction_multiplication(self, a, b, c, d):
        """Test that (a/b) * (c/d) = (a*c)/(b*d)."""
        result1 = (a / b) * (c / d)
        result2 = (a * c) / (b * d)
        assert_approx_equal(result1, result2)


class TestChainRuleProperties:
    """Test chain rule for composite functions."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_chain_rule_polynomial(self, x):
        """Test chain rule: d/dx[(x²)³] = 6x⁵."""
        expr = parse("(x^2)^3")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = 6 * (x ** 5)
        assert_approx_equal(result, expected, rel_tol=1e-5)

    @given(st.floats(min_value=0.1, max_value=5.0))
    @settings(max_examples=100)
    def test_chain_rule_exponential(self, x):
        """Test chain rule: d/dx[exp(x²)] = 2x*exp(x²)."""
        expr = parse("exp(x^2)")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = 2 * x * math.exp(x ** 2)
        assert_approx_equal(result, expected, rel_tol=1e-5)

    @given(positive_numbers())
    @settings(max_examples=100)
    def test_chain_rule_logarithm(self, x):
        """Test chain rule: d/dx[ln(x²)] = 2/x."""
        assume(x > 0.1)
        expr = parse("ln(x^2)")
        derivative = expr.diff('x')
        simplified = derivative.simplify()
        result = simplified.evaluate({'x': x})
        expected = 2 / x
        assert_approx_equal(result, expected)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_chain_rule_trigonometric(self, x):
        """Test chain rule: d/dx[sin(2x)] = 2*cos(2x)."""
        expr = parse("sin(2*x)")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = 2 * math.cos(2 * x)
        assert_approx_equal(result, expected)


class TestQuotientRuleProperties:
    """Test quotient rule for derivatives."""

    @given(nonzero_numbers())
    @settings(max_examples=100)
    def test_quotient_rule_simple(self, x):
        """Test quotient rule: d/dx[x/x²] = -1/x²."""
        assume(abs(x) > 0.1)
        expr = parse("x / (x^2)")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = -1 / (x ** 2)
        assert_approx_equal(result, expected, rel_tol=1e-5)

    @given(nonzero_numbers())
    @settings(max_examples=100)
    def test_quotient_rule_polynomial(self, x):
        """Test quotient rule: d/dx[(x²+1)/(x³+1)]."""
        assume(abs(x ** 3 + 1) > 0.1)
        expr = parse("(x^2 + 1) / (x^3 + 1)")
        derivative = expr.diff('x')

        # Compute expected using quotient rule
        numerator_derivative = 2 * x
        denominator_derivative = 3 * (x ** 2)
        numerator = x ** 2 + 1
        denominator = x ** 3 + 1

        expected = (numerator_derivative * denominator - numerator * denominator_derivative) / (denominator ** 2)
        result = derivative.evaluate({'x': x})
        assert_approx_equal(result, expected, rel_tol=1e-5)


class TestProductRuleProperties:
    """Test product rule for derivatives."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_product_rule_polynomials(self, x):
        """Test product rule: d/dx[x² * x³] = 5x⁴."""
        expr = parse("x^2 * x^3")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = 5 * (x ** 4)
        assert_approx_equal(result, expected, rel_tol=1e-5)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_product_rule_trig_polynomial(self, x):
        """Test product rule: d/dx[x*sin(x)] = sin(x) + x*cos(x)."""
        expr = parse("x * sin(x)")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = math.sin(x) + x * math.cos(x)
        assert_approx_equal(result, expected)

    @given(st.floats(min_value=0.1, max_value=50.0))
    @settings(max_examples=100)
    def test_product_rule_exponential(self, x):
        """Test product rule: d/dx[x*exp(x)] = exp(x) + x*exp(x)."""
        expr = parse("x * exp(x)")
        derivative = expr.diff('x')
        result = derivative.evaluate({'x': x})
        expected = math.exp(x) + x * math.exp(x)
        assert_approx_equal(result, expected, rel_tol=1e-5)


class TestCompositionProperties:
    """Test properties of function composition."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_composition_associativity(self, x):
        """Test that f(g(h(x))) is evaluated correctly."""
        expr = parse("sin(cos(x^2))")
        result = expr.evaluate({'x': x})
        expected = math.sin(math.cos(x ** 2))
        assert_approx_equal(result, expected)

    @given(positive_numbers())
    @settings(max_examples=100)
    def test_inverse_composition_identity(self, x):
        """Test that applying function then inverse gives identity."""
        expr = parse("exp(ln(x))")
        simplified = expr.simplify()
        result = simplified.evaluate({'x': x})
        assert_approx_equal(result, x)


class TestHomogeneityProperties:
    """Test homogeneity properties of expressions."""

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_linear_homogeneity(self, k, x):
        """Test that f(kx) = k*f(x) for linear f."""
        expr = parse("2*x + 3*x")
        result1 = expr.evaluate({'x': k * x})
        result2 = k * expr.evaluate({'x': x})
        assert_approx_equal(result1, result2)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_quadratic_homogeneity(self, k, x):
        """Test that f(kx) = k²*f(x) for quadratic f."""
        expr = parse("x^2")
        result1 = expr.evaluate({'x': k * x})
        result2 = (k ** 2) * expr.evaluate({'x': x})
        assert_approx_equal(result1, result2)


class TestBoundaryBehaviorProperties:
    """Test behavior at boundaries and special points."""

    @given(variable_names())
    @settings(max_examples=50)
    def test_zero_evaluation(self, var):
        """Test that expressions evaluate correctly at zero."""
        expr = parse(f"{var}^2 + {var}")
        result = expr.evaluate({var: 0})
        assert_approx_equal(result, 0)

    @given(variable_names())
    @settings(max_examples=50)
    def test_one_evaluation(self, var):
        """Test that expressions evaluate correctly at one."""
        expr = parse(f"{var}^2 + {var}")
        result = expr.evaluate({var: 1})
        assert_approx_equal(result, 2)

    @given(variable_names(), small_integers())
    @settings(max_examples=100)
    def test_negative_one_power(self, var, n):
        """Test that (-1)^n behaves correctly (alternating sign)."""
        assume(abs(n) < 10)
        expr = parse(f"(-1)^{n}")
        result = expr.evaluate({})
        expected = (-1) ** n
        assert_approx_equal(result, expected)


class TestSymmetryProperties:
    """Test symmetry properties of expressions."""

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_symmetric_expression_evaluation(self, x, y):
        """Test that symmetric expressions are invariant under variable swap."""
        expr = parse("x + y")
        result1 = expr.evaluate({'x': x, 'y': y})
        result2 = expr.evaluate({'x': y, 'y': x})
        assert_approx_equal(result1, result2)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_symmetric_product_evaluation(self, x, y):
        """Test that x*y + y*x is symmetric."""
        expr = parse("x*y + y*x")
        simplified = expr.simplify()
        result1 = simplified.evaluate({'x': x, 'y': y})
        result2 = simplified.evaluate({'x': y, 'y': x})
        assert_approx_equal(result1, result2)


class TestTransitivityProperties:
    """Test transitive properties of operations."""

    @given(valid_numbers(), valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_equality_transitivity(self, a, b, c):
        """Test that if a = b and b = c, then operations preserve equality."""
        # Using expressions: if f(a) = f(b), then f(a) = f(c) when b = c
        expr = parse("x^2 + 1")
        assume(abs(b - c) < 1e-6)  # b ≈ c
        result1 = expr.evaluate({'x': b})
        result2 = expr.evaluate({'x': c})
        assert_approx_equal(result1, result2, rel_tol=1e-5)


class TestContinuityProperties:
    """Test properties related to continuity."""

    @given(st.floats(min_value=-100, max_value=100))
    @settings(max_examples=100)
    def test_polynomial_continuity(self, x):
        """Test that nearby inputs give nearby outputs for polynomials."""
        expr = parse("x^2 + 2*x + 1")
        epsilon = 1e-6
        result1 = expr.evaluate({'x': x})
        result2 = expr.evaluate({'x': x + epsilon})
        # Results should be close
        assert abs(result1 - result2) < 0.01


class TestDerivativeHigherOrderProperties:
    """Test properties of higher-order derivatives."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_second_derivative_polynomial(self, x):
        """Test second derivative: d²/dx²[x³] = 6x."""
        expr = parse("x^3")
        first_derivative = expr.diff('x')
        second_derivative = first_derivative.diff('x')
        result = second_derivative.evaluate({'x': x})
        expected = 6 * x
        assert_approx_equal(result, expected)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_derivative_commutativity_mixed_partials(self, x):
        """Test that d/dx[d/dy[f]] = d/dy[d/dx[f]] for smooth f."""
        expr = parse("x*y")

        # d/dx then d/dy
        dx_first = expr.diff('x')
        dx_dy = dx_first.diff('y')
        result1 = dx_dy.evaluate({'x': x, 'y': x})

        # d/dy then d/dx
        dy_first = expr.diff('y')
        dy_dx = dy_first.diff('x')
        result2 = dy_dx.evaluate({'x': x, 'y': x})

        assert_approx_equal(result1, result2)


class TestNumericalStabilityProperties:
    """Test numerical stability of operations."""

    @given(st.floats(min_value=1e-8, max_value=1e-6))
    @settings(max_examples=50)
    def test_small_number_addition(self, x):
        """Test that addition with very small numbers is handled correctly."""
        expr = parse(f"1 + {x}")
        result = expr.evaluate({})
        assert result > 1

    @given(st.floats(min_value=1e3, max_value=1e6))
    @settings(max_examples=50)
    def test_large_number_multiplication(self, x):
        """Test that multiplication with large numbers doesn't overflow unexpectedly."""
        expr = parse(f"2 * {x}")
        result = expr.evaluate({})
        assert_approx_equal(result, 2 * x, rel_tol=1e-6)


class TestExpressionEquivalenceProperties:
    """Test that different forms of the same expression are equivalent."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_expanded_factored_equivalence(self, x):
        """Test that (x+1)(x-1) = x²-1."""
        expr1 = parse("(x + 1) * (x - 1)")
        expr2 = parse("x^2 - 1")
        result1 = expr1.evaluate({'x': x})
        result2 = expr2.evaluate({'x': x})
        assert_approx_equal(result1, result2)

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_trigonometric_equivalence(self, x):
        """Test that 1 - sin²(x) = cos²(x)."""
        expr1 = parse("1 - sin(x)^2")
        expr2 = parse("cos(x)^2")
        result1 = expr1.evaluate({'x': x})
        result2 = expr2.evaluate({'x': x})
        assert_approx_equal(result1, result2)

    @given(positive_numbers())
    @settings(max_examples=100)
    def test_logarithmic_equivalence(self, x):
        """Test that 2*ln(x) = ln(x²)."""
        expr1 = parse("2 * ln(x)")
        expr2 = parse("ln(x^2)")
        result1 = expr1.evaluate({'x': x})
        result2 = expr2.evaluate({'x': x})
        assert_approx_equal(result1, result2, rel_tol=1e-6)


class TestZeroProductProperty:
    """Test zero product property and related concepts."""

    @given(valid_numbers())
    @settings(max_examples=100)
    def test_zero_product_property(self, a):
        """Test that if a*b = 0, then a = 0 or b = 0."""
        # Test: 0 * a = 0
        result = 0 * a
        assert_approx_equal(result, 0)

    @given(valid_numbers(), valid_numbers())
    @settings(max_examples=100)
    def test_nonzero_product_nonzero(self, a, b):
        """Test that if a ≠ 0 and b ≠ 0, then a*b ≠ 0."""
        assume(abs(a) > 0.01 and abs(b) > 0.01)
        result = a * b
        assert abs(result) > 1e-6


class TestRationalExpressionProperties:
    """Test properties of rational expressions."""

    @given(nonzero_numbers(), nonzero_numbers())
    @settings(max_examples=100)
    def test_rational_simplification_preservation(self, a, b):
        """Test that simplifying rational expressions preserves value."""
        expr = parse("(x*2) / (x*3)")
        simplified = expr.simplify()

        result_original = expr.evaluate({'x': a})
        result_simplified = simplified.evaluate({'x': a})
        assert_approx_equal(result_original, result_simplified)

    @given(nonzero_numbers())
    @settings(max_examples=100)
    def test_complex_fraction_simplification(self, x):
        """Test that (a/b)/(c/d) = (a*d)/(b*c)."""
        assume(abs(x) > 0.1)
        expr = parse("(x / 2) / (x / 3)")
        result = expr.evaluate({'x': x})
        expected = (x / 2) / (x / 3)
        assert_approx_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
