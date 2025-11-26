import math
import pytest
from src.engine.Expression import parse, Expression
from src.nodes.Literal import Literal
from src.nodes.Variable import Variable
from src.nodes.BinaryOperator import BinaryOperator
from src.nodes.UnaryOperator import UnaryOperator


class TestEndToEndFeatureFlow:
    """Tests complete workflows through multiple operations."""

    def test_parse_simplify_evaluate_flow(self):
        """Test parsing, simplifying, and evaluating a complex expression."""
        expr = parse("2*x + 3*x")
        simplified = expr.simplify()
        assert simplified.to_string() == "5x"

        result = simplified.evaluate({'x': 10})
        assert result == 50

    def test_parse_substitute_simplify_evaluate_flow(self):
        """Test full pipeline: parse -> substitute -> simplify -> evaluate."""
        expr = parse("x^2 + 2*x*y + y^2")

        # Substitute y with x
        substituted = expr.substitute({'y': 'x'}, simplify=False)
        assert 'y' not in substituted.to_string()

        # Simplify
        simplified = substituted.simplify()
        expected = "4x ^ 2"  # x^2 + 2x^2 + x^2
        assert simplified.to_string() == expected

        # Evaluate
        result = simplified.evaluate({'x': 3})
        assert result == 36

    def test_differentiate_simplify_evaluate_flow(self):
        """Test differentiation followed by simplification and evaluation."""
        expr = parse("x^3 + 2*x^2 + x")

        # Differentiate
        derivative = expr.diff('x')

        # Simplify
        simplified = derivative.simplify()

        # Evaluate at x=2
        result = simplified.evaluate({'x': 2})
        # f'(x) = 3x^2 + 4x + 1, at x=2: 3(4) + 4(2) + 1 = 21
        assert result == 21

    def test_trigonometric_identity_simplification_flow(self):
        """Test trigonometric expression simplification and evaluation."""
        expr = parse("sin(x)^2 + cos(x)^2")

        # Should simplify to 1
        simplified = expr.simplify()
        assert simplified.to_string() == "1"

        # Should evaluate to 1 for any x
        result = simplified.evaluate({'x': math.pi / 4})
        assert abs(result - 1.0) < 1e-10

    def test_complex_algebraic_manipulation_flow(self):
        """Test complex algebraic manipulation through multiple steps."""
        # Start with (x + 1)^2
        expr = parse("(x + 1)^2")

        # Expand
        expanded = expr.simplify()
        # Should be x^2 + 2x + 1

        # Differentiate
        derivative = expanded.diff('x')
        simplified_deriv = derivative.simplify()

        # Should be 2x + 2
        result = simplified_deriv.evaluate({'x': 5})
        assert result == 12  # 2*5 + 2

    def test_nested_function_composition_flow(self):
        """Test nested function compositions through evaluation."""
        expr = parse("exp(ln(x))")

        # Should simplify to x
        simplified = expr.simplify()
        assert simplified.to_string() == "x"

        # Evaluate
        result = simplified.evaluate({'x': 42})
        assert result == 42

    def test_fraction_simplification_and_evaluation(self):
        """Test fraction operations end-to-end."""
        expr = parse("(x^2 - 4) / (x - 2)")

        result = expr.evaluate({'x': 3})
        # (9 - 4) / (3 - 2) = 5 / 1 = 5
        assert result == 5

    def test_multiple_substitutions_with_simplification(self):
        """Test multiple variable substitutions in sequence."""
        expr = parse("x + y + z")

        # First substitution
        step1 = expr.substitute({'x': '2*a'}, simplify=True)

        # Second substitution
        step2 = step1.substitute({'y': '3*a'}, simplify=True)

        # Third substitution
        step3 = step2.substitute({'z': 'a'}, simplify=True)

        # Should be 6a
        assert '6a' in step3.to_string()

        result = step3.evaluate({'a': 10})
        assert result == 60

    def test_differentiation_chain_rule_flow(self):
        """Test chain rule differentiation with complex expressions."""
        expr = parse("sin(x^2)")

        derivative = expr.diff('x')

        # d/dx[sin(x^2)] = cos(x^2) * 2x
        result = derivative.evaluate({'x': 0})
        # cos(0) * 0 = 0
        assert abs(result) < 1e-10

    def test_expression_building_with_operators(self):
        """Test building expressions using Python operators."""
        x = parse("x")
        y = parse("y")

        # Build (x + y) * (x - y)
        expr = (x + y) * (x - y)

        # Should simplify to x^2 - y^2
        result = expr.evaluate({'x': 5, 'y': 3})
        assert result == 16  # 25 - 9


class TestNodeInteractions:
    """Tests interactions between different node types."""

    def test_literal_and_variable_interaction(self):
        """Test operations between literals and variables."""
        lit = Literal(5)
        var = Variable('x')

        # Create binary operation
        expr = Expression(BinaryOperator('+', lit, var))

        assert expr.to_string() in ("5 + x", "x + 5")

        result = expr.evaluate({'x': 10})
        assert result == 15

    def test_unary_and_binary_operator_nesting(self):
        """Test nested unary and binary operators."""
        # Create -(x + 5)
        var = Variable('x')
        lit = Literal(5)
        add = BinaryOperator('+', var, lit)
        neg = UnaryOperator('-', add)

        expr = Expression(neg)
        simplified = expr.simplify()

        result = simplified.evaluate({'x': 10})
        assert result == -15

    def test_multiple_binary_operators_precedence(self):
        """Test correct precedence handling with multiple operators."""
        expr = parse("2 + 3 * 4")

        # Should be 14, not 20
        result = expr.evaluate({})
        assert result == 14

    def test_power_and_multiplication_interaction(self):
        """Test interaction between power and multiplication operators."""
        expr = parse("2 * x^2 * 3")

        simplified = expr.simplify()
        result = simplified.evaluate({'x': 4})
        assert result == 96  # 2 * 16 * 3

    def test_nested_function_calls(self):
        """Test nested unary function applications."""
        expr = parse("sin(cos(x))")

        result = expr.evaluate({'x': 0})
        # sin(cos(0)) = sin(1)
        expected = math.sin(1)
        assert abs(result - expected) < 1e-10

    def test_division_and_multiplication_interaction(self):
        """Test interaction between division and multiplication."""
        expr = parse("x / 2 * 3")

        # Should be (x / 2) * 3 = 1.5x
        result = expr.evaluate({'x': 10})
        assert result == 15  # (10 / 2) * 3

    def test_subtraction_and_addition_interaction(self):
        """Test interaction between subtraction and addition."""
        expr = parse("x - 5 + 3")

        result = expr.evaluate({'x': 10})
        assert result == 8  # 10 - 5 + 3

    def test_complex_nested_structure(self):
        """Test deeply nested expression structures."""
        expr = parse("((x + 1) * (x - 1)) / (x + 2)")

        result = expr.evaluate({'x': 5})
        # ((5 + 1) * (5 - 1)) / (5 + 2) = (6 * 4) / 7 = 24/7
        assert abs(result - 24 / 7) < 1e-10

    def test_mixed_operators_with_functions(self):
        """Test mixing arithmetic operators with functions."""
        expr = parse("2 * sin(x) + 3 * cos(x)")

        result = expr.evaluate({'x': math.pi / 2})
        # 2 * sin(π/2) + 3 * cos(π/2) = 2 * 1 + 3 * 0 = 2
        assert abs(result - 2) < 1e-10

    def test_variable_substitution_in_nested_operations(self):
        """Test variable substitution in deeply nested structures."""
        expr = parse("(x + y) * (x - y) + z")

        substituted = expr.substitute({'x': '2*a', 'y': 'a', 'z': '5'}, simplify=True)

        # (2a + a)(2a - a) + 5 = 3a * a + 5 = 3a^2 + 5
        result = substituted.evaluate({'a': 3})
        assert result == 32  # 3 * 9 + 5


class TestEdgeCasesAndBoundaries:
    """Tests edge cases and boundary conditions in integration."""

    def test_empty_substitution_evaluation(self):
        """Test evaluation with no variable substitutions needed."""
        expr = parse("2 + 3")

        result = expr.evaluate({})
        assert result == 5

    def test_identity_operations_simplification(self):
        """Test simplification of identity operations."""
        test_cases = [
            ("x + 0", "x"),
            ("x * 1", "x"),
            ("x - 0", "x"),
            ("x / 1", "x"),
            ("x * 0", "0"),
        ]

        for expr_str, expected in test_cases:
            expr = parse(expr_str)
            simplified = expr.simplify()
            assert simplified.to_string() == expected

    def test_negative_number_handling(self):
        """Test correct handling of negative numbers throughout pipeline."""
        expr = parse("-5 + x")

        result = expr.evaluate({'x': 10})
        assert result == 5

    def test_floating_point_precision(self):
        """Test floating point precision in calculations."""
        expr = parse("0.1 + 0.2")

        result = expr.evaluate({})
        # Should handle floating point correctly
        assert abs(result - 0.3) < 1e-10

    def test_large_exponent_evaluation(self):
        """Test evaluation with large exponents."""
        expr = parse("2^10")

        result = expr.evaluate({})
        assert result == 1024

    def test_trigonometric_special_values(self):
        """Test trigonometric functions at special angle values."""
        test_cases = [
            ("sin(0)", 0),
            ("cos(0)", 1),
        ]

        for expr_str, expected in test_cases:
            expr = parse(expr_str)
            result = expr.evaluate({})
            assert abs(result - expected) < 1e-10

    def test_logarithm_special_values(self):
        """Test logarithmic functions with special values."""
        expr = parse("ln(1)")
        simplified = expr.simplify()

        assert simplified.to_string() == "0"

    def test_exponential_and_logarithm_inverse(self):
        """Test inverse relationship between exp and ln."""
        expr = parse("exp(ln(x))")
        simplified = expr.simplify()

        assert simplified.to_string() == "x"

        result = simplified.evaluate({'x': 100})
        assert abs(result - 100) < 1e-10

    def test_power_special_cases(self):
        """Test special power operation cases."""
        test_cases = [
            ("x^0", {'x': 5}, 1),
            ("x^1", {'x': 5}, 5),
            ("1^x", {'x': 100}, 1),
        ]

        for expr_str, variables, expected in test_cases:
            expr = parse(expr_str)
            simplified = expr.simplify()
            result = simplified.evaluate(variables)
            assert result == expected


class TestComplexWorkflows:
    """Tests complex real-world workflows."""

    def test_polynomial_evaluation_workflow(self):
        """Test complete polynomial workflow with multiple operations."""
        # Create polynomial: 3x^2 + 2x + 1
        expr = parse("3*x^2 + 2*x + 1")

        # Evaluate at multiple points
        points = [0, 1, 2, -1]
        expected = [1, 6, 17, 2]

        for point, exp in zip(points, expected):
            result = expr.evaluate({'x': point})
            assert result == exp

    def test_derivative_of_product_workflow(self):
        """Test product rule differentiation workflow."""
        expr = parse("x^2 * sin(x)")

        derivative = expr.diff('x')

        # d/dx[x^2 * sin(x)] = 2x*sin(x) + x^2*cos(x)
        result = derivative.evaluate({'x': math.pi})
        # At x=π: 2π*sin(π) + π^2*cos(π) = 0 + π^2*(-1) = -π^2
        expected = -math.pi ** 2
        assert abs(result - expected) < 1e-8

    def test_quotient_rule_differentiation_workflow(self):
        """Test quotient rule differentiation workflow."""
        expr = parse("x / (x + 1)")

        derivative = expr.diff('x')
        simplified = derivative.simplify()

        # d/dx[x/(x+1)] = [(x+1) - x] / (x+1)^2 = 1 / (x+1)^2
        result = simplified.evaluate({'x': 1})
        assert abs(result - 0.25) < 1e-10  # 1/4

    def test_chain_rule_with_composition_workflow(self):
        """Test chain rule with function composition."""
        expr = parse("(x^2 + 1)^3")

        derivative = expr.diff('x')

        # d/dx[(x^2+1)^3] = 3(x^2+1)^2 * 2x = 6x(x^2+1)^2
        result = derivative.evaluate({'x': 1})
        # 6*1*(1+1)^2 = 6*4 = 24
        assert result == 24

    def test_symbolic_to_numeric_conversion_workflow(self):
        """Test converting symbolic expressions to numeric values."""
        # Start with symbolic
        expr = parse("a*x^2 + b*x + c")

        # Substitute constants
        with_constants = expr.substitute({
            'a': '2',
            'b': '3',
            'c': '1'
        }, simplify=True)

        # Now evaluate with x
        result = with_constants.evaluate({'x': 5})
        assert result == 66  # 2*25 + 3*5 + 1

    def test_expression_transformation_pipeline(self):
        """Test complete expression transformation pipeline."""
        # Start with complex expression
        expr = parse("(x + 2)^2 - 4")

        # Expand
        expanded = expr.simplify()

        # Differentiate
        derivative = expanded.diff('x')

        # Simplify derivative
        simplified_deriv = derivative.simplify()

        # Evaluate
        result = simplified_deriv.evaluate({'x': 3})
        # d/dx[(x+2)^2 - 4] = 2(x+2) = 2x + 4, at x=3: 10
        assert result == 10

    def test_multivariable_expression_workflow(self):
        """Test workflow with multiple variables."""
        expr = parse("x*y + y*z + z*x")

        # Partial derivative with respect to x
        dx = expr.diff('x')
        simplified_dx = dx.simplify()

        # Should be y + z
        result = simplified_dx.evaluate({'y': 2, 'z': 3})
        assert result == 5

    def test_optimization_problem_workflow(self):
        """Test workflow similar to optimization problems."""
        # Area of rectangle: A = x * y with constraint x + y = 10
        area = parse("x * (10 - x)")

        # Expand
        expanded = area.simplify()

        # Find critical points: dA/dx = 0
        derivative = expanded.diff('x')

        # At critical point (x=5): derivative should be 0
        result = derivative.evaluate({'x': 5})
        assert abs(result) < 1e-10

    def test_physics_formula_workflow(self):
        """Test workflow with physics-like formulas."""
        # Kinetic energy: KE = 0.5 * m * v^2
        ke = parse("0.5 * m * v^2")

        # Calculate for specific values
        result = ke.evaluate({'m': 10, 'v': 5})
        assert result == 125  # 0.5 * 10 * 25

    def test_recursive_substitution_workflow(self):
        """Test recursive variable substitution."""
        expr = parse("x^2 + a")

        # First substitution
        step1 = expr.substitute({'a': 'b + 1'}, simplify=False)

        # Second substitution
        step2 = step1.substitute({'b': '2'}, simplify=True)

        # Should be x^2 + 3
        result = step2.evaluate({'x': 2})
        assert result == 7  # 4 + 3


class TestErrorHandlingIntegration:
    """Tests error handling across integrated components."""

    def test_unbound_variable_error_propagation(self):
        """Test that unbound variables are caught during evaluation."""
        expr = parse("x + y")

        with pytest.raises(ValueError, match="Unbound variable"):
            expr.evaluate({'x': 5})  # y is missing

    def test_domain_error_propagation(self):
        """Test that domain errors propagate correctly."""
        expr = parse("ln(x)")

        with pytest.raises(ValueError, match="Domain error"):
            expr.evaluate({'x': -1})

    def test_division_by_zero_propagation(self):
        """Test division by zero error propagation."""
        expr = parse("5 / x")

        with pytest.raises(ZeroDivisionError):
            expr.evaluate({'x': 0})

    def test_tan_asymptote_error(self):
        """Test tangent function at asymptote."""
        expr = parse("tan(x)")

        with pytest.raises(ValueError, match="Domain error"):
            expr.evaluate({'x': math.pi / 2})

    def test_sqrt_negative_error(self):
        """Test square root of negative number."""
        expr = parse("sqrt(x)")

        with pytest.raises(ValueError, match="Domain error"):
            expr.evaluate({'x': -4})

    def test_invalid_operation_in_pipeline(self):
        """Test that invalid operations are caught in pipeline."""
        expr = parse("x")

        # Cannot differentiate with respect to non-existent variable
        derivative = expr.diff('y')

        # Should be 0 (x is constant with respect to y)
        result = derivative.evaluate({'x': 5})
        assert result == 0


class TestPerformanceAndScalability:
    """Tests related to performance with complex expressions."""

    def test_deeply_nested_expression_evaluation(self):
        """Test evaluation of deeply nested expressions."""
        # Build ((((x + 1) + 1) + 1) + 1) + 1
        expr_str = "x"
        for _ in range(10):
            expr_str = f"({expr_str} + 1)"

        expr = parse(expr_str)
        result = expr.evaluate({'x': 0})
        assert result == 10

    def test_wide_expression_evaluation(self):
        """Test evaluation of expressions with many terms."""
        # Build x + x + x + ... (20 times)
        terms = " + ".join(["x"] * 20)
        expr = parse(terms)

        simplified = expr.simplify()
        result = simplified.evaluate({'x': 2})
        assert result == 40

    def test_multiple_simplification_passes(self):
        """Test that multiple simplifications don't break anything."""
        expr = parse("(x + 1) * (x + 1)")

        # Multiple simplification passes should be idempotent
        s1 = expr.simplify()
        s2 = s1.simplify()
        s3 = s2.simplify()

        assert s1.to_string() == s2.to_string()
        assert s2.to_string() == s3.to_string()

    def test_complex_mixed_operations(self):
        """Test complex expression with mixed operations."""
        expr = parse("sin(x^2) + cos(y^2) + exp(z) + ln(w) + sqrt(a*b)")

        result = expr.evaluate({
            'x': 1,
            'y': 1,
            'z': 0,
            'w': 1,
            'a': 4,
            'b': 9
        })

        expected = math.sin(1) + math.cos(1) + 1 + 0 + 6
        assert abs(result - expected) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
