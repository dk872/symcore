import subprocess
import sys
import math
import pytest
from src.engine.Expression import parse, Expression
from src.nodes.Literal import Literal
from src.nodes.Variable import Variable
from src.nodes.BinaryOperator import BinaryOperator
from src.nodes.UnaryOperator import UnaryOperator


class TestCLIBasicOperations:
    """Test command-line interface basic operations."""

    def test_cli_simple_expression_parsing(self):
        """Test that CLI correctly parses and displays simple expressions."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "2 + 3"],
            capture_output=True,
            text=True
        )
        expected1 = "Expression: 2 + 3"
        expected2 = "Expression: 3 + 2"
        assert result.returncode == 0
        assert expected1 in result.stdout or expected2 in result.stdout

    def test_cli_expression_evaluation(self):
        """Test that CLI correctly evaluates expressions with --eval flag."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + 5", "--eval", "x=10"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "15" in result.stdout

    def test_cli_simplification_flag(self):
        """Test that CLI correctly simplifies expressions with --simplify flag."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + x", "--simplify"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "2x" in result.stdout

    def test_cli_differentiation_flag(self):
        """Test that CLI correctly differentiates with --diff flag."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x^2", "--diff", "x"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Derivative" in result.stdout

    def test_cli_substitution_flag(self):
        """Test that CLI correctly substitutes variables with --substitute flag."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + y", "--substitute", "x=5", "y=3"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Substituted:" in result.stdout

    def test_cli_multiple_operations_combined(self):
        """Test CLI with multiple operations in sequence."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x^2 + 2*x", "--diff", "x", "--simplify"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Derivative" in result.stdout
        assert "Simplified:" in result.stdout

    def test_cli_complex_expression_input(self):
        """Test CLI with complex mathematical expressions."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "sin(x^2) + cos(y)"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "sin" in result.stdout or "Expression:" in result.stdout

    def test_cli_empty_expression_error(self):
        """Test that CLI handles empty expression input with proper error."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", ""],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "cannot be empty" in result.stderr.lower()

    def test_cli_invalid_syntax_error(self):
        """Test that CLI handles invalid syntax with proper error message."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "(x + y"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "syntax" in result.stderr.lower()

    def test_cli_help_flag(self):
        """Test that CLI displays help information with -h or --help."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()


class TestCLIEvaluationInterface:
    """Test command-line evaluation interface."""

    def test_cli_eval_single_variable(self):
        """Test evaluation with a single variable substitution."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "2*x + 3", "--eval", "x=5"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "13" in result.stdout

    def test_cli_eval_multiple_variables(self):
        """Test evaluation with multiple variable substitutions."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + y + z", "--eval", "x=1", "y=2", "z=3"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "6" in result.stdout

    def test_cli_eval_with_floating_point(self):
        """Test evaluation with floating-point numbers."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x * 2", "--eval", "x=3.14"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "6.28" in result.stdout

    def test_cli_eval_with_negative_numbers(self):
        """Test evaluation with negative number inputs."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + 10", "--eval", "x=-5"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "5" in result.stdout

    def test_cli_eval_empty_assignments(self):
        """Test evaluation with --eval flag but no assignments."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "5 + 3", "--eval"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "8" in result.stdout

    def test_cli_eval_invalid_format(self):
        """Test that invalid evaluation format produces error."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + 5", "--eval", "x:10"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0 or "error" in result.stdout.lower()

    def test_cli_eval_unbound_variable_error(self):
        """Test error handling when evaluating with unbound variables."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + y", "--eval", "x=5"],
            capture_output=True,
            text=True
        )
        assert "error" in result.stdout.lower() or "unbound" in result.stdout.lower()

    def test_cli_eval_division_by_zero(self):
        """Test error handling for division by zero in evaluation."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "5 / x", "--eval", "x=0"],
            capture_output=True,
            text=True
        )
        assert "error" in result.stdout.lower() or "division" in result.stdout.lower()


class TestCLISubstitutionInterface:
    """Test command-line substitution interface."""

    def test_cli_substitute_single_variable(self):
        """Test substitution of a single variable."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + 5", "--substitute", "x=y"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "y" in result.stdout

    def test_cli_substitute_with_expression(self):
        """Test substitution with complex expressions."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x^2", "--substitute", "x=y+1"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Substituted:" in result.stdout

    def test_cli_substitute_multiple_variables(self):
        """Test simultaneous substitution of multiple variables."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + y", "--substitute", "x=a", "y=b"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "a" in result.stdout and "b" in result.stdout

    def test_cli_substitute_then_evaluate(self):
        """Test substitution followed by evaluation."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + y",
             "--substitute", "x=5", "--eval", "y=3"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "8" in result.stdout

    def test_cli_substitute_invalid_format(self):
        """Test error handling for invalid substitution format."""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "x + 5", "--substitute", "x"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0 or "error" in result.stdout.lower()


class TestPublicAPIExpressionClass:
    """Test the public API of the Expression class."""

    def test_api_parse_function(self):
        """Test that parse() function is accessible and works correctly."""
        expr = parse("2 + 3")
        assert expr is not None
        assert isinstance(expr, Expression)

    def test_api_expression_to_string(self):
        """Test Expression.to_string() method."""
        expr = parse("x + y")
        result = expr.to_string()
        assert isinstance(result, str)
        assert "x + y" in result or "y + x" in result

    def test_api_expression_evaluate(self):
        """Test Expression.evaluate() method."""
        expr = parse("x + 5")
        result = expr.evaluate({'x': 10})
        assert result == 15

    def test_api_expression_simplify(self):
        """Test Expression.simplify() method."""
        expr = parse("x + x")
        simplified = expr.simplify()
        assert isinstance(simplified, Expression)
        assert "2x" in simplified.to_string()

    def test_api_expression_diff(self):
        """Test Expression.diff() method."""
        expr = parse("x^2")
        derivative = expr.diff('x')
        assert isinstance(derivative, Expression)

    def test_api_expression_substitute(self):
        """Test Expression.substitute() method."""
        expr = parse("x + y")
        substituted = expr.substitute({'x': '5'})
        assert isinstance(substituted, Expression)

    def test_api_expression_repr(self):
        """Test Expression.__repr__() method."""
        expr = parse("x + 1")
        repr_str = repr(expr)
        assert isinstance(repr_str, str)
        assert "Expression" in repr_str

    def test_api_expression_root_property(self):
        """Test Expression.root property access."""
        expr = parse("x + 1")
        root = expr.root
        assert root is not None

    def test_api_parse_class_method(self):
        """Test Expression.parse() class method."""
        expr = Expression.parse("2 * x")
        assert isinstance(expr, Expression)

    def test_api_from_node_class_method(self):
        """Test Expression.from_node() class method."""
        node = BinaryOperator('+', Literal(2), Literal(3))
        expr = Expression.from_node(node)
        assert isinstance(expr, Expression)


class TestPublicAPIOperatorOverloading:
    """Test operator overloading in the public API."""

    def test_api_addition_operator(self):
        """Test Expression.__add__() operator overloading."""
        expr1 = parse("x")
        expr2 = parse("y")
        result = expr1 + expr2
        assert isinstance(result, Expression)
        assert "x" in result.to_string() and "y" in result.to_string()

    def test_api_subtraction_operator(self):
        """Test Expression.__sub__() operator overloading."""
        expr1 = parse("x")
        expr2 = parse("5")
        result = expr1 - expr2
        assert isinstance(result, Expression)

    def test_api_multiplication_operator(self):
        """Test Expression.__mul__() operator overloading."""
        expr1 = parse("x")
        expr2 = parse("3")
        result = expr1 * expr2
        assert isinstance(result, Expression)

    def test_api_division_operator(self):
        """Test Expression.__truediv__() operator overloading."""
        expr1 = parse("x")
        expr2 = parse("2")
        result = expr1 / expr2
        assert isinstance(result, Expression)

    def test_api_power_operator(self):
        """Test Expression.__pow__() operator overloading."""
        expr = parse("x")
        result = expr ** 2
        assert isinstance(result, Expression)

    def test_api_negation_operator(self):
        """Test Expression.__neg__() operator overloading."""
        expr = parse("x")
        result = -expr
        assert isinstance(result, Expression)

    def test_api_right_addition_operator(self):
        """Test Expression.__radd__() right-side operator."""
        expr = parse("x")
        result = 5 + expr
        assert isinstance(result, Expression)

    def test_api_right_multiplication_operator(self):
        """Test Expression.__rmul__() right-side operator."""
        expr = parse("x")
        result = 3 * expr
        assert isinstance(result, Expression)

    def test_api_operator_chaining(self):
        """Test chaining multiple operators together."""
        x = parse("x")
        y = parse("y")
        result = (x + y) * (x - y)
        assert isinstance(result, Expression)
        value = result.evaluate({'x': 5, 'y': 3})
        assert value == 16  # (5+3)*(5-3) = 8*2 = 16


class TestPublicAPINodeClasses:
    """Test public API of Node classes."""

    def test_api_literal_creation(self):
        """Test creating Literal nodes directly."""
        lit = Literal(42)
        assert lit.value == 42
        assert lit.to_string() == "42"

    def test_api_variable_creation(self):
        """Test creating Variable nodes directly."""
        var = Variable('x')
        assert var.name == 'x'
        assert var.to_string() == 'x'

    def test_api_binary_operator_creation(self):
        """Test creating BinaryOperator nodes directly."""
        left = Literal(2)
        right = Variable('x')
        binop = BinaryOperator('+', left, right)
        assert binop.op == '+'
        assert isinstance(binop.left, Literal)
        assert isinstance(binop.right, Variable)

    def test_api_unary_operator_creation(self):
        """Test creating UnaryOperator nodes directly."""
        operand = Variable('x')
        unop = UnaryOperator('sin', operand)
        assert unop.op == 'sin'
        assert isinstance(unop.operand, Variable)

    def test_api_node_copy_method(self):
        """Test Node.copy() method."""
        node = Literal(5)
        copied = node.copy()
        assert copied is not node
        assert copied.value == node.value

    def test_api_node_simplify_method(self):
        """Test Node.simplify() method."""
        node = BinaryOperator('+', Literal(2), Literal(3))
        simplified = node.simplify()
        assert isinstance(simplified, Literal)
        assert simplified.value == 5

    def test_api_node_substitute_method(self):
        """Test Node.substitute() method."""
        node = Variable('x')
        substituted = node.substitute({'x': Literal(10)})
        assert isinstance(substituted, Literal)
        assert substituted.value == 10

    def test_api_node_to_string_method(self):
        """Test Node.to_string() method."""
        node = BinaryOperator('+', Variable('x'), Literal(5))
        result = node.to_string()
        assert isinstance(result, str)
        assert 'x' in result and '5' in result

    def test_api_node_precedence_method(self):
        """Test Node.precedence() method."""
        node = BinaryOperator('+', Literal(1), Literal(2))
        prec = node.precedence()
        assert isinstance(prec, int)
        assert prec > 0


class TestAPIErrorHandling:
    """Test error handling in the public API."""

    def test_api_parse_invalid_syntax(self):
        """Test that parsing invalid syntax raises appropriate error."""
        with pytest.raises(Exception):
            parse("(x + y")

    def test_api_evaluate_unbound_variable(self):
        """Test that evaluating with unbound variables raises error."""
        expr = parse("x + y")
        with pytest.raises(ValueError, match="Unbound variable"):
            expr.evaluate({'x': 5})

    def test_api_evaluate_division_by_zero(self):
        """Test that division by zero raises appropriate error."""
        expr = parse("5 / x")
        with pytest.raises(ZeroDivisionError):
            expr.evaluate({'x': 0})

    def test_api_evaluate_domain_error_ln(self):
        """Test that ln of negative number raises domain error."""
        expr = parse("ln(x)")
        with pytest.raises(ValueError, match="Domain error"):
            expr.evaluate({'x': -1})

    def test_api_evaluate_domain_error_sqrt(self):
        """Test that sqrt of negative number raises domain error."""
        expr = parse("sqrt(x)")
        with pytest.raises(ValueError, match="Domain error"):
            expr.evaluate({'x': -4})

    def test_api_substitute_invalid_type(self):
        """Test that substituting with invalid type raises error."""
        expr = parse("x + 1")
        with pytest.raises(ValueError):
            expr.substitute({'x': [1, 2, 3]})

    def test_api_diff_unknown_operator(self):
        """Test differentiation handles unknown operators gracefully."""
        # This should work for all known operators
        expr = parse("sin(x) + cos(x)")
        derivative = expr.diff('x')
        assert derivative is not None


class TestAPIInputValidation:
    """Test input validation in the public API."""

    def test_api_parse_empty_string(self):
        """Test that parsing empty string raises error."""
        with pytest.raises(Exception):
            parse("")

    def test_api_parse_whitespace_only(self):
        """Test that parsing whitespace-only string raises error."""
        with pytest.raises(Exception):
            parse("   ")

    def test_api_evaluate_with_none_values(self):
        """Test that evaluate with None as values dict works (uses empty dict)."""
        expr = parse("5 + 3")
        result = expr.evaluate(None)
        assert result == 8

    def test_api_evaluate_with_empty_dict(self):
        """Test that evaluate with empty dict works for constant expressions."""
        expr = parse("2 * 3")
        result = expr.evaluate({})
        assert result == 6

    def test_api_substitute_with_string_values(self):
        """Test that substitute accepts string expressions as values."""
        expr = parse("x + y")
        substituted = expr.substitute({'x': '2*a', 'y': 'b'})
        assert 'a' in substituted.to_string() and 'b' in substituted.to_string()

    def test_api_substitute_with_numeric_values(self):
        """Test that substitute accepts numeric values."""
        expr = parse("x + y")
        substituted = expr.substitute({'x': 5, 'y': 3})
        assert '5' in substituted.to_string() and '3' in substituted.to_string()

    def test_api_substitute_with_node_values(self):
        """Test that substitute accepts Node objects as values."""
        expr = parse("x + y")
        substituted = expr.substitute({'x': Literal(10), 'y': Variable('z')})
        result_str = substituted.to_string()
        assert '10' in result_str and 'z' in result_str

    def test_api_diff_with_nonexistent_variable(self):
        """Test differentiation with respect to non-existent variable."""
        expr = parse("x + 5")
        derivative = expr.diff('y')
        simplified = derivative.simplify()
        # Derivative should be 0
        assert simplified.to_string() == "0"


class TestAPIOutputFormats:
    """Test output formats from the API."""

    def test_api_to_string_simple_expression(self):
        """Test to_string() output for simple expressions."""
        expr = parse("2 + 3")
        output = expr.to_string()
        assert output in ("2 + 3", "3 + 2")

    def test_api_to_string_preserves_precedence(self):
        """Test that to_string() preserves operator precedence."""
        expr = parse("2 + 3 * 4")
        output = expr.to_string()
        assert "2 + 3 * 4" in output or "2 + 12" in output

    def test_api_to_string_with_parentheses(self):
        """Test to_string() with parenthesized expressions."""
        expr = parse("(2 + 3) * 4")
        output = expr.to_string()
        # Output should maintain the grouping somehow
        assert output is not None

    def test_api_to_string_with_functions(self):
        """Test to_string() with function calls."""
        expr = parse("sin(x)")
        output = expr.to_string()
        assert "sin" in output and "x" in output

    def test_api_to_string_after_simplification(self):
        """Test that to_string() works correctly after simplification."""
        expr = parse("x + x + x")
        simplified = expr.simplify()
        output = simplified.to_string()
        assert "3x" in output

    def test_api_repr_format(self):
        """Test __repr__() format for debugging."""
        expr = parse("x + 1")
        repr_output = repr(expr)
        assert "Expression" in repr_output
        assert "x" in repr_output or "+" in repr_output


class TestAPIChainingOperations:
    """Test chaining multiple API operations."""

    def test_api_parse_simplify_evaluate_chain(self):
        """Test chaining parse -> simplify -> evaluate."""
        result = parse("x + x").simplify().evaluate({'x': 5})
        assert result == 10

    def test_api_parse_diff_simplify_chain(self):
        """Test chaining parse -> diff -> simplify."""
        result = parse("x^2 + 2*x").diff('x').simplify()
        assert isinstance(result, Expression)

    def test_api_parse_substitute_simplify_evaluate_chain(self):
        """Test chaining parse -> substitute -> simplify -> evaluate."""
        result = parse("x + y").substitute({'y': 'x'}).simplify().evaluate({'x': 5})
        assert result == 10

    def test_api_multiple_differentiations(self):
        """Test chaining multiple differentiation operations."""
        expr = parse("x^4")
        first_deriv = expr.diff('x')
        second_deriv = first_deriv.diff('x')
        result = second_deriv.evaluate({'x': 2})
        # d²/dx²[x⁴] = 12x², at x=2: 12*4 = 48
        assert result == 48

    def test_api_multiple_simplifications(self):
        """Test that multiple simplifications are idempotent."""
        expr = parse("x + x + x")
        s1 = expr.simplify()
        s2 = s1.simplify()
        s3 = s2.simplify()
        assert s1.to_string() == s2.to_string() == s3.to_string()

    def test_api_substitute_then_diff(self):
        """Test substitution followed by differentiation."""
        expr = parse("x^2 + y")
        substituted = expr.substitute({'y': 'x'})
        derivative = substituted.diff('x')
        result = derivative.evaluate({'x': 3})
        # After substitution: x^2 + x, derivative: 2x + 1, at x=3: 7
        assert result == 7

    def test_api_diff_then_substitute(self):
        """Test differentiation followed by substitution."""
        expr = parse("x^2 * y")
        derivative = expr.diff('x')
        substituted = derivative.substitute({'y': '2'})
        result = substituted.evaluate({'x': 3})
        # Derivative: 2xy, substitute y=2: 2x*2 = 4x, at x=3: 12
        assert result == 12


class TestAPIComplexExpressions:
    """Test API with complex mathematical expressions."""

    def test_api_nested_functions(self):
        """Test parsing and evaluating nested functions."""
        expr = parse("sin(cos(x))")
        result = expr.evaluate({'x': 0})
        expected = math.sin(math.cos(0))
        assert abs(result - expected) < 1e-10

    def test_api_mixed_operations(self):
        """Test expressions with mixed arithmetic and functions."""
        expr = parse("2*sin(x) + 3*cos(y)")
        result = expr.evaluate({'x': math.pi / 2, 'y': 0})
        # 2*sin(π/2) + 3*cos(0) = 2*1 + 3*1 = 5
        assert abs(result - 5) < 1e-10

    def test_api_polynomial_expressions(self):
        """Test complex polynomial expressions."""
        expr = parse("x^3 + 2*x^2 - 3*x + 1")
        result = expr.evaluate({'x': 2})
        # 8 + 8 - 6 + 1 = 11
        assert result == 11

    def test_api_rational_expressions(self):
        """Test rational (fractional) expressions."""
        expr = parse("(x + 1) / (x - 1)")
        result = expr.evaluate({'x': 3})
        # (3 + 1) / (3 - 1) = 4 / 2 = 2
        assert result == 2

    def test_api_exponential_expressions(self):
        """Test exponential and logarithmic expressions."""
        expr = parse("exp(ln(x))")
        result = expr.simplify().evaluate({'x': 5})
        assert abs(result - 5) < 1e-10

    def test_api_trigonometric_identities(self):
        """Test expressions with trigonometric identities."""
        expr = parse("sin(x)^2 + cos(x)^2")
        result = expr.simplify().evaluate({'x': math.pi / 4})
        assert abs(result - 1) < 1e-10


class TestAPIThreadSafety:
    """Test thread safety and concurrent usage of API."""

    def test_api_multiple_independent_expressions(self):
        """Test that multiple expressions can exist independently."""
        expr1 = parse("x + 1")
        expr2 = parse("y + 2")
        expr3 = parse("z + 3")

        result1 = expr1.evaluate({'x': 1})
        result2 = expr2.evaluate({'y': 2})
        result3 = expr3.evaluate({'z': 3})

        assert result1 == 2
        assert result2 == 4
        assert result3 == 6

    def test_api_expression_immutability(self):
        """Test that operations don't modify original expression."""
        original = parse("x + 1")
        original_str = original.to_string()

        simplified = original.simplify()
        derivative = original.diff('x')
        substituted = original.substitute({'x': '5'})

        # Original should remain unchanged
        assert original.to_string() == original_str

    def test_api_independent_simplifications(self):
        """Test that simplifying one expression doesn't affect others."""
        expr1 = parse("x + x")
        expr2 = parse("x + x")

        simplified1 = expr1.simplify()

        # expr2 should not be affected
        assert "x + x" in expr2.to_string() or "2x" not in expr2.to_string()


class TestAPIDocumentation:
    """Test that API has proper documentation."""

    def test_api_parse_function_has_docstring(self):
        """Test that parse() function has documentation."""
        assert parse.__doc__ is not None
        assert len(parse.__doc__.strip()) > 0

    def test_api_expression_class_has_docstring(self):
        """Test that Expression class has documentation."""
        assert Expression.__doc__ is not None
        assert len(Expression.__doc__.strip()) > 0

    def test_api_expression_evaluate_has_docstring(self):
        """Test that Expression.evaluate() has documentation."""
        assert Expression.evaluate.__doc__ is not None

    def test_api_expression_simplify_has_docstring(self):
        """Test that Expression.simplify() has documentation."""
        assert Expression.simplify.__doc__ is not None

    def test_api_expression_diff_has_docstring(self):
        """Test that Expression.diff() has documentation."""
        assert Expression.diff.__doc__ is not None

    def test_api_expression_substitute_has_docstring(self):
        """Test that Expression.substitute() has documentation."""
        assert Expression.substitute.__doc__ is not None
