import argparse
from .engine.Expression import parse
from typing import Dict, Union, Any


def _parse_eval_assignments(assignments: list[str]) -> Dict[str, Union[int, float]]:
    """Parses key=value strings into a dictionary for numerical evaluation."""
    eval_values = {}

    for assignment in assignments:
        if '=' not in assignment:
            raise ValueError(f"Invalid format: {assignment}. Use key=value.")

        key, value_str = assignment.split('=', 1)

        try:
            value = float(value_str)
            # Convert to int if the value is an exact integer (e.g., 5.0 -> 5)
            if value.is_integer():
                value = int(value)
            eval_values[key.strip()] = value

        except ValueError:
            raise ValueError(f"Invalid numeric value for '{key}': {value_str}")

    return eval_values


def _handle_differentiation(expr: Any, variable_name: str) -> Any:
    """Differentiates the expression with respect to the given variable."""
    try:
        derived_expr = expr.diff(variable_name)
        print(f"Derivative w.r.t {variable_name}: {derived_expr.to_string()}")
        return derived_expr
    except Exception as e:
        raise RuntimeError(f"Failed to differentiate: {e}")


def _handle_simplification(expr: Any) -> Any:
    """Simplifies the expression algebraically."""
    try:
        simplified_expr = expr.simplify()
        print(f"Simplified: {simplified_expr.to_string()}")
        return simplified_expr
    except Exception as e:
        raise RuntimeError(f"Failed to simplify expression: {e}")


def _handle_substitution(expr: Any, substitution_args: list[str]) -> Any:
    """Performs symbolic substitution based on key=expression strings."""
    sub_values = {}

    for arg in substitution_args:
        if '=' not in arg:
            raise ValueError(f"Invalid substitution format: {arg}. Use key=value.")

        key, value_expr = arg.split('=', 1)
        sub_values[key.strip()] = value_expr.strip()

    substituted_expr = expr.substitute(sub_values)
    print(f"Substituted: {substituted_expr.to_string()}")
    return substituted_expr


def _handle_evaluation(expr: Any, eval_args: list[str]):
    """Numerically evaluates the expression using provided substitutions."""
    try:
        eval_values = _parse_eval_assignments(eval_args)
        result = expr.evaluate(eval_values)
        print(f"Numeric value: {result}")
    except ValueError as e:
        # Handles errors from _parse_eval_assignments and unbound variables
        print(f"Evaluation error: {e}")
    except Exception as e:
        # Handles runtime evaluation errors (e.g., division by zero)
        print(f"Evaluation error: {e}")


def main():
    """Main function to parse command-line arguments and execute symbolic operations."""
    parser = argparse.ArgumentParser(description="Symbolic Expression Evaluator CLI")

    # 1. Positional Argument
    parser.add_argument("expression", type=str, help="The mathematical expression string.")

    # 2. Optional Arguments
    parser.add_argument("--diff", type=str, help="Variable name for differentiation (e.g., 'x').")
    parser.add_argument("--simplify", action="store_true", help="Applies algebraic simplification rules.")
    parser.add_argument(
        '--eval',
        nargs='*',
        help='Numerically evaluates the expression (e.g., x=2 y=5).'
    )
    parser.add_argument(
        '--substitute', '-sub',
        nargs='+',
        help='Performs symbolic substitution (e.g., x=y+1 z=10).'
    )

    args = parser.parse_args()

    if not args.expression or not args.expression.strip():
        parser.error("Expression cannot be empty.")

    expr_object = None

    try:
        # Initial parsing of the input string
        expr_object = parse(args.expression)
    except Exception as e:
        parser.error(f"Syntax error while parsing expression: {e}")

    # We execute operations in a specific order: Diff -> Simplify -> Substitute -> Eval
    # If Diff is performed, the result is saved back to expr_object for further steps.
    current_expr = expr_object

    try:
        if args.diff:
            current_expr = _handle_differentiation(current_expr, args.diff)

        if args.simplify:
            current_expr = _handle_simplification(current_expr)

        if args.substitute:
            current_expr = _handle_substitution(current_expr, args.substitute)

        if args.eval is not None:
            _handle_evaluation(current_expr, args.eval)

    except RuntimeError as e:
        # Catch exceptions thrown by helper functions and exit gracefully
        parser.error(str(e))

    if not (args.diff or args.simplify or args.eval is not None or args.substitute):
        print(f"Expression: {current_expr.to_string()}")


if __name__ == "__main__":
    main()
