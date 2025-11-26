# This script demonstrates parsing, evaluation, differentiation, substitution,
# and programmatic expression building
from symcore import parse

print("--- Symcore Library API Demonstration ---")

expr1 = None
deriv_y = None

print("\n[1. Parsing and Evaluation]")
try:
    # Use the high-level 'parse' function to create an Expression object from a string
    math_string = "x * x + 2 * y - 3"
    expr1 = parse(math_string)

    # to_string() is used to display the expression
    print(f"Original Expression: {expr1.to_string()}")

    # evaluate(assignments) - Numerically calculate the value
    values = {'x': 5, 'y': 10}
    result = expr1.evaluate(values)
    print(f"Evaluated at x=5, y=10: {result}")  # Expected: 5*5 + 2*10 - 3 = 42

except Exception as e:
    print(f"Error during parsing or basic evaluation: {e}")
    expr1 = None
    deriv_y = None

if expr1:
    print("\n[2. Symbolic Differentiation]")

    # diff(variable_name) - Returns a new Expression object
    deriv_x = expr1.diff('x')
    print(f"Derivative w.r.t x: {deriv_x.to_string()}")  # Expected: 2*x

    deriv_y = expr1.diff('y')
    print(f"Derivative w.r.t y: {deriv_y.to_string()}")  # Expected: 2

if deriv_y:
    print("\n[3. Simplification]")

    # simplify() - Simplifies the expression algebraically.
    simplified_deriv_y = deriv_y.simplify()
    print(f"Simplified d/dy: {simplified_deriv_y.to_string()}")

    # Example: simplify a more complex expression (requires it to be parsed first)
    complex_expr = parse("a + 1*c - 0*b")
    simplified_complex_expr = complex_expr.simplify()
    print(f"Complex expression: {complex_expr.to_string()}")
    print(f"Simplified result: {simplified_complex_expr.to_string()}")  # Expected: a + c

if expr1:
    print("\n[4. Symbolic Substitution]")

    # substitute(substitution_dict) - Replaces variables with other expressions/strings
    # Substitute x with 't + 1'
    sub_expr = expr1.substitute({'x': 't + 1'})
    print(f"Original: {expr1.to_string()}")
    print(f"After substituting x = t+1: {sub_expr.to_string()}")

    # You can now evaluate the substituted expression
    sub_eval_result = sub_expr.evaluate({'t': 4, 'y': 10})
    # Check: t=4 -> x=5. Result should still be 42.
    print(f"Evaluated substituted expression at t=4, y=10: {sub_eval_result}")
