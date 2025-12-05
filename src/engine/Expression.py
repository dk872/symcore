from __future__ import annotations
import math
from typing import Dict, Union, Any
from .Parser import Parser
from ..nodes.Node import Node
from ..nodes.Literal import Literal
from ..nodes.Variable import Variable
from ..nodes.UnaryOperator import UnaryOperator
from ..nodes.BinaryOperator import BinaryOperator

Number = Union[int, float]


def parse(expr_str: str) -> "Expression":
    """Parse a string into an Expression object."""
    return Expression.parse(expr_str)


class Expression:
    """Represents a mathematical expression with parsing, evaluation, and symbolic manipulation capabilities."""

    __slots__ = ("_root",)

    def __init__(self, root: Node):
        """Initialize expression with an AST root node."""
        self._root = root

    def __repr__(self) -> str:
        """Return string representation of the expression."""
        return f"Expression({self.to_string()})"

    @classmethod
    def parse(cls, expr_str: str) -> "Expression":
        """Parse a string into an Expression."""
        parser = Parser(expr_str)
        root = parser.parse()
        return cls(root)

    @classmethod
    def from_node(cls, node: Node) -> "Expression":
        """Create a simplified Expression from an AST node."""
        return cls(node.simplify())

    def to_string(self) -> str:
        """Convert expression to string representation."""
        return self._root.to_string()

    @property
    def root(self) -> Node:
        """Get a copy of the root AST node."""
        return self._root.copy()

    def simplify(self) -> "Expression":
        """Return a simplified version of the expression."""
        return Expression(self._root.simplify())

    def substitute(self, values: Dict[str, Union[Number, Node, str]], simplify: bool = False) -> "Expression":
        """Substitute variables with given values (symbolic or numeric)."""
        node_map = self._build_substitution_map(values)
        substituted = self._root.substitute(node_map)
        if simplify:
            substituted = substituted.simplify()
        return Expression(substituted)

    def evaluate(self, values: Dict[str, Number] = None) -> Number:
        """Evaluate expression numerically with given variable values."""
        values = values or {}
        substitution_map = {k: Literal(v) for k, v in values.items()}
        node = self._root.substitute(substitution_map).simplify()
        try:
            return _eval_node(node)
        except Exception as exc:
            raise ValueError(f"Error evaluating expression {node.to_string()}: {exc}") from exc

    def diff(self, variable: str) -> "Expression":
        """Compute the derivative with respect to the given variable."""
        derived = _diff_node(self._root, variable)
        return Expression(derived).simplify()

    def _build_substitution_map(self, values: Dict[str, Union[Number, Node, str]]) -> Dict[str, Node]:
        """Convert substitution values to Node objects."""
        node_map: Dict[str, Node] = {}
        for key, value in values.items():
            node_map[key] = self._convert_value_to_node(value)
        return node_map

    @staticmethod
    def _convert_value_to_node(value: Union[Number, Node, str, "Expression"]) -> Node:
        """Convert a single value to a Node object."""
        if isinstance(value, Node):
            return value
        if isinstance(value, Expression):
            return value.root
        if isinstance(value, (int, float)):
            return Literal(value)
        if isinstance(value, str):
            return Parser(value).parse()
        raise ValueError(f"Unsupported substitution type: {type(value)}")

    def __add__(self, other: Any) -> "Expression":
        """Addition operator overload."""
        right = _ensure_node(other)
        return Expression(BinaryOperator("+", self._root, right))

    def __radd__(self, other: Any) -> "Expression":
        """Right addition operator overload."""
        left = _ensure_node(other)
        return Expression(BinaryOperator("+", left, self._root))

    def __sub__(self, other: Any) -> "Expression":
        """Subtraction operator overload."""
        right = _ensure_node(other)
        return Expression(BinaryOperator("-", self._root, right))

    def __rsub__(self, other: Any) -> "Expression":
        """Right subtraction operator overload."""
        left = _ensure_node(other)
        return Expression(BinaryOperator("-", left, self._root))

    def __mul__(self, other: Any) -> "Expression":
        """Multiplication operator overload."""
        right = _ensure_node(other)
        return Expression(BinaryOperator("*", self._root, right))

    def __rmul__(self, other: Any) -> "Expression":
        """Right multiplication operator overload."""
        left = _ensure_node(other)
        return Expression(BinaryOperator("*", left, self._root))

    def __truediv__(self, other: Any) -> "Expression":
        """Division operator overload."""
        right = _ensure_node(other)
        return Expression(BinaryOperator("/", self._root, right))

    def __rtruediv__(self, other: Any) -> "Expression":
        """Right division operator overload."""
        left = _ensure_node(other)
        return Expression(BinaryOperator("/", left, self._root))

    def __pow__(self, other: Any) -> "Expression":
        """Power operator overload."""
        right = _ensure_node(other)
        return Expression(BinaryOperator("^", self._root, right))

    def __rpow__(self, other: Any) -> "Expression":
        """Right power operator overload."""
        left = _ensure_node(other)
        return Expression(BinaryOperator("^", left, self._root))

    def __neg__(self) -> "Expression":
        """Negation operator overload."""
        return Expression(UnaryOperator("-", self._root))


def _ensure_node(value: Any) -> Node:
    """Convert a value to a Node if possible, otherwise raise an error."""
    if isinstance(value, Node):
        return value
    if isinstance(value, Expression):
        return value.root.copy()
    if isinstance(value, (int, float)):
        return Literal(value)
    if isinstance(value, str):
        return Parser(value).parse()
    raise ValueError(f"Cannot convert {value!r} to AST Node.")


def _eval_node(node: Node) -> Number:
    """Recursively evaluate an AST node to a numeric value."""
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, UnaryOperator):
        operand_value = _eval_node(node.operand)
        return node.eval_unary(operand_value)
    if isinstance(node, BinaryOperator):
        left_value = _eval_node(node.left)
        right_value = _eval_node(node.right)
        return node.eval_binary(left_value, right_value)
    if isinstance(node, Variable):
        raise ValueError(f"Unbound variable {node.name}")
    raise TypeError(f"Unknown node type {type(node)}")


def _diff_node(node: Node, variable: str) -> Node:
    """Compute the derivative of an AST node with respect to a variable."""
    if isinstance(node, Literal):
        return _diff_literal()
    if isinstance(node, Variable):
        return _diff_variable(node, variable)
    if isinstance(node, UnaryOperator):
        return _diff_unary_operator(node, variable)
    if isinstance(node, BinaryOperator):
        return _diff_binary_operator(node, variable)
    raise TypeError(f"Unknown node type for differentiation: {type(node)}")


def _diff_literal() -> Node:
    """Derivative of a constant is zero."""
    return Literal(0)


def _diff_variable(node: Variable, variable: str) -> Node:
    """Derivative of a variable: 1 if same variable, 0 otherwise."""
    return Literal(1) if node.name == variable else Literal(0)


def _diff_unary_operator(node: UnaryOperator, variable: str) -> Node:
    """Compute derivative of a unary operator using chain rule."""
    operand = node.operand
    operand_derivative = _diff_node(operand, variable)
    operator = node.op

    if operator == "+":
        return operand_derivative
    if operator == "-":
        return UnaryOperator("-", operand_derivative)
    if operator == "sin":
        return _diff_sin(operand, operand_derivative)
    if operator == "cos":
        return _diff_cos(operand, operand_derivative)
    if operator == "tan":
        return _diff_tan(operand, operand_derivative)
    if operator == "cot":
        return _diff_cot(operand, operand_derivative)
    if operator == "exp":
        return _diff_exp(operand, operand_derivative)
    if operator == "ln":
        return _diff_ln(operand, operand_derivative)
    if operator == "log":
        return _diff_log(operand, operand_derivative)
    if operator == "sqrt":
        return _diff_sqrt(operand, operand_derivative)

    raise ValueError(f"Derivative of unknown unary op {operator}")


def _diff_sin(operand: Node, operand_derivative: Node) -> Node:
    """Derivative of sin(u): cos(u) * u'."""
    return BinaryOperator("*", UnaryOperator("cos", operand), operand_derivative)


def _diff_cos(operand: Node, operand_derivative: Node) -> Node:
    """Derivative of cos(u): -sin(u) * u'."""
    return BinaryOperator("*", UnaryOperator("-", UnaryOperator("sin", operand)), operand_derivative)


def _diff_tan(operand: Node, operand_derivative: Node) -> Node:
    """Derivative of tan(u): (1/cos²(u)) * u'."""
    cos_operand = UnaryOperator("cos", operand)
    denominator = BinaryOperator("*", cos_operand, cos_operand)
    return BinaryOperator("*", BinaryOperator("/", Literal(1), denominator), operand_derivative)


def _diff_cot(operand: Node, operand_derivative: Node) -> Node:
    """Derivative of cot(u): (-1/sin²(u)) * u'."""
    sin_operand = UnaryOperator("sin", operand)
    denominator = BinaryOperator("*", sin_operand, sin_operand)
    neg_one = UnaryOperator("-", Literal(1))
    neg_derivative = BinaryOperator("*", neg_one, operand_derivative)
    return BinaryOperator("/", neg_derivative, denominator)


def _diff_exp(operand: Node, operand_derivative: Node) -> Node:
    """Derivative of exp(u): exp(u) * u'."""
    return BinaryOperator("*", UnaryOperator("exp", operand), operand_derivative)


def _diff_ln(operand: Node, operand_derivative: Node) -> Node:
    """Derivative of ln(u): (1/u) * u'."""
    return BinaryOperator("/", operand_derivative, operand)


def _diff_log(operand: Node, operand_derivative: Node) -> Node:
    """Derivative of log₁₀(u): (1/(u*ln(10))) * u'."""
    ln_10 = Literal(math.log(10))
    denominator = BinaryOperator("*", operand, ln_10)
    return BinaryOperator("*", BinaryOperator("/", Literal(1), denominator), operand_derivative)


def _diff_sqrt(operand: Node, operand_derivative: Node) -> Node:
    """Derivative of sqrt(u): (1/(2*sqrt(u))) * u'."""
    denominator = BinaryOperator("*", Literal(2), UnaryOperator("sqrt", operand))
    return BinaryOperator("*", operand_derivative, BinaryOperator("/", Literal(1), denominator))


def _diff_binary_operator(node: BinaryOperator, variable: str) -> Node:
    """Compute derivative of a binary operator."""
    operator = node.op
    left = node.left
    right = node.right
    left_derivative = _diff_node(left, variable)
    right_derivative = _diff_node(right, variable)

    if operator == "+":
        return BinaryOperator("+", left_derivative, right_derivative)
    if operator == "-":
        return BinaryOperator("-", left_derivative, right_derivative)
    if operator == "*":
        return _diff_product(left, right, left_derivative, right_derivative)
    if operator == "/":
        return _diff_quotient(left, right, left_derivative, right_derivative)
    if operator == "^":
        return _diff_power(left, right, left_derivative, right_derivative)

    raise ValueError(f"Derivative of unknown binary op {operator}")


def _diff_product(left: Node, right: Node, left_derivative: Node, right_derivative: Node) -> Node:
    """Product rule: (uv)' = u'v + uv'."""
    term1 = BinaryOperator("*", left_derivative, right)
    term2 = BinaryOperator("*", left, right_derivative)
    return BinaryOperator("+", term1, term2)


def _diff_quotient(left: Node, right: Node, left_derivative: Node, right_derivative: Node) -> Node:
    """Quotient rule: (u/v)' = (u'v - uv')/v²."""
    numerator_term1 = BinaryOperator("*", left_derivative, right)
    numerator_term2 = BinaryOperator("*", left, right_derivative)
    numerator = BinaryOperator("-", numerator_term1, numerator_term2)
    denominator = BinaryOperator("*", right, right)
    return BinaryOperator("/", numerator, denominator)


def _diff_power(base: Node, exponent: Node, base_derivative: Node, exponent_derivative: Node) -> Node:
    """Power rule: (u^v)' depends on whether exponent is constant or variable."""
    if isinstance(exponent, Literal):
        return _diff_power_constant_exponent(base, exponent, base_derivative)
    return _diff_power_general(base, exponent, base_derivative, exponent_derivative)


def _diff_power_constant_exponent(base: Node, exponent: Literal, base_derivative: Node) -> Node:
    """Power rule for constant exponent: (u^n)' = n*u^(n-1)*u'."""
    n = exponent.value
    coefficient = Literal(n)
    new_exponent = Literal(n - 1)
    power_term = BinaryOperator("^", base, new_exponent)
    return BinaryOperator("*", BinaryOperator("*", coefficient, power_term), base_derivative)


def _diff_power_general(base: Node, exponent: Node, base_derivative: Node, exponent_derivative: Node) -> Node:
    """General power rule: (u^v)' = u^v * (v'*ln(u) + v*u'/u)."""
    part1 = BinaryOperator("*", exponent_derivative, UnaryOperator("ln", base))
    part2 = BinaryOperator("*", exponent, BinaryOperator("/", base_derivative, base))
    inner_sum = BinaryOperator("+", part1, part2)
    power_term = BinaryOperator("^", base, exponent)
    return BinaryOperator("*", power_term, inner_sum)
