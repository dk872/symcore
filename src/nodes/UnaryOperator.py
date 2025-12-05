from __future__ import annotations
from typing import Dict, Union, Generator, Optional
import math
from .Node import Node
from .Literal import Literal

# Type alias for numeric values
Number = Union[int, float]
EPSILON = 1e-10


class UnaryOperator(Node):
    """
    Represents a unary operation (e.g., -x, sin(x), sqrt(x)).
    """
    PRECEDENCE = 25

    # Set of operators that support constant folding
    FOLDABLE_OPS = {"ln", "log", "exp", "sin", "cos", "tan", "cot", "sqrt", "+", "-"}

    def __init__(self, op: str, operand: Node):
        """Initializes a UnaryOperator with an operator string and an operand node."""
        self.op = op
        self.operand = operand

    def __repr__(self) -> str:
        """Returns a string representation for debugging."""
        return f"UnaryOperator('{self.op}', {self.operand!r})"

    def to_string(self, parent_prec: int = 0, position: str = "") -> str:
        """Converts the AST node back to a string expression."""
        if parent_prec == 0 and position == "" and self._string_cache is not None:
            return self._string_cache

        if self.op in ("+", "-"):
            # Format: -x
            inner = self.operand.to_string(self.precedence(), 'right')
            s = f"{self.op}{inner}"
        else:
            # Format: func(x)
            inner = self.operand.to_string()
            s = f"{self.op}({inner})"

        # Add parentheses if precedence requires it
        if self.precedence() < parent_prec:
            result = f"({s})"
        else:
            result = s

        if parent_prec == 0 and position == "":
            self._string_cache = result

        return result

    def precedence(self) -> int:
        """Returns the precedence level of this operator."""
        return self.PRECEDENCE

    def substitute(self, values: Dict[str, Node]) -> Node:
        """Substitutes variables within the operand."""
        return UnaryOperator(self.op, self.operand.substitute(values))

    def simplify(self) -> Node:
        """Simplifies the unary expression using algebraic rules and constant folding."""
        # Simplify the operand first (bottom-up approach)
        operand = self.operand.simplify()

        # 1. Attempt constant folding (e.g., sqrt(4) -> 2)
        if isinstance(operand, Literal) and self.op in self.FOLDABLE_OPS:
            folded = self._fold_constant(operand)
            if folded:
                return folded

        # 2. Dispatch to specific simplification logic based on the operator
        match self.op:
            case "sqrt":
                return self._simplify_sqrt(operand)
            case "-":
                return self._simplify_negation(operand)
            case "ln":
                return self._simplify_ln(operand)
            case "exp":
                return self._simplify_exp(operand)
            case "sin" | "cos":
                return self._simplify_trig(operand)
            case "+" if isinstance(operand, UnaryOperator) and operand.op == "+":
                # Rule: +(+x) -> x
                return operand.operand

        # Default: return new node with simplified operand
        return UnaryOperator(self.op, operand)

    def eval_unary(self, v: Number) -> Number:
        """Evaluates the unary operation numerically."""
        self._check_domain(v)

        match self.op:
            case "ln":
                return math.log(v)
            case "log":
                return math.log10(v)
            case "exp":
                return math.exp(v)
            case "sqrt":
                return math.sqrt(v)
            case "sin":
                return math.sin(v)
            case "cos":
                return math.cos(v)
            case "tan":
                return math.tan(v)
            case "cot":
                return 1.0 / math.tan(v)
            case "+":
                return +v
            case "-":
                return -v
            case _:
                raise ValueError(f"Unknown unary operator {self.op}")

    def _fold_constant(self, operand: Literal) -> Optional[Literal]:
        """Attempts to evaluate the expression if the operand is a literal."""
        val = self.eval_unary(operand.value)
        return self._to_literal(val)

    def _simplify_sqrt(self, operand: Node) -> Node:
        """Handles simplification rules for square roots."""
        # Rule 1: sqrt(x^n) -> simplify powers
        res = self._simplify_sqrt_power(operand)
        if res:
            return res

        # Rule 2: sqrt(a * b * ...) -> distribute sqrt over perfect squares
        res = self._simplify_sqrt_product(operand)
        if res:
            return res

        return UnaryOperator("sqrt", operand)

    def _simplify_sqrt_power(self, operand: Node) -> Optional[Node]:
        """Simplifies sqrt(x^n) where n is even."""
        from .BinaryOperator import BinaryOperator

        if isinstance(operand, BinaryOperator) and operand.op == "^":
            # Case: sqrt(x^2) -> x
            if isinstance(operand.right, Literal) and operand.right.value == 2:
                return operand.left

            # Case: sqrt(x^n) where n is even -> x^(n/2)
            if isinstance(operand.right, Literal) and operand.right.value % 2 == 0:
                new_exp = operand.right.value / 2
                return BinaryOperator("^", operand.left, self._to_literal(new_exp)).simplify()
        return None

    def _simplify_sqrt_product(self, operand: Node) -> Optional[Node]:
        """Simplifies sqrt(product) by extracting perfect squares."""
        from .BinaryOperator import BinaryOperator

        if isinstance(operand, BinaryOperator) and operand.op == "*":
            factors = list(self._flatten_mul(operand))
            sqrt_factors = []

            for f in factors:
                simplified_factor = self._extract_perfect_square(f)
                if simplified_factor:
                    sqrt_factors.append(simplified_factor)
                else:
                    sqrt_factors.append(UnaryOperator("sqrt", f))

            # Reconstruct the product from simplified factors
            if not sqrt_factors:
                return None

            result = sqrt_factors[0]
            for f in sqrt_factors[1:]:
                result = BinaryOperator("*", result, f)
            return result

        return None

    def _extract_perfect_square(self, factor: Node) -> Optional[Node]:
        """Helper to check if a factor is a perfect square and return its root."""
        from .BinaryOperator import BinaryOperator

        # Case: Perfect square number (e.g., 4, 9, 16)
        if isinstance(factor, Literal) and factor.value >= 0:
            sqrt_val = math.sqrt(factor.value)
            if abs(sqrt_val - round(sqrt_val)) < EPSILON:
                return self._to_literal(sqrt_val)

        # Case: Perfect square power (e.g., x^4 -> x^2)
        if (isinstance(factor, BinaryOperator) and factor.op == "^" and
                isinstance(factor.right, Literal) and factor.right.value % 2 == 0):
            new_exp = factor.right.value / 2
            if new_exp == 1:
                return factor.left
            return BinaryOperator("^", factor.left, self._to_literal(new_exp))

        return None

    @staticmethod
    def _simplify_negation(operand: Node) -> Node:
        """Handles simplification for unary minus (-)."""
        from .BinaryOperator import BinaryOperator

        # Rule: -(-x) -> x
        if isinstance(operand, UnaryOperator) and operand.op == '-':
            return operand.operand

        # Rule: -(+x) -> -x
        if isinstance(operand, UnaryOperator) and operand.op == '+':
            return UnaryOperator("-", operand.operand)

        # Rule: -(a * b) -> (-a) * b (distribute negation to first factor)
        if isinstance(operand, BinaryOperator) and operand.op == '*':
            neg_left = UnaryOperator('-', operand.left).simplify()
            return BinaryOperator('*', neg_left, operand.right).simplify()

        # Rule: -(a + b) -> -1 * (a + b)
        if isinstance(operand, BinaryOperator) and operand.op == '+':
            return BinaryOperator('*', Literal(-1), operand).simplify()

        return UnaryOperator("-", operand)

    @staticmethod
    def _simplify_ln(operand: Node) -> Node:
        """Handles simplification for natural log (ln)."""
        from .BinaryOperator import BinaryOperator

        # Rule: ln(exp(x)) -> x
        if isinstance(operand, UnaryOperator) and operand.op == "exp":
            return operand.operand

        # Rule: ln(1) -> 0
        if isinstance(operand, Literal) and operand.value == 1:
            return Literal(0)

        # Rule: ln(x^y) -> y * ln(x)
        if isinstance(operand, BinaryOperator) and operand.op == "^":
            return BinaryOperator("*", operand.right, UnaryOperator("ln", operand.left)).simplify()

        return UnaryOperator("ln", operand)

    @staticmethod
    def _simplify_exp(operand: Node) -> Node:
        """Handles simplification for exponent function (exp)."""
        from .BinaryOperator import BinaryOperator

        # Rule: exp(ln(x)) -> x
        if isinstance(operand, UnaryOperator) and operand.op == "ln":
            return operand.operand

        # Rule: exp(ln(x) * y) -> x^y (inverse of log power rule)
        if isinstance(operand, BinaryOperator) and operand.op == "*" and \
                isinstance(operand.right, UnaryOperator) and operand.right.op == "ln":
            base = operand.right.operand
            exponent = operand.left
            return BinaryOperator("^", base, exponent).simplify()

        return UnaryOperator("exp", operand)

    def _simplify_trig(self, operand: Node) -> Node:
        """Handles simplification for trig functions with negative arguments."""
        # Rules: sin(-x) -> -sin(x), cos(-x) -> cos(x)
        if isinstance(operand, UnaryOperator) and operand.op == '-':
            inner = operand.operand
            if self.op == 'sin':
                return UnaryOperator("-", UnaryOperator("sin", inner)).simplify()
            if self.op == 'cos':
                return UnaryOperator("cos", inner).simplify()

        return UnaryOperator(self.op, operand)

    def _flatten_mul(self, node: Node) -> Generator[Node, None, None]:
        """Recursively flattens nested multiplication nodes."""
        from .BinaryOperator import BinaryOperator
        if isinstance(node, BinaryOperator) and node.op == "*":
            yield from self._flatten_mul(node.left)
            yield from self._flatten_mul(node.right)
        else:
            yield node

    def _check_domain(self, v: Number) -> None:
        """Validates domain constraints for evaluation."""
        if self.op in ("ln", "log") and v <= 0:
            raise ValueError(f"Domain error: {self.op} requires argument > 0, got {v}")
        if self.op == "sqrt" and v < 0:
            raise ValueError(f"Domain error: sqrt requires argument >= 0, got {v}")
        if self.op == "tan" and abs(math.cos(v)) < EPSILON:
            raise ValueError(f"Domain error: tan({v}) is undefined (asymptote).")
        if self.op == "cot" and abs(math.sin(v)) < EPSILON:
            raise ValueError(f"Domain error: cot({v}) is undefined (asymptote).")

    @staticmethod
    def _to_literal(val: float) -> Literal:
        """Converts a float to a Literal, handling integer rounding."""
        if abs(val - round(val)) < EPSILON:
            return Literal(int(round(val)))
        return Literal(val)
