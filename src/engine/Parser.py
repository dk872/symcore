from __future__ import annotations
import math
from ..nodes.Node import Node
from ..nodes.Literal import Literal
from ..nodes.Variable import Variable
from ..nodes.UnaryOperator import UnaryOperator
from ..nodes.BinaryOperator import BinaryOperator
from .Lexer import Lexer
from .Token import Token

FUNCTIONS = {"sin", "cos", "tan", "cot", "ln", "log", "exp", "sqrt"}


class Parser:
    """Expression parser implementing precedence climbing."""

    PRECEDENCE = {
        "+": 10,
        "-": 10,
        "*": 20,
        "/": 20,
        "^": 30,
    }

    def __init__(self, text: str):
        """Initialize parser with input expression."""
        self.lexer = Lexer(text)
        self.current = self.lexer.next()

    def _eat(self, kind: str, value=None):
        """Consume the current token if it matches expected type/value."""
        if self.current.kind != kind or (value is not None and self.current.value != value):
            raise ValueError(f"Unexpected token {self.current}, expected {kind} {value}")
        self.current = self.lexer.next()

    @staticmethod
    def _op_value(raw: str) -> str:
        """Normalize operator token value (** → ^)."""
        return "^" if raw == "**" else raw

    def parse(self) -> Node:
        """Parse full expression and return AST root."""
        node = self._expr(0)
        if self.current.kind != Token.EOF:
            raise ValueError("Unexpected input after end of expression")
        return node

    def _expr(self, min_prec: int) -> Node:
        """Parse expression using precedence climbing algorithm."""
        left = self._unary()

        # Continue while the next token is an operator with sufficient precedence
        while self.current.kind == Token.OP and (self._op_value(self.current.value) in self.PRECEDENCE):
            op_symbol = self._op_value(self.current.value)
            prec = self.PRECEDENCE[op_symbol]

            if prec < min_prec:
                break

            # Exponentiation is right-associative
            next_min = prec if op_symbol == "^" else prec + 1
            self._eat(Token.OP)

            right = self._expr(next_min)
            left = BinaryOperator(op_symbol, left, right)

        return left

    def _unary(self) -> Node:
        """Parse unary operators (+, -)."""
        if self.current.kind == Token.OP and self.current.value in ("+", "-"):
            op = self.current.value
            self._eat(Token.OP)
            return UnaryOperator(op, self._unary())
        return self._primary()

    def _primary(self) -> Node:
        """Parse literals, variables, function calls, and parentheses."""
        token = self.current

        if token.kind == Token.NUMBER:
            self._eat(Token.NUMBER)
            value = token.value
            # Convert float 5.0 → int 5
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return Literal(value)

        if token.kind == Token.IDENT:
            name = token.value
            self._eat(Token.IDENT)

            # Constant pi
            if name == "pi":
                return Literal(math.pi)

            # Function call: name(...)
            if self._is_function_call(name):
                return self._parse_function_call(name)

            return Variable(name)

        # Parenthesized expression
        if token.kind == Token.OP and token.value == "(":
            return self._parse_parenthesized()

        raise ValueError(f"Unexpected token in primary: {token}")

    def _is_function_call(self, name: str) -> bool:
        """Return True if IDENT is followed by '(' and is a known function."""
        return (
            name in FUNCTIONS
            and self.current.kind == Token.OP
            and self.current.value == "("
        )

    def _parse_function_call(self, name: str) -> Node:
        """Parse f(expr) style unary function call."""
        self._eat(Token.OP, "(")
        arg = self._expr(0)
        self._eat(Token.OP, ")")
        return UnaryOperator(name, arg)

    def _parse_parenthesized(self) -> Node:
        """Parse expression inside parentheses."""
        self._eat(Token.OP, "(")
        node = self._expr(0)
        self._eat(Token.OP, ")")
        return node
