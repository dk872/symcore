from __future__ import annotations
from typing import Dict
from .Node import Node


class Variable(Node):
    """Represents a symbolic variable (e.g., x, y, a)."""

    def __init__(self, name: str):
        """Initializes the variable node with its symbolic name."""
        self.name = name

    def __repr__(self) -> str:
        """Returns a string representation for debugging (e.g., Variable('x'))."""
        return f"Variable('{self.name}')"

    def to_string(self, parent_prec: int = 0, position: str = "") -> str:
        """Renders the variable name to a string."""
        return self.name

    def simplify(self) -> "Node":
        """Simplification of a variable returns itself."""
        return self

    def substitute(self, values: Dict[str, "Node"]) -> "Node":
        """
        Substitutes the variable if its name is present in the mapping.
        If found, returns a deep copy of the replacement expression/node; otherwise, returns itself.
        """
        # Check if the variable name exists in the substitution map
        if self.name in values:
            # Return a copy of the substitution node to avoid modifying the substitution map's AST
            return values[self.name].copy()
        return self

    def precedence(self) -> int:
        """Returns the highest precedence for a leaf node (100)."""
        # Variables, like Literals, are leaf nodes in the AST.
        return 100
