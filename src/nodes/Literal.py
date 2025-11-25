from __future__ import annotations
from .Node import Node
from typing import Dict, Union

# Type alias for clarity in method signatures
Number = Union[int, float]


class Literal(Node):
    """Represents a constant numerical value (e.g., 5, 3.14, 0)."""

    def __init__(self, value: Number):
        """Initializes the literal node with a numerical value."""
        # Ensure boolean values are treated as integers (True=1, False=0).
        if isinstance(value, bool):
            value = int(value)
        self.value = value

    def __repr__(self) -> str:
        """Returns a string representation for debugging (e.g., Literal(5))."""
        return f"Literal({self.value})"

    def to_string(self, parent_prec: int = 0, position: str = "") -> str:
        """Renders the numerical value to a string, formatting floats as integers if possible."""
        # Check if the float value is mathematically equivalent to an integer.
        if isinstance(self.value, float) and self.value.is_integer():
            return str(int(self.value))
        return str(self.value)

    def simplify(self) -> "Node":
        """Simplification of a literal returns itself as it cannot be simplified further."""
        return self

    def substitute(self, values: Dict[str, "Node"]) -> "Node":
        """Substitution on a literal returns itself as it has no variables to replace."""
        return self

    def precedence(self) -> int:
        """Returns the highest precedence for a leaf node (100)."""
        return 100
