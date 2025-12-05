from __future__ import annotations
from typing import Dict, Union
from .Node import Node

# Type alias for numeric values
Number = Union[int, float]
EPSILON = 1e-10


class Literal(Node):
    """Represents a constant numeric value in the AST."""

    def __init__(self, value: Number):
        """Initializes the literal node with a numeric value."""
        # Ensure value is float for consistent handling, unless it's a "clean" integer
        if abs(value - round(value)) < EPSILON:
            self.value: Number = int(round(value))
        else:
            self.value: Number = float(value)

    def __repr__(self) -> str:
        """Returns a string representation for debugging (e.g., Literal(5))."""
        return f"Literal({self.value})"

    def to_string(self, parent_prec: int = 0, position: str = "") -> str:
        """Renders the literal value to a string."""
        if self._string_cache is not None:
            return self._string_cache

        # Compute and store
        if isinstance(self.value, int):
            result = str(self.value)
        else:
            # Handle float values, ensuring clean representation
            result = str(self.value)

        self._string_cache = result
        return result

    def simplify(self) -> "Node":
        """Simplification of a literal returns itself."""
        return self

    def substitute(self, values: Dict[str, "Node"]) -> "Node":
        """Substitution on a literal returns itself."""
        return self

    def precedence(self) -> int:
        """Returns the highest precedence for a leaf node (100)."""
        return 100
