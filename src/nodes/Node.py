from __future__ import annotations
from typing import Dict
import copy


class Node:
    """Base abstract class for all Abstract Syntax Tree (AST) nodes."""

    def copy(self) -> "Node":
        """Returns a deep copy of the current node and its entire subtree."""
        return copy.deepcopy(self)

    def simplify(self) -> "Node":
        """Applies algebraic simplification rules recursively (base implementation returns self)."""
        return self

    def substitute(self, values: Dict[str, "Node"]) -> "Node":
        """Substitutes variables within the node's subtree with given expressions (base implementation returns self)."""
        return self

    def to_string(self, parent_prec: int = 0, position: str = "") -> str:
        """Renders the node and its subtree into a readable mathematical string."""
        # All concrete subclasses must implement this method.
        raise NotImplementedError

    def precedence(self) -> int:
        """Returns the operator precedence level of the node (100 for leaf nodes by default)."""
        return 100
