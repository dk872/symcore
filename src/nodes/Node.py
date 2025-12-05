from __future__ import annotations
from typing import Dict, Optional
import copy


class Node:
    """Base abstract class for all Abstract Syntax Tree (AST) nodes."""
    op: Optional[str] = None
    left: Optional[Node] = None
    right: Optional[Node] = None
    operand: Optional[Node] = None
    _string_cache: Optional[str] = None

    def copy(self) -> "Node":
        """Returns a deep copy of the current node and its entire subtree."""
        new_node = copy.copy(self)
        new_node._string_cache = None
        return new_node

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
