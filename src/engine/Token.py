from __future__ import annotations
import re
from typing import Optional, Any

# Regular expression to tokenize the input string.
# It handles three groups: numbers, identifiers, and operators.
TOKEN_REGEX = re.compile(r"\s*(?:(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([a-zA-Z_]\w*)|(\*\*|[+\-*/^()]))")


class Token:
    """Represents a single lexical unit (token) in the expression."""

    # Constants representing token types
    NUMBER: str = "NUMBER"
    IDENT: str = "IDENT"
    OP: str = "OP"
    EOF: str = "EOF"  # End Of File (End of input stream)

    def __init__(self, kind: str, value: Optional[Any] = None):
        """Initializes a token with its kind (type) and an optional value."""
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:
        """Returns a string representation for debugging (e.g., Token(NUMBER, 42))."""
        return f"Token({self.kind}, {self.value})"
