from __future__ import annotations
from typing import List
from .Token import Token, TOKEN_REGEX  # TOKEN_REGEX is imported but defined elsewhere


class Lexer:
    """Tokenizes an input mathematical string into a sequence of Tokens."""

    def __init__(self, text: str):
        """Initializes the lexer by tokenizing the input text."""
        self.tokens = self._tokenize(text)
        self.pos = 0  # Current position in the token list

    @staticmethod
    def _tokenize(text: str) -> List[Token]:
        """Converts the raw input string into a list of Token objects."""
        token_list: List[Token] = []

        # Use the predefined TOKEN_REGEX to find all components
        for match in TOKEN_REGEX.finditer(text):
            # The regex groups should match in this order
            number_str, identifier_str, operator_str = match.groups()

            if number_str:
                # Convert numbers to float immediately
                token_list.append(Token(Token.NUMBER, float(number_str)))
            elif identifier_str:
                token_list.append(Token(Token.IDENT, identifier_str))
            elif operator_str:
                token_list.append(Token(Token.OP, operator_str))

        # Append the End Of File token to mark the end of the input stream
        token_list.append(Token(Token.EOF, None))
        return token_list

    def next(self) -> Token:
        """Returns the current token and advances the position to the next token."""
        current_token = self.tokens[self.pos]
        self.pos += 1
        return current_token
