# Symcore

**Symcore** is a lightweight, pure Python library for symbolic mathematics. It provides tools for parsing mathematical expressions, performing algebraic simplification, symbolic differentiation, variable substitution, and numerical evaluation.
Designed with modularity in mind, Symcore includes a custom lexer, a recursive descent parser, and an Abstract Syntax Tree (AST) engine, making it suitable for educational purposes and lightweight symbolic computation tasks.

## Project Description

The goal of **Symcore** is to provide a standalone engine for symbolic computation without external dependencies (other than testing tools). It allows users to input mathematical strings, manipulate them symbolically, and evaluate them numerically.

Key structural components include:
* **Lexer & Parser:** converts strings into token streams and ASTs.
* **AST Nodes:** represents operations (`+`, `-`, `*`, `/`, `^`), functions (`sin`, `cos`, `exp`, `ln`, etc.), variables, and literals.
* **Engine:** handles logic for differentiation, simplification, and evaluation.

## Features

* **Parsing:** supports standard infix notation, operator precedence, and nested parentheses.
* **Symbolic Differentiation:** derivatives can be computed with respect to any variable using standard calculus rules (Product, Quotient, Chain rules, etc.).
* **Algebraic Simplification:**
    * Constant folding (e.g., `2 + 3` to `5`).
    * Identity application (e.g., `x * 1` to `x`, `x + 0` to `x`).
    * Term combination (e.g., `2x + 3x` to `5x`).
* **Substitution:** replace variables with numbers or other symbolic expressions.
* **Numerical Evaluation:** calculate the precise value of an expression given a context.
* **CLI Interface:** perform operations directly from the command line.


## Installation & Usage as a Library

### Installation
Clone the repository and install the dependencies (primarily for testing):
```bash
git clone https://github.com/dk872/symcore
cd symcore
pip install -r requirements.txt
```

To use the library in your code, ensure the `src` directory is in your `PYTHONPATH` or install the package in editable mode:
```bash
pip install -e .
```

### Library Examples
Here is how you can use `Symcore` in your Python scripts:
```Python
from symcore import parse

# 1. Parsing
expr = parse("x^2 + 3*x*y + sin(y)")
print(f"Original: {expr.to_string()}")

# 2. Symbolic Differentiation
# Differentiate with respect to 'x'
deriv = expr.diff('x')
print(f"Derivative (d/dx): {deriv.to_string()}")

# 3. Simplification
# Simplifies complex algebraic structures
simple_expr = parse("x + x + 2*x * 0 + 5").simplify()
print(f"Simplified: {simple_expr.to_string()}")  # Output: 2x + 5

# 4. Substitution & Evaluation
# Substitute y = 2, then evaluate at x = 3
subbed = expr.substitute({'y': 2})
result = subbed.evaluate({'x': 3})
print(f"Result: {result}")
```

You can also view and run the `usage_example.py` file, which also contains demo cases:

```bash
python usage_example.py
```

## CLI Usage

Symcore provides a Command Line Interface (CLI) to process expressions without writing Python scripts.

Command Structure:
```bash
python -m src.cli "expression" [options]
```

### CLI Examples

- Simplify an expression:
```bash
python -m src.cli "x + x + 3 - 1" --simplify
# Output: Simplified: 2x + 2
```
- Differentiate an expression:
```bash
python -m src.cli "x^2 + sin(x)" --diff x
# Output: Derivative w.r.t x: 2x + cos(x)
```
- Symbolic substitution:
```bash
python -m src.cli "x^2 + y" --substitute x=a+1 y=5
# Output: Substituted: 5 + (a + 1) ^ 2
```
- Evaluate numerically:
```bash
python -m src.cli "x * y + 10" --eval x=5 y=2
# Output: Numeric value: 20
```
- Chain operations (Differentiate -> Simplify):
```bash
python -m src.cli "(x^2 + 1)^3" --diff x --simplify
# Output: 
# Derivative w.r.t x: 2 * (3x^4 + 6x^2 + 3) * x
# Simplified: (6x^4 + 12x^2 + 6) * x
```

## Testing

The project maintains a high standard of code quality through a comprehensive test suite using `pytest` and `hypothesis`. The tests cover various levels of the application, from individual components to full system workflows.

To run all tests:
```bash
pytest
```
To run a specific category of tests (e.g., `performance`) with output of the received data:
```bash
pytest -s -m performance
```

### Test types

The repository includes the following types of tests:

- **Unit Tests** (`tests/test_lexer.py`, `tests/test_parser.py`, `tests/test_differentiation.py`, `test_simplification.py`, `test_evaluation_substitution.py`):
  - Verify the correctness of isolated components like the `Lexer`, `Parser`, and specific differentiation rules.
  - Ensure that individual AST nodes (`Binary/Unary operators`) behave correctly.
- **PBT Tests (Property-Based Testing)** (`tests/test_properties.py`):
  - Powered by the `hypothesis` library.
  - These tests generate thousands of random inputs to verify mathematical properties such as:
    - Commutativity: `a + b == b + a`
    - Associativity: `(a * b) * c == a * (b * c)`
    - Inverses: `exp(ln(x)) == x`
- **Performance Tests** (`tests/test_performance.py`):
  - Measure the time complexity of Parsing, Simplification, and Differentiation.
  - Benchmarks are run against inputs of increasing size (e.g., deeply nested expressions or long sums).
- **Integration Tests** (`tests/test_integration.py`):
  - Validate the interaction between multiple modules.
  - Examples include pipelines like `Parse` -> `Substitute` -> `Simplify` -> `Evaluate` to ensure data flows correctly through the system.
- **System Tests** (`tests/test_system_interface.py`):
  - Treat the application as a black box.
  - These tests spawn subprocesses to execute the `CLI` commands and verify the standard output (`stdout`) and error reporting.

## Author info
Dmytro Kulyk, a student of group IM-32.