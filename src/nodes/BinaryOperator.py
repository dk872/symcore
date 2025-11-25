from __future__ import annotations
from typing import Dict, Union, Iterable, List, Any
from .Literal import Literal
from .Node import Node
from .UnaryOperator import UnaryOperator
from .Variable import Variable

Number = Union[int, float]


class BinaryOperator(Node):
    """Represents a binary operation in the AST (e.g., addition, multiplication)."""

    PRECEDENCES = {
        "+": 10,
        "-": 10,
        "*": 20,
        "/": 20,
        "^": 30,
    }

    def __init__(self, op: str, left: Node, right: Node):
        """Initializes the binary operator with an operator string and two operands."""
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        """Returns a string representation for debugging."""
        return f"BinaryOperator('{self.op}', {self.left!r}, {self.right!r})"

    def precedence(self) -> int:
        """Returns the precedence level of the operator."""
        return self.PRECEDENCES.get(self.op, 0)

    def to_string(self, parent_prec: int = 0, position: str = "") -> str:
        """Converts the AST node back to a mathematical string."""
        my_prec = self.precedence()

        if self.op == "*":
            return self._mul_to_string(my_prec, parent_prec)

        if self.op == "+":
            return self._add_to_string(my_prec, parent_prec)

        # Generic handling for -, /, ^
        left_s = self.left.to_string(my_prec, 'left')
        right_s = self.right.to_string(my_prec, 'right')

        # Parenthesize denominator if it's a binary operator (e.g., a / (b + c))
        if self.op == "/" and isinstance(self.right, BinaryOperator):
            right_s = f"({self.right.to_string()})"

        s = f"{left_s} {self.op} {right_s}"
        return s if my_prec >= parent_prec else f"({s})"

    def _mul_to_string(self, my_prec: int, parent_prec: int) -> str:
        """Helper to format multiplication strings."""
        # Handle special case: -1 * x -> -x
        if self._is_literal(self.left, -1) and not isinstance(self.right, Literal):
            right_s = self.right.to_string(my_prec, 'right')
            s = f"-{right_s}"
            return f"({s})" if my_prec < parent_prec else s

        # Handle special case: 1 * x -> x
        if self._is_literal(self.left, 1) and not isinstance(self.right, Literal):
            return self.right.to_string(my_prec, 'right')

        # Handle implicit multiplication: 2 * x -> 2x
        if isinstance(self.left, Literal) and not isinstance(self.right, Literal):
            ls = self.left.to_string()
            rs = self.right.to_string()
            s = f"{ls}{rs}"
            return f"({s})" if my_prec < parent_prec else s

        left_s = self.left.to_string(my_prec, 'left')
        right_s = self.right.to_string(my_prec, 'right')
        s = f"{left_s} * {right_s}"
        return s if my_prec >= parent_prec else f"({s})"

    def _add_to_string(self, my_prec: int, parent_prec: int) -> str:
        """Helper to format addition strings, sorting terms by power."""
        terms = self._collect_terms()
        # Sort terms: higher powers first, then coefficients
        terms.sort(key=lambda item: (-item['power'], -item['coef'] if item['coef'] is not None else 0))

        parts = [t['text'] for t in terms]
        if not parts:
            return "0"

        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p

        return s if my_prec >= parent_prec else f"({s})"

    def _collect_terms(self) -> List[Dict[str, Any]]:
        """Recursively collects terms for addition printing (e.g. 2x, x^2)."""
        terms = []
        self._process_term(self, terms)
        return terms

    def _process_term(self, node: Node, terms: List[Dict[str, Any]]) -> None:
        """Dispatches term processing based on node type."""
        # 1. Recurse into addition
        if isinstance(node, BinaryOperator) and node.op == "+":
            self._process_term(node.left, terms)
            self._process_term(node.right, terms)
            return

        # 2. Handle Coefficients (multiplication)
        if isinstance(node, BinaryOperator) and node.op == "*":
            if self._try_process_coefficient(node, terms):
                return

        # 3. Variable alone
        if isinstance(node, Variable):
            terms.append({"text": node.name, "power": 1, "coef": 1})
            return

        # 4. Literal alone
        if isinstance(node, Literal):
            terms.append({"text": str(node.value), "power": 0, "coef": node.value})
            return

        # 5. Fallback
        terms.append({"text": node.to_string(), "power": 0, "coef": None})

    def _try_process_coefficient(self, node: BinaryOperator, terms: List[Dict[str, Any]]) -> bool:
        """Attempts to identify and process terms with coefficients like 2x or 3x^2."""
        if not isinstance(node.left, Literal):
            return False

        coef = node.left.value

        # Case: coef * Variable (e.g., 2x)
        if isinstance(node.right, Variable):
            text = self._format_term_text(coef, node.right.name)
            terms.append({"text": text, "power": 1, "coef": coef})
            return True

        # Case: coef * Variable^Power (e.g., 2x^3)
        if (isinstance(node.right, BinaryOperator) and node.right.op == "^" and
                isinstance(node.right.left, Variable) and isinstance(node.right.right, Literal)):
            power = node.right.right.value
            base = node.right.left.name
            text = self._format_term_text(coef, f"{base}^{power}")
            terms.append({"text": text, "power": power, "coef": coef})
            return True

        return False

    @staticmethod
    def _format_term_text(coef: Number, base_text: str) -> str:
        """Formats the string representation of a term, handling sign and 1/-1 coefficients."""
        abs_coef = abs(coef)
        prefix = "" if coef > 0 else "-"

        if abs_coef == 1:
            return f"{prefix}{base_text}"
        return f"{prefix}{abs_coef}{base_text}"

    def substitute(self, values: Dict[str, Node]) -> Node:
        """Recursively substitutes variables with given values."""
        return BinaryOperator(self.op, self.left.substitute(values), self.right.substitute(values))

    def eval_binary(self, a: Number, b: Number) -> Number:
        """Evaluates the binary operation numerically."""
        match self.op:
            case "+":
                return a + b
            case "-":
                return a - b
            case "*":
                return a * b
            case "^":
                return a ** b
            case "/":
                if b == 0:
                    if a == 0:
                        raise ValueError("Undefined result (0/0)")
                    else:
                        raise ZeroDivisionError("Division by zero")
                return a / b
        raise ValueError(f"Unknown binary operator {self.op}")

    def simplify(self) -> Node:
        """Simplifies the expression using algebraic rules."""
        left = self.left.simplify()
        right = self.right.simplify()

        # 1. Constant Folding: If both sides are Literals, compute the value.
        if isinstance(left, Literal) and isinstance(right, Literal):
            try:
                return Literal(self.eval_binary(left.value, right.value))
            except Exception:
                # If evaluation fails (e.g. division by zero), keep the structure
                pass

        # 2. Dispatch to specific operator handlers
        match self.op:
            case "+":
                return self._simplify_add(left, right)
            case "*":
                return self._simplify_mul(left, right)
            case "-":
                return self._simplify_sub(left, right)
            case "/":
                return self._simplify_div(left, right)
            case "^":
                return self._simplify_pow(left, right)

        return BinaryOperator(self.op, left, right)

    def _simplify_add(self, left: Node, right: Node) -> Node:
        """Handles simplification for Addition (+)."""
        # Identity: x + 0 = x
        if self._is_literal(right, 0):
            return left
        if self._is_literal(left, 0):
            return right

        # Fraction Addition: (a/b) + (c/b) -> (a+c)/b
        if (isinstance(left, BinaryOperator) and left.op == "/" and
                isinstance(right, BinaryOperator) and right.op == "/"):
            if left.right.to_string() == right.right.to_string():
                new_num = BinaryOperator("+", left.left, right.left).simplify()
                return BinaryOperator("/", new_num, left.right).simplify()

        # Trig Identity: sin^2(x) + cos^2(x) = 1
        if self._is_trig_identity(left, right):
            return Literal(1)

        # Combine Like Terms
        terms = self._flatten_add([left, right])
        return self._combine_add_terms(terms)

    def _simplify_mul(self, left: Node, right: Node) -> Node:
        """Handles simplification for Multiplication (*)."""
        # Identity: x * 1 = x
        if self._is_literal(right, 1):
            return left
        if self._is_literal(left, 1):
            return right

        # Zero Property: x * 0 = 0
        if self._is_literal(right, 0) or self._is_literal(left, 0):
            return Literal(0)

        # Sqrt Rules: sqrt(x) * sqrt(x) = x, sqrt(x) * sqrt(y) = sqrt(x*y)
        if (isinstance(left, UnaryOperator) and left.op == "sqrt" and
                isinstance(right, UnaryOperator) and right.op == "sqrt"):
            if left.operand.to_string() == right.operand.to_string():
                return left.operand.copy()
            inner_product = BinaryOperator("*", left.operand.copy(), right.operand.copy()).simplify()
            return UnaryOperator("sqrt", inner_product).simplify()

        # Negation Rules: -x * -y = x * y, -x * y = -(x * y)
        if self._is_unary(left, '-') and self._is_unary(right, '-'):
            return BinaryOperator('*', left.operand, right.operand).simplify()

        if self._is_unary(left, '-'):
            inner = BinaryOperator('*', left.operand, right)
            return BinaryOperator('*', Literal(-1), inner).simplify()

        if self._is_unary(right, '-'):
            inner = BinaryOperator('*', left, right.operand)
            return BinaryOperator('*', Literal(-1), inner).simplify()

        # Distributive Property: x * (a +/- b) -> xa +/- xb
        if isinstance(right, BinaryOperator) and right.op in ("+", "-"):
            term1 = BinaryOperator("*", left.copy(), right.left.copy()).simplify()
            term2 = BinaryOperator("*", left.copy(), right.right.copy()).simplify()
            return BinaryOperator(right.op, term1, term2).simplify()

        # Exponent Rules: x^a * x^b -> x^(a+b)
        if (isinstance(left, BinaryOperator) and left.op == "^" and
                isinstance(right, Variable) and isinstance(left.left, Variable) and
                right.name == left.left.name):
            new_exp = BinaryOperator("+", left.right, Literal(1)).simplify()
            return BinaryOperator("^", left.left, new_exp).simplify()

        if (isinstance(right, BinaryOperator) and right.op == "^" and
                isinstance(left, Variable) and isinstance(right.left, Variable) and
                left.name == right.left.name):
            new_exp = BinaryOperator("+", right.right, Literal(1)).simplify()
            return BinaryOperator("^", left, new_exp).simplify()

        # General Term Combination
        num_factors, den_factors = self._flatten_mul_div([left, right])
        return self._combine_mul_div_factors(num_factors, den_factors)

    def _simplify_sub(self, left: Node, right: Node) -> Node:
        """Handles simplification for Subtraction (-)."""
        # Identity: x - 0 = x
        if self._is_literal(right, 0):
            return left

        # Identity: x - x = 0
        if left.to_string() == right.to_string():
            return Literal(0)

        # Constant Folding: handled in main simplify, but specific literal subtraction here
        if isinstance(left, Literal) and isinstance(right, Literal):
            return Literal(left.value - right.value)

        # Double Negation: x - (-y) -> x + y
        if self._is_unary(right, "-"):
            return BinaryOperator("+", left, right.operand).simplify()

        # Combine Terms: convert subtraction to addition of negative terms
        terms_left = self._flatten_add([left])
        terms_right = self._flatten_add([right])
        all_terms = terms_left.copy()

        for t in terms_right:
            if isinstance(t, Literal):
                all_terms.append(Literal(-t.value))
            else:
                all_terms.append(BinaryOperator("*", Literal(-1), t))

        return self._combine_add_terms(all_terms)

    def _simplify_div(self, left: Node, right: Node) -> Node:
        """Handles simplification for Division (/)."""
        # Identity: 0 / x = 0
        if self._is_literal(left, 0):
            return Literal(0)
        # Identity: x / 1 = x
        if self._is_literal(right, 1):
            return left

        # Exponent Rule: x^a / x^b -> x^(a-b)
        if (isinstance(left, BinaryOperator) and left.op == "^" and
                isinstance(right, BinaryOperator) and right.op == "^" and
                left.left.to_string() == right.left.to_string()):
            new_exp = BinaryOperator("-", left.right, right.right).simplify()
            return BinaryOperator("^", left.left.simplify(), new_exp).simplify()

        # Trig Identities: sin/cos = tan, cos/sin = cot
        if self._is_unary(left, 'sin') and self._is_unary(right, 'cos'):
            if left.operand.to_string() == right.operand.to_string():
                return UnaryOperator('tan', left.operand.copy())

        if self._is_unary(left, 'cos') and self._is_unary(right, 'sin'):
            if left.operand.to_string() == right.operand.to_string():
                return UnaryOperator('cot', left.operand.copy())

        # General Factor Combination
        num_factors, den_factors = self._flatten_mul_div([left])
        r_num, r_den = self._flatten_mul_div([right])
        num_factors.extend(r_den)  # Divide by fraction = multiply by reciprocal
        den_factors.extend(r_num)
        return self._combine_mul_div_factors(num_factors, den_factors)

    def _simplify_pow(self, left: Node, right: Node) -> Node:
        """Handles simplification for Power (^)."""
        # Power of Power: (x^a)^b -> x^(a*b)
        if isinstance(self.left, BinaryOperator) and self.left.op == "^":
            new_exponent = BinaryOperator('*', self.left.right, right).simplify()
            return BinaryOperator('^', self.left.left, new_exponent).simplify()

        # Sqrt Power: sqrt(x)^2 -> x
        if self._is_unary(left, "sqrt") and self._is_literal(right, 2):
            return left.operand.copy()

        # Square expansion: (A +/- B)^2 -> A^2 +/- 2AB + B^2
        if self._is_literal(right, 2) and isinstance(left, BinaryOperator) and left.op in ("+", "-"):
            a, b = left.left, left.right
            a_sq = BinaryOperator("^", a.copy(), Literal(2)).simplify()
            b_sq = BinaryOperator("^", b.copy(), Literal(2)).simplify()
            two_ab = BinaryOperator("*", Literal(2), BinaryOperator("*", a.copy(), b.copy())).simplify()

            if left.op == "+":
                result = BinaryOperator("+", BinaryOperator("+", a_sq, two_ab), b_sq)
            else:
                result = BinaryOperator("+", BinaryOperator("-", a_sq, two_ab), b_sq)
            return result.simplify()

        # Identity: x^1 = x
        if self._is_literal(right, 1):
            return left

        # Identity: x^0 = 1 (except 0^0 which is handled by Literal eval or 1)
        if self._is_literal(right, 0):
            if self._is_literal(left, 0):
                return BinaryOperator("^", left, right)  # 0^0 undefined-ish
            return Literal(1)

        # Identity: 0^x = 0 (if x != 0)
        if self._is_literal(left, 0) and not self._is_literal(right, 0):
            return Literal(0)

        return BinaryOperator("^", left, right)

    @staticmethod
    def _is_literal(node: Node, value: float) -> bool:
        """Helper to check if a node is a Literal with a specific value."""
        return isinstance(node, Literal) and node.value == value

    @staticmethod
    def _is_unary(node: Node, op: str) -> bool:
        """Helper to check if a node is a specific UnaryOperator."""
        return isinstance(node, UnaryOperator) and node.op == op

    @staticmethod
    def _is_trig_identity(left: Node, right: Node) -> bool:
        """Checks for sin^2(x) + cos^2(x) pattern."""

        def is_func_squared(node, func_name):
            return (isinstance(node, BinaryOperator) and node.op == "^" and
                    isinstance(node.left, UnaryOperator) and node.left.op == func_name and
                    isinstance(node.right, Literal) and node.right.value == 2)

        return ((is_func_squared(left, "sin") and is_func_squared(right, "cos")) or
                (is_func_squared(left, "cos") and is_func_squared(right, "sin")))

    def _flatten_add(self, parts: Iterable[Node]) -> List[Node]:
        """Flattens nested addition/subtraction into a list of terms."""
        out: List[Node] = []
        for p in parts:
            if isinstance(p, BinaryOperator) and p.op == "+":
                out.extend(self._flatten_add([p.left, p.right]))
            elif isinstance(p, BinaryOperator) and p.op == "-":
                left_terms = self._flatten_add([p.left])
                out.extend(left_terms)
                right_terms = self._flatten_add([p.right])
                for t in right_terms:
                    # Distribute negation
                    if isinstance(t, Literal):
                        out.append(Literal(-t.value))
                    elif isinstance(t, BinaryOperator) and t.op == "*" and isinstance(t.left, Literal):
                        out.append(BinaryOperator("*", Literal(-t.left.value), t.right))
                    else:
                        out.append(BinaryOperator("*", Literal(-1), t))
            else:
                out.append(p)
        return out

    def _flatten_mul_div(self, parts: Iterable[Node]) -> tuple[List[Node], List[Node]]:
        """Flattens nested multiplication/division into numerator and denominator lists."""
        numerators = []
        denominators = []
        for p in parts:
            if isinstance(p, BinaryOperator) and p.op == "*":
                n, d = self._flatten_mul_div([p.left, p.right])
                numerators.extend(n)
                denominators.extend(d)
            elif isinstance(p, BinaryOperator) and p.op == "/":
                n_left, d_left = self._flatten_mul_div([p.left])
                n_right, d_right = self._flatten_mul_div([p.right])
                numerators.extend(n_left)
                numerators.extend(d_right)  # division flips the right side
                denominators.extend(d_left)
                denominators.extend(n_right)
            else:
                numerators.append(p)
        return numerators, denominators

    @staticmethod
    def _get_base_and_exponent(node: Node) -> tuple[Node, Node]:
        """Extracts base and exponent. x -> (x, 1), x^2 -> (x, 2)."""
        if isinstance(node, BinaryOperator) and node.op == "^":
            return node.left, node.right
        return node, Literal(1)

    def _combine_mul_div_factors(self, num_factors: List[Node], den_factors: List[Node]) -> Node:
        """Main orchestration method for combining multiplication/division factors."""
        # 1. Analyze factors to separate numeric values and powers
        numeric_val, power_map, base_map = self._analyze_factors(num_factors, den_factors)

        # 2. Build lists of terms for numerator and denominator based on powers
        num_terms, den_terms = self._build_term_lists(power_map, base_map)

        # 3. Construct the final AST node
        return self._construct_fraction(numeric_val, num_terms, den_terms)

    def _analyze_factors(self, num_factors: List[Node], den_factors: List[Node]) -> (
            tuple)[float, Dict[str, float], Dict[str, Node]]:
        """
        Iterates through factors to calculate the combined numeric coefficient
        and map bases to their total exponents.
        """
        power_map = {}
        base_map = {}
        numeric_num = 1.0
        numeric_den = 1.0

        def process_list(factors, sign):
            nonlocal numeric_num, numeric_den
            for factor in factors:
                base, exp = self._get_base_and_exponent(factor)

                # Case 1: Numeric literal coefficients (e.g., 2 * x)
                if isinstance(base, Literal) and isinstance(exp, Literal) and exp.value == 1:
                    if sign == 1:
                        numeric_num *= base.value
                    else:
                        numeric_den *= base.value
                    continue

                # Case 2: Non-literal exponent (e.g., x^y) - treat whole term as base
                if not isinstance(exp, Literal):
                    key = factor.to_string()
                    base_map[key] = factor
                    power_map[key] = power_map.get(key, 0) + sign
                    continue

                # Case 3: Standard power combination (e.g., x^2 * x^3 -> x^5)
                key = base.to_string()
                base_map[key] = base
                power_map[key] = power_map.get(key, 0) + sign * exp.value

        process_list(num_factors, 1)
        process_list(den_factors, -1)

        # Avoid division by zero in coefficient calculation
        if numeric_den == 0:
            raise ZeroDivisionError("Division by zero detected during simplification.")

        return (numeric_num / numeric_den), power_map, base_map

    @staticmethod
    def _build_term_lists(power_map: Dict[str, float], base_map: Dict[str, Node]) \
            -> tuple[List[Node], List[Node]]:
        """Converts the power map back into lists of AST nodes for numerator and denominator."""
        num_terms = []
        den_terms = []

        for key in sorted(power_map.keys()):
            p = power_map[key]
            if abs(p) < 1e-12:  # Ignore zero powers (x^0 = 1)
                continue

            base = base_map[key]
            # Determine target list based on sign of exponent
            target_list = num_terms if p > 0 else den_terms
            abs_p = abs(p)

            # Optimization: x^1 -> x
            if abs(abs_p - 1) < 1e-12:
                target_list.append(base)
            else:
                target_list.append(BinaryOperator("^", base, Literal(abs_p)))

        return num_terms, den_terms

    @staticmethod
    def _construct_fraction(numeric_val: float, num_terms: List[Node], den_terms: List[Node]) -> Node:
        """Assembles the final AST node from the numeric coefficient and term lists."""
        # Handle numeric precision (e.g., 2.0 -> 2)
        if abs(numeric_val - round(numeric_val)) < 1e-10:
            numeric_val = int(round(numeric_val))

        # Handle zero coefficient early
        if numeric_val == 0:
            return Literal(0)

        # Add numeric coefficient to numerator if it's not 1
        if numeric_val != 1:
            num_terms.insert(0, Literal(numeric_val))

        # Construct Numerator Node
        if not num_terms:
            numerator = Literal(1)
        else:
            numerator = num_terms[0]
            for t in num_terms[1:]:
                numerator = BinaryOperator("*", numerator, t)

        # Construct Denominator Node (if exists)
        if not den_terms:
            return numerator

        denominator = den_terms[0]
        for t in den_terms[1:]:
            denominator = BinaryOperator("*", denominator, t)

        return BinaryOperator("/", numerator, denominator)

    def _combine_add_terms(self, terms: List[Node]) -> Node:
        """Combines like terms in addition (e.g., 2x + 3x -> 5x)."""
        coef_map = {}
        base_map = {}

        for t in terms:
            coef, base = self._decompose_term(t)
            c = coef.value if isinstance(coef, Literal) else 1
            key = base.to_string()
            base_map[key] = base
            coef_map[key] = coef_map.get(key, 0) + c

        result_terms = []
        for key in sorted(coef_map.keys()):
            c = coef_map[key]
            if abs(c) < 1e-12:
                continue

            base = base_map[key]
            if isinstance(base, Literal):
                result_terms.append(Literal(c))
            else:
                if c == 1:
                    result_terms.append(base)
                elif c == -1:
                    result_terms.append(BinaryOperator("*", Literal(-1), base))
                else:
                    result_terms.append(BinaryOperator("*", Literal(c), base))

        if not result_terms:
            return Literal(0)

        result = result_terms[0]
        for t in result_terms[1:]:
            result = BinaryOperator("+", result, t)

        return result

    def _decompose_term(self, term: Node) -> tuple[Node, Node]:
        """Separates coefficient from the symbolic part (e.g. 2x -> (2, x))."""
        if isinstance(term, Literal):
            return Literal(term.value), Literal(1)

        if isinstance(term, BinaryOperator) and term.op == "*":
            factors = self._flatten_mul([term.left, term.right])
            coeff = 1.0
            non_literals: List[Node] = []

            for f in factors:
                if isinstance(f, Literal):
                    coeff *= f.value
                else:
                    non_literals.append(f)

            if not non_literals:
                return Literal(coeff), Literal(1)

            # Rebuild symbol part
            base = non_literals[0]
            for f in non_literals[1:]:
                base = BinaryOperator("*", base, f)

            # Format coefficient
            if abs(coeff - round(coeff)) < 1e-10:
                coeff_node = Literal(int(round(coeff)))
            else:
                coeff_node = Literal(coeff)
            return coeff_node, base

        return Literal(1), term

    def _flatten_mul(self, parts: Iterable[Node]) -> List[Node]:
        """Recursively flattens nested multiplication nodes."""
        out: List[Node] = []
        for p in parts:
            if isinstance(p, BinaryOperator) and p.op == "*":
                out.extend(self._flatten_mul([p.left, p.right]))
            else:
                out.append(p)
        return out
