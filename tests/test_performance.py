import time
import sys
import statistics
from typing import Callable
import pytest
from src.engine.Expression import parse


sys.setrecursionlimit(5000)


def generate_linear_expression(n: int) -> str:
    """
    Generates a long linear expression: x + x + ... + x (n times).
    Tests parsing and simplification of flat structures (lists of terms).
    """
    return " + ".join(["x"] * n)


def generate_nested_expression(n: int) -> str:
    """
    Generates a deeply nested expression: sin(sin(...(x)...)) (n times).
    Tests parsing and differentiation of deep tree structures (chain rule).
    """
    expr = "x"
    for _ in range(n):
        expr = f"sin({expr})"
    return expr


def benchmark(func: Callable, iterations: int = 5) -> float:
    """Runs a function multiple times and returns the average execution time."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    return statistics.mean(times)


@pytest.mark.performance
class TestPerformance:
    """
    Performance tests to measure algorithmic complexity.
    Run with 'pytest -s tests/test_performance.py' to see the output table.
    """

    # Sizes of input to test.
    SIZES = [10, 50, 100, 200, 500]

    def test_parsing_complexity(self):
        """Measures how parsing time scales with expression length."""
        print("\n\n--- Parsing Performance (Linear Sum: x + x + ... + x) ---")
        print(f"{'N Terms':<10} | {'Avg Time (s)':<15} | {'Items/sec':<15}")
        print("-" * 45)

        for n in self.SIZES:
            expr_str = generate_linear_expression(n)

            # Measure parsing time
            avg_time = benchmark(lambda expr=expr_str: parse(expr))

            items_per_sec = n / avg_time if avg_time > 0 else 0
            print(f"{n:<10} | {avg_time:<15.6f} | {items_per_sec:<15.0f}")

    def test_simplification_complexity(self):
        """
        Measures simplification performance (combining like terms).
        Input: x + x + ... + x -> simplifies to N*x.
        This stresses the _collect_terms and _combine_add_terms logic.
        """
        print("\n\n--- Simplification Performance (Combining N terms) ---")
        print(f"{'N Terms':<10} | {'Avg Time (s)':<15}")
        print("-" * 30)

        for n in self.SIZES:
            expr = parse(generate_linear_expression(n))

            # Measure simplify() time
            avg_time = benchmark(expr.simplify)

            print(f"{n:<10} | {avg_time:<15.6f}")

    def test_differentiation_complexity(self):
        """
        Measures differentiation performance on deep trees (Chain Rule).
        Input: sin(sin(...(x)...)).
        This creates a massive output expression due to the chain rule (product of cosines).
        """
        print("\n\n--- Differentiation Performance (Chain Rule Depth) ---")
        print(f"{'Depth':<10} | {'Avg Time (s)':<15}")
        print("-" * 30)

        # Use smaller sizes for differentiation because the output grows fast
        diff_sizes = [10, 50, 100, 200]

        for n in diff_sizes:
            expr = parse(generate_nested_expression(n))

            # Measure diff() time
            avg_time = benchmark(lambda e=expr: e.diff('x'))

            print(f"{n:<10} | {avg_time:<15.6f}")

    def test_evaluation_complexity(self):
        """Measures numerical evaluation performance."""
        print("\n\n--- Evaluation Performance (Evaluating N terms) ---")
        print(f"{'N Terms':<10} | {'Avg Time (s)':<15}")
        print("-" * 30)

        for n in self.SIZES:
            # x + x + ...
            expr = parse(generate_linear_expression(n))
            values = {'x': 1.5}

            # Measure evaluate() time
            avg_time = benchmark(lambda: expr.evaluate(values))

            print(f"{n:<10} | {avg_time:<15.6f}")

    def test_substitution_complexity(self):
        """
        Measures substitution performance.
        Scenario: Replace 'x' with 'y + 1' in a long expression.
        This tests the overhead of traversing the tree and creating new nodes.
        """
        print("\n\n--- Substitution Performance (Replacing x -> y + 1 in N terms) ---")
        print(f"{'N Terms':<10} | {'Avg Time (s)':<15}")
        print("-" * 30)

        # Substitution allows substituting with another expression, not just numbers
        sub_target = "y + 1"

        for n in self.SIZES:
            # x + x + ... + x
            expr = parse(generate_linear_expression(n))

            # Measure substitute() time
            # We use simplify=False to measure pure substitution cost without simplification overhead
            avg_time = benchmark(lambda e=expr: e.substitute({'x': sub_target}, simplify=False))

            print(f"{n:<10} | {avg_time:<15.6f}")


if __name__ == "__main__":
    t = TestPerformance()
    t.test_parsing_complexity()
    t.test_simplification_complexity()
    t.test_differentiation_complexity()
    t.test_substitution_complexity()
    t.test_evaluation_complexity()
