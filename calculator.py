"""
calculator.py — Safe arithmetic solver for StudyBuddy
=====================================================

Lets the chatbot answer natural-language maths like a mini calculator:

  * ``what is 12 * 8``          → 96
  * ``calculate 15% of 200``    → 30
  * ``square root of 144``      → 12
  * ``2 to the power 10``       → 1024
  * ``(2 + 3) * 4``             → 20

Security
--------
Expressions are parsed with :mod:`ast` and only *constant* arithmetic nodes
(numbers, + - * / % **, unary minus) are accepted. Anything else — names,
calls, attributes, imports — is rejected, so the solver can never execute
arbitrary code. If an expression cannot be solved, :meth:`solve` returns
``None`` and the caller simply routes the message through the normal flow.
"""

from __future__ import annotations

import ast
import re

# Word -> operator translations for natural-language arithmetic.
_WORD_OPERATORS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bplus\b", re.I), "+"),
    (re.compile(r"\b(?:minus|take away)\b", re.I), "-"),
    (re.compile(r"\b(?:times|multiplied by)\b", re.I), "*"),
    (re.compile(r"\b(?:divided by|over)\b", re.I), "/"),
    (re.compile(r"\bto the power of\b", re.I), "**"),
    (re.compile(r"\bto the power\b", re.I), "**"),
    (re.compile(r"\bpower of\b", re.I), "**"),
    (re.compile(r"\bwhat is\b", re.I), ""),
    (re.compile(r"\bwhats\b", re.I), ""),
    (re.compile(r"\bwhat's\b", re.I), ""),
    (re.compile(r"\bcalculate\b", re.I), ""),
    (re.compile(r"\bcompute\b", re.I), ""),
    (re.compile(r"\bsolve\b", re.I), ""),
    (re.compile(r"\bwork out\b", re.I), ""),
    (re.compile(r"\bequals?\b", re.I), "="),
]

# A "×" or letter "x" between two numbers is multiplication: 12x8 → 12*8.
_MULTIPLY_BETWEEN = re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)", re.I)
_SQRT = re.compile(
    r"(?:square root of|sqrt)\s*\(?(\d+(?:\.\d+)?)\)?", re.I
)
# "15% of 200" is a percentage. The "of" is REQUIRED so that the modulo
# operator ("100 % 7") is never misinterpreted as a percentage.
_PERCENT_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.I
)
# Spoken form: "15 percent of 200" -> "(15/100)*200"
_PERCENT_WORD_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*percent\s*of\s*(\d+(?:\.\d+)?)", re.I
)
_HAS_DIGIT = re.compile(r"\d")
_OPERATOR_HINT = re.compile(r"[-+*/%^()]|\b(plus|minus|times|divided|multiply|sqrt|root|percent|power)\b", re.I)
_EQUATION_GUARD = re.compile(r"[a-wyzA-WYZ]")  # any letter that isn't 'x' as operator

# Node types permitted by the evaluator.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)


class MathSolver:
    """Safe, natural-language-friendly arithmetic solver."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_math_query(self, text: str) -> bool:
        """Best-effort check: does this message look like arithmetic?

        A query counts as maths when it contains at least one digit *and*
        an arithmetic hint (operator character or a maths word).
        """
        if not text:
            return False
        if not _HAS_DIGIT.search(text):
            return False
        return bool(_OPERATOR_HINT.search(text))

    def solve(self, text: str) -> str | None:
        """Return a human-friendly answer string, or ``None`` if unsolvable.

        Example
        -------
        >>> MathSolver().solve("what is 15% of 200")
        '30'
        """
        try:
            expression = self._to_expression(text)
            value = self._evaluate(expression)
        except (ValueError, ZeroDivisionError, SyntaxError, TypeError, OverflowError):
            return None
        return self._format(value)

    # ------------------------------------------------------------------
    # Translation: natural language -> arithmetic expression
    # ------------------------------------------------------------------
    def _to_expression(self, text: str) -> str:
        expr = text.strip()
        # Percentage: "15% of 200" -> "(15/100)*200"
        expr = _PERCENT_OF.sub(lambda m: f"({m.group(1)}/100)*{m.group(2)}", expr)
        # Spoken percent: "15 percent of 200" -> "(15/100)*200"
        expr = _PERCENT_WORD_OF.sub(lambda m: f"({m.group(1)}/100)*{m.group(2)}", expr)
        # Square root: "sqrt 144" -> "(144**0.5)"
        expr = _SQRT.sub(r"(\1**0.5)", expr)
        # Spoken operator words -> symbols
        for pattern, symbol in _WORD_OPERATORS:
            expr = pattern.sub(symbol, expr)
        # × / x-between-digits -> *
        expr = _MULTIPLY_BETWEEN.sub("*", expr)
        # Remove commas in numbers (1,000 -> 1000)
        expr = re.sub(r"(?<=\d),(?=\d)", "", expr)
        # Caret as power
        expr = expr.replace("^", "**")
        # Drop the question mark / trailing "?"
        expr = expr.replace("?", "").strip()
        # Safety: reject anything that still contains non-maths letters.
        if _EQUATION_GUARD.search(expr):
            raise ValueError("non-arithmetic content")
        return expr

    # ------------------------------------------------------------------
    # Safe evaluation via the AST
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate(expression: str) -> float:
        tree = ast.parse(expression, mode="eval")

        def check(node: ast.AST) -> float:
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(f"disallowed node: {type(node).__name__}")
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)):
                    raise ValueError("non-numeric constant")
                return float(node.value)
            if isinstance(node, ast.BinOp):
                left = check(node.left)
                right = check(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    if right == 0:
                        raise ZeroDivisionError
                    return left / right
                if isinstance(node.op, ast.Mod):
                    if right == 0:
                        raise ZeroDivisionError
                    return left % right
                if isinstance(node.op, ast.Pow):
                    return left ** right
                raise ValueError("unknown binary operator")
            if isinstance(node, ast.UnaryOp):
                operand = check(node.operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return +operand
                raise ValueError("unknown unary operator")
            raise ValueError("unsupported node")

        return check(tree.body)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6f}".rstrip("0").rstrip(".")
