"""
graders.py
Programmatic preference graders for CodePref-Bench.
Each grader takes (code: str, task: dict) and returns a GradeResult.
No LLM judge anywhere - all checks are AST, regex, or execution-based.
"""

import ast
import re
import sys
import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GradeResult:
    dimension: str
    score: float          # 0.0 or 1.0
    passed: bool
    reason: str
    expected: str


# ── Stdlib allowlist ──────────────────────────────────────────────────────────

def _get_imports(code: str) -> list[str]:
    """Extract all top-level imported module names from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


# Python stdlib module names (subset covering common usage)
STDLIB_MODULES = {
    "abc", "ast", "asyncio", "builtins", "collections", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "decimal", "enum",
    "functools", "gc", "hashlib", "heapq", "html", "http", "io",
    "itertools", "json", "logging", "math", "operator", "os",
    "pathlib", "pickle", "pprint", "queue", "random", "re",
    "shutil", "signal", "socket", "sqlite3", "statistics", "string",
    "struct", "subprocess", "sys", "tempfile", "textwrap", "threading",
    "time", "traceback", "typing", "unittest", "urllib", "uuid",
    "warnings", "weakref", "xml", "zipfile", "zlib",
}


def grade_deps(code: str, task: dict) -> GradeResult:
    """
    Check whether the solution uses only stdlib imports.
    Grader: AST-based import extraction + allowlist check.
    """
    expected_deps = task["theta_star"]["deps"]
    imports = _get_imports(code)
    third_party = [m for m in imports if m not in STDLIB_MODULES and m != "solution"]

    if expected_deps == "stdlib_only":
        passed = len(third_party) == 0
        reason = (
            f"stdlib only - OK" if passed
            else f"found third-party imports: {third_party}"
        )
    else:
        # Future: handle "allow_third_party" preference
        passed = True
        reason = "no deps constraint"

    return GradeResult(
        dimension="deps",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason=reason,
        expected=expected_deps,
    )


# ── Error handling ────────────────────────────────────────────────────────────

def grade_error_handling(code: str, task: dict) -> GradeResult:
    """
    Check whether error handling matches the preference.
    - 'explicit': no bare except, uses specific exception types or re.search patterns
    - 'raise_on_missing': raises KeyError/ValueError/FileNotFoundError, not returns None
    Grader: AST-based exception handler inspection.
    """
    expected = task["theta_star"]["error_handling"]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return GradeResult("error_handling", 0.0, False, "syntax error in code", expected)

    handlers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
    ]
    raises = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
    ]

    if expected == "explicit":
        # Fail if there is a bare except (ExceptHandler with type=None)
        bare_excepts = [h for h in handlers if h.type is None]
        passed = len(bare_excepts) == 0
        reason = "no bare except - OK" if passed else f"found {len(bare_excepts)} bare except clause(s)"

    elif expected == "raise_on_missing":
        # Must have at least one raise statement
        passed = len(raises) > 0
        reason = "raises on error - OK" if passed else "no raise statements found; silent failure likely"

    else:
        passed = True
        reason = "no error handling constraint"

    return GradeResult(
        dimension="error_handling",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason=reason,
        expected=expected,
    )


# ── Style ─────────────────────────────────────────────────────────────────────

def grade_style(code: str, task: dict) -> GradeResult:
    """
    Check whether the solution matches the style preference.
    - 'functional': uses comprehensions / Counter / map / filter; no class
    - 'imperative': uses explicit for loops; avoids comprehensions for core logic
    Grader: AST-based node counting.
    """
    expected = task["theta_star"]["style"]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return GradeResult("style", 0.0, False, "syntax error in code", expected)

    comprehensions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp))
    ]
    for_loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
    ]
    class_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]

    if expected == "functional":
        # Functional: uses comprehensions or known functional builtins, no class
        has_functional = len(comprehensions) > 0
        has_class = len(class_defs) > 0
        passed = has_functional and not has_class
        reason = (
            "functional style - OK" if passed
            else (
                "no comprehensions/functional constructs found" if not has_functional
                else "uses class definition (not functional)"
            )
        )

    elif expected == "imperative":
        # Imperative: explicit for loops for core logic
        passed = len(for_loops) > 0
        reason = (
            "imperative style (explicit loops) - OK" if passed
            else "no explicit for loops found"
        )

    else:
        passed = True
        reason = "no style constraint"

    return GradeResult(
        dimension="style",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason=reason,
        expected=expected,
    )


# ── Verbosity ─────────────────────────────────────────────────────────────────

def grade_verbosity(code: str, task: dict) -> GradeResult:
    """
    Check whether the solution matches the verbosity preference.
    - 'terse': no inline comments, short functions
    - 'verbose': has docstring, has inline comments
    Grader: AST-based docstring detection + regex comment counting.
    """
    expected = task["theta_star"]["verbosity"]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return GradeResult("verbosity", 0.0, False, "syntax error in code", expected)

    # Count inline comments (# lines, not docstrings)
    comment_lines = [l for l in code.splitlines() if re.search(r"^\s*#", l)]
    inline_comments = len(comment_lines)

    # Check for docstring in any function/class
    has_docstring = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                has_docstring = True
                break

    total_lines = len([l for l in code.splitlines() if l.strip()])

    if expected == "terse":
        passed = inline_comments == 0
        reason = (
            "terse - no inline comments - OK" if passed
            else f"found {inline_comments} inline comment(s)"
        )

    elif expected == "verbose":
        passed = has_docstring and inline_comments >= 2
        reason = (
            "verbose - has docstring and comments - OK" if passed
            else (
                "missing docstring" if not has_docstring
                else f"only {inline_comments} inline comment(s), expected >= 2"
            )
        )

    else:
        passed = True
        reason = "no verbosity constraint"

    return GradeResult(
        dimension="verbosity",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason=reason,
        expected=expected,
    )


# ── Functional correctness ────────────────────────────────────────────────────

def grade_functional(code: str, task: dict) -> GradeResult:
    """
    Run the task's functional test in a subprocess sandbox.
    Returns pass/fail based on exit code.
    """
    # JSON already decodes \n escape sequences to real newlines when loading.
    # test_code is therefore already a valid multi-line Python script.
    test_code = task["functional_test"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write solution
        sol_path = os.path.join(tmpdir, "solution.py")
        with open(sol_path, "w") as f:
            f.write(code)

        # Write test - wrap in a function so pytest can collect it
        test_path = os.path.join(tmpdir, "test_solution.py")
        indented = "\n".join("    " + line for line in test_code.splitlines())
        full_test = (
            "import pytest, sys, os\n"
            "sys.path.insert(0, os.path.dirname(__file__))\n\n"
            "def test_preference():\n"
            + indented + "\n"
        )
        with open(test_path, "w") as f:
            f.write(full_test)

        # Run pytest silently
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-x", "-q", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=15,
        )

    passed = result.returncode == 0
    reason = "tests passed" if passed else result.stdout[-500:] + result.stderr[-200:]

    return GradeResult(
        dimension="functional",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason=reason.strip(),
        expected="all tests pass",
    )


# ── Master grader ─────────────────────────────────────────────────────────────

def grade_all(code: str, task: dict) -> dict[str, GradeResult]:
    """Run all graders and return a dict of dimension -> GradeResult."""
    return {
        "functional":     grade_functional(code, task),
        "deps":           grade_deps(code, task),
        "error_handling": grade_error_handling(code, task),
        "style":          grade_style(code, task),
        "verbosity":      grade_verbosity(code, task),
    }


def preference_score(results: dict[str, GradeResult]) -> float:
    """Preference alignment score: mean over non-functional dimensions."""
    pref_dims = ["deps", "error_handling", "style", "verbosity"]
    scores = [results[d].score for d in pref_dims if d in results]
    return sum(scores) / len(scores) if scores else 0.0


def total_reward(results: dict[str, GradeResult], num_questions: int = 0) -> float:
    """
    RL reward function.
    R = 0.5 * functional + 0.5 * preference_score - 0.1 * num_questions
    """
    functional = results.get("functional", GradeResult("functional", 0.0, False, "", "")).score
    pref = preference_score(results)
    question_penalty = 0.1 * num_questions
    return round(0.5 * functional + 0.5 * pref - question_penalty, 4)