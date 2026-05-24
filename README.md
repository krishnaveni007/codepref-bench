# CodePref-Bench

A benchmark for evaluating **preference-aligned code generation**.

Current benchmarks (HumanEval, SWE-bench) measure functional correctness only.  
CodePref-Bench measures whether a coding agent produces code that matches *how* a user wants a task done — not just whether it works.

---

## The core idea

Every coding task has a latent preference vector **θ\*** encoding the user's intent:

```python
theta_star = {
    "deps":           "stdlib_only",     # vs allow third-party
    "error_handling": "explicit",        # vs bare try/except
    "style":          "functional",      # vs imperative
    "verbosity":      "terse",           # vs verbose with comments
}
```

The same task is run under three **visibility conditions**:

| Condition | What the agent sees |
|-----------|---------------------|
| `hidden`  | Bare task description |
| `partial` | Task + 1-2 preference hints |
| `revealed`| Task + full θ\* as explicit requirements |

This would let us decompose failures:

- **Capability failure**: model fails even when θ\* is fully revealed
- **Preference failure**: model fails when hidden, passes when revealed
- **Aligned**: passes in all conditions

---

## Reward function

```
R = 0.5 * functional_score
  + 0.5 * preference_score
  - 0.1 * num_questions
```

All graders are **programmatic** — AST-based, no LLM judge.

---

## Project structure

```
codepref-bench/
├── tasks/
│   └── tasks.json          # 5 tasks with θ*, prompts, functional tests
├── graders/
│   └── graders.py          # Preference graders (deps, error_handling, style, verbosity)
│                           # + functional correctness via pytest subprocess
├── env/
│   └── env.py              # Gym-style RL environment + UserSimulator
├── results/
│   └── results.json        # Output from run_eval.py
├── run_eval.py             # Main eval script
└── README.md
```

---

## Quickstart

```bash
pip install openai pytest

export OPENAI_API_KEY=sk-...

# Run one task, one visibility condition
python run_eval.py --task timestamp_extractor --visibility hidden

# Run one task, all three conditions (see the visibility gap)
python run_eval.py --task timestamp_extractor

# Run full benchmark
python run_eval.py

# Use a different model (default: gpt-4o)
python run_eval.py --model gpt-4o-mini
python run_eval.py --model o1

# Interactive mode: manually step through an environment
python run_eval.py --task timestamp_extractor --interactive
```

---

## Example output

```
────────────────────────────────────────────────────
CODEPREF-BENCH RESULTS
════════════════════════════════════════════════════════════════════════════════
Task                      Visibility    Func  Deps  ErrH Style  Verb  Pref   Reward  Qs
────────────────────────────────────────────────────────────────────────────────
timestamp_extractor       hidden           ✓     ✗     ✗     ✗     ✓  0.50   0.5000   1
timestamp_extractor       partial          ✓     ✓     ✗     ✓     ✓  0.75   0.6250   1
timestamp_extractor       revealed         ✓     ✓     ✓     ✓     ✓  1.00   0.8500   1

VISIBILITY GAP ANALYSIS
────────────────────────────────────────────────────────────
  timestamp_extractor       PREFERENCE FAILURE (gap=0.50)
```

The visibility gap tells you: the model *can* write the right code when told what the user wants. It just doesn't know without being told. That's a preference inference problem, not a coding problem.

---

## Graders

All graders are in `graders/graders.py`. They return a `GradeResult(score, passed, reason)`.

| Dimension | What it checks | Method |
|-----------|---------------|--------|
| `deps` | Only stdlib imports? | `ast.parse()` + allowlist |
| `error_handling` | Explicit exceptions, no bare `except`? | AST `ExceptHandler` inspection |
| `style` | Functional (comprehensions) vs imperative (for loops)? | AST node counting |
| `verbosity` | Docstring present? Inline comments? | AST + regex |
| `functional` | Does the code pass tests? | `pytest` subprocess |

---

## RL Environment

`env/env.py` implements a gym-style environment:

```python
from env.env import CodePrefEnv
import json

task = json.load(open("tasks/tasks.json"))[0]
env = CodePrefEnv(task, visibility="hidden")
obs = env.reset()

# Agent asks a question
obs, reward, done, info = env.step({
    "type": "question",
    "content": "Should I use third-party libraries?"
})
# info["user_response"] -> "Please keep it to the standard library."
# info["revealed_dimension"] -> "deps"

# Agent submits code
obs, reward, done, info = env.step({
    "type": "submit",
    "content": "import re\ndef extract_timestamps(filepath):\n    ..."
})
# info["reward"] -> 0.75
# info["preference_score"] -> 0.75
# info["functional_score"] -> 1.0
```

---

## Research Questions

This project operationalizes three research questions from *Inferring User Intent from Interaction*:

| RQ | How this project addresses it |
|----|-------------------------------|
| RQ1: Disentangle preference error from capability error | Visibility gap: `revealed_score - hidden_score` |
| RQ2: When does a task require interaction? | Measure whether asking questions actually improves preference score |
| RQ3: How should the agent learn θ\*? | UserSimulator provides a controlled oracle for testing preference inference |

---