"""
run_eval.py
Run CodePref-Bench evaluation against a frontier model.

Usage:
    python run_eval.py                          # all tasks, all visibilities
    python run_eval.py --task timestamp_extractor
    python run_eval.py --task timestamp_extractor --visibility hidden
    python run_eval.py --model gpt-4o
    python run_eval.py --interactive             # step through one task manually

Requires: OPENAI_API_KEY in environment.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from env.env import CodePrefEnv
from graders.graders import grade_all, preference_score, total_reward


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a coding agent. You will be given a coding task.

You may ask the user at most 3 clarifying questions before writing code. 
Ask questions only when genuinely uncertain about requirements.

When you are ready to submit code, output ONLY the Python code block:
```python
<your code here>
```

Do not include explanations outside the code block in your final submission.
"""

QUESTION_PROMPT = """Task: {task_prompt}

{conversation_context}

You can either:
1. Ask a clarifying question (output: QUESTION: <your question>)
2. Submit your solution (output: ```python\\n<code>\\n```)

Respond now."""


def extract_code(text: str) -> str | None:
    """Extract Python code from a markdown code block."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if it looks like raw code
    if "def " in text and "```" not in text:
        return text.strip()
    return None


def extract_question(text: str) -> str | None:
    """Extract a clarifying question from agent output."""
    if text.strip().startswith("QUESTION:"):
        return text.strip()[9:].strip()
    # Heuristic: ends with ?
    lines = [l.strip() for l in text.strip().splitlines() if l.strip().endswith("?")]
    return lines[0] if lines else None


import re


def run_agent_episode(
    task: dict,
    visibility: str,
    client: OpenAI,
    model: str,
    verbose: bool = True,
) -> dict:
    """
    Run one episode: agent interacts with env until code is submitted.
    Returns the info dict from the final step.
    """
    env = CodePrefEnv(task, visibility=visibility)
    obs = env.reset()

    messages = []
    max_agent_turns = 6  # question + answer pairs + final submission

    for turn in range(max_agent_turns):
        # Build conversation context for the agent
        conv_context = ""
        if obs["conversation"]:
            lines = []
            for msg in obs["conversation"]:
                prefix = "You asked" if msg["role"] == "agent" else "User said"
                lines.append(f"{prefix}: {msg['content']}")
            conv_context = "Previous conversation:\n" + "\n".join(lines) + "\n"

        prompt = QUESTION_PROMPT.format(
            task_prompt=obs["task_prompt"],
            conversation_context=conv_context,
        )
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        agent_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": agent_text})

        if verbose:
            print(f"\n  [Agent turn {turn+1}]: {agent_text[:200]}{'...' if len(agent_text)>200 else ''}")

        # Try to extract code first (submission)
        code = extract_code(agent_text)
        if code:
            obs, reward, done, info = env.step({"type": "submit", "content": code})
            if verbose:
                env.render()
            return info

        # Try to extract a question
        question = extract_question(agent_text)
        if question:
            obs, reward, done, info = env.step({"type": "question", "content": question})
            messages.append({"role": "user", "content": info["user_response"]})
            if obs["done"]:
                break
        else:
            # No question, no code — treat entire response as a submission attempt
            obs, reward, done, info = env.step({"type": "submit", "content": agent_text})
            if verbose:
                env.render()
            return info

    # Force submission with whatever we have
    obs, reward, done, info = env.step({"type": "submit", "content": ""})
    return info


# ── Results formatting ────────────────────────────────────────────────────────

def print_results_table(all_results: list[dict]):
    """Print a comparison table across visibility conditions."""
    print("\n" + "=" * 80)
    print("CODEPREF-BENCH RESULTS")
    print("=" * 80)
    print(f"{'Task':<25} {'Visibility':<12} {'Func':>6} {'Deps':>6} {'ErrH':>6} {'Style':>6} {'Verb':>6} {'Pref':>6} {'Reward':>8} {'Qs':>4}")
    print("-" * 80)

    for r in all_results:
        gr = r.get("grade_results", {})
        func  = gr.get("functional",     {}).get("score", 0)
        deps  = gr.get("deps",           {}).get("score", 0)
        errh  = gr.get("error_handling", {}).get("score", 0)
        style = gr.get("style",          {}).get("score", 0)
        verb  = gr.get("verbosity",      {}).get("score", 0)
        pref  = r.get("preference_score", 0)
        rew   = r.get("reward", 0)
        qs    = r.get("num_questions", 0)

        def fmt(v): return f"{'✓' if v==1.0 else '✗':>6}"

        print(
            f"{r['task_name']:<25} {r['visibility']:<12}"
            f"{fmt(func)}{fmt(deps)}{fmt(errh)}{fmt(style)}{fmt(verb)}"
            f"{pref:>6.2f}{rew:>8.4f}{qs:>4}"
        )

    print("=" * 80)

    # Summary: visibility gap analysis (the core RQ3 insight)
    print("\nVISIBILITY GAP ANALYSIS (RQ3: preference error vs capability error)")
    print("-" * 60)
    for task_name in set(r["task_name"] for r in all_results):
        task_results = {r["visibility"]: r for r in all_results if r["task_name"] == task_name}
        if "hidden" in task_results and "revealed" in task_results:
            hidden_pref   = task_results["hidden"]["preference_score"]
            revealed_pref = task_results["revealed"]["preference_score"]
            gap = revealed_pref - hidden_pref
            hidden_func   = task_results["hidden"].get("grade_results", {}).get("functional", {}).get("score", 0)
            revealed_func = task_results["revealed"].get("grade_results", {}).get("functional", {}).get("score", 0)

            if revealed_func == 0:
                diagnosis = "CAPABILITY FAILURE (fails even with full info)"
            elif gap > 0.25:
                diagnosis = f"PREFERENCE FAILURE (gap={gap:.2f})"
            else:
                diagnosis = f"OK (gap={gap:.2f})"

            print(f"  {task_name:<25}  {diagnosis}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def load_tasks(task_name: str | None = None) -> list[dict]:
    tasks_path = Path(__file__).parent / "tasks" / "tasks.json"
    with open(tasks_path) as f:
        tasks = json.load(f)
    if task_name:
        tasks = [t for t in tasks if t["name"] == task_name]
        if not tasks:
            print(f"Task '{task_name}' not found. Available: {[t['name'] for t in json.load(open(tasks_path))]}")
            sys.exit(1)
    return tasks


def interactive_mode(task: dict):
    """Manually step through an environment for debugging."""
    print(f"\nInteractive mode: {task['name']}")
    visibility = input("Visibility [hidden/partial/revealed]: ").strip() or "hidden"
    env = CodePrefEnv(task, visibility=visibility)
    obs = env.reset()

    print(f"\nTask prompt:\n{obs['task_prompt']}\n")
    print("Actions: 'q <question>' to ask, 'c' to submit code (paste until EOF), 'r' to render state\n")

    while not obs["done"]:
        cmd = input("> ").strip()
        if cmd.startswith("q "):
            obs, _, done, info = env.step({"type": "question", "content": cmd[2:]})
            print(f"User: {info['user_response']}")
            if info["revealed_dimension"]:
                print(f"[revealed: {info['revealed_dimension']}]")
        elif cmd == "c":
            print("Paste code (end with a line containing just 'END'):")
            lines = []
            while True:
                line = input()
                if line == "END":
                    break
                lines.append(line)
            code = "\n".join(lines)
            obs, reward, done, info = env.step({"type": "submit", "content": code})
            env.render()
            break
        elif cmd == "r":
            env.render()


def main():
    parser = argparse.ArgumentParser(description="CodePref-Bench evaluator")
    parser.add_argument("--task", type=str, default=None, help="Task name to run (default: all)")
    parser.add_argument("--visibility", type=str, default=None,
                        choices=["hidden", "partial", "revealed"],
                        help="Visibility condition (default: all three)")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--interactive", action="store_true",
                        help="Manually step through a task")
    parser.add_argument("--output", type=str, default="results/results.json",
                        help="Path to save results JSON")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    tasks = load_tasks(args.task)
    visibilities = [args.visibility] if args.visibility else ["hidden", "partial", "revealed"]

    if args.interactive:
        interactive_mode(tasks[0])
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    all_results = []
    for task in tasks:
        for vis in visibilities:
            print(f"\n{'─'*50}")
            print(f"Running: {task['name']} | visibility={vis}")
            print(f"{'─'*50}")
            try:
                info = run_agent_episode(task, vis, client, args.model, verbose=args.verbose)
                info["task_name"] = task["name"]
                info["task_id"] = task["id"]
                info["visibility"] = vis
                info["model"] = args.model
                all_results.append(info)
                time.sleep(0.5)  # Rate limit buffer
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "task_name": task["name"],
                    "task_id": task["id"],
                    "visibility": vis,
                    "model": args.model,
                    "error": str(e),
                    "grade_results": {},
                    "preference_score": 0,
                    "reward": 0,
                    "num_questions": 0,
                })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print_results_table(all_results)


if __name__ == "__main__":
    main()