"""
env.py
CodePref-Bench: Gym-style RL environment for preference-aligned code generation.

The environment models the paper's formulation:
  - theta* is the latent preference vector (known to the user simulator, hidden from the agent)
  - The agent can ask clarifying questions (actions) or submit code (terminal action)
  - Reward = functional_correctness + preference_alignment - question_penalty

Usage:
    env = CodePrefEnv(task, visibility="hidden")
    obs = env.reset()
    # Agent asks a question
    obs, reward, done, info = env.step({"type": "question", "content": "Should I use stdlib only?"})
    # Agent submits code
    obs, reward, done, info = env.step({"type": "submit", "content": "<python code>"})
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graders.graders import grade_all, preference_score, total_reward


# ── User Simulator ────────────────────────────────────────────────────────────

class UserSimulator:
    """
    Rule-based user simulator that answers clarifying questions about theta*.

    The simulator knows theta* and responds to questions that target a
    specific preference dimension. Off-target questions get a vague reply.
    This models indirectness from the ICLR paper.
    """

    DIMENSION_KEYWORDS = {
        "deps": [
            "library", "libraries", "package", "packages", "import", "imports",
            "dependency", "dependencies", "third.party", "third party", "stdlib",
            "standard library", "external",
        ],
        "error_handling": [
            "error", "errors", "exception", "exceptions", "handle", "handling",
            "raise", "raises", "try", "except", "fail", "failure",
        ],
        "style": [
            "style", "approach", "pattern", "functional", "imperative", "oop",
            "class", "loop", "loops", "comprehension", "lambda", "map", "filter",
        ],
        "verbosity": [
            "comment", "comments", "document", "documentation", "docstring",
            "verbose", "terse", "concise", "brief", "explain", "readable",
        ],
    }

    RESPONSES = {
        "deps": {
            "stdlib_only": "Please keep it to the standard library. No third-party packages.",
            "allow_third_party": "You can use third-party packages if helpful.",
        },
        "error_handling": {
            "explicit": "Be explicit about errors — use pattern matching rather than broad try/except blocks.",
            "raise_on_missing": "Raise a clear exception if something is missing or invalid. Don't return None silently.",
        },
        "style": {
            "functional": "I'd prefer a functional approach — list comprehensions, no classes.",
            "imperative": "Write it step by step with explicit loops. Easy to follow.",
        },
        "verbosity": {
            "terse": "Keep it short and clean. No comments needed.",
            "verbose": "Add a docstring and some inline comments so it's easy to read later.",
        },
    }

    VAGUE_RESPONSES = [
        "Just write whatever you think is best for the task.",
        "Use your judgment.",
        "I'm not sure what you mean — just write the function.",
        "Whatever works.",
    ]

    def __init__(self, theta_star: dict):
        self.theta = theta_star
        self._vague_idx = 0

    def respond(self, question: str) -> tuple[str, Optional[str]]:
        """
        Returns (response_text, revealed_dimension_or_None).
        If the question targets a known dimension, reveal it.
        Otherwise return a vague answer and reveal nothing.
        """
        q_lower = question.lower()
        for dim, keywords in self.DIMENSION_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                value = self.theta.get(dim)
                response = self.RESPONSES.get(dim, {}).get(value, "Use your judgment.")
                return response, dim

        # Off-target question
        response = self.VAGUE_RESPONSES[self._vague_idx % len(self.VAGUE_RESPONSES)]
        self._vague_idx += 1
        return response, None


# ── Environment ───────────────────────────────────────────────────────────────

@dataclass
class EnvState:
    task_id: str
    visibility: str
    theta_star: dict
    conversation: list = field(default_factory=list)
    revealed_dimensions: set = field(default_factory=set)
    num_questions: int = 0
    done: bool = False
    submitted_code: Optional[str] = None
    grade_results: Optional[dict] = None


class CodePrefEnv:
    """
    Gym-style environment for preference-aware code generation.

    Observation space: dict with keys:
        - task_prompt: str (varies by visibility)
        - conversation: list of {role, content} dicts
        - revealed_dimensions: list of dimensions the user has clarified

    Action space: dict with keys:
        - type: "question" | "submit"
        - content: str (the question text or code)

    Reward: float in [-inf, 1.0]
        = 0.5 * functional + 0.5 * preference_score - 0.1 * num_questions
    """

    MAX_TURNS = 10  # cap interaction rounds

    def __init__(self, task: dict, visibility: str = "hidden"):
        assert visibility in ("hidden", "partial", "revealed"), \
            "visibility must be 'hidden', 'partial', or 'revealed'"
        self.task = task
        self.visibility = visibility
        self.state: Optional[EnvState] = None
        self.user_sim: Optional[UserSimulator] = None

    def reset(self) -> dict:
        """Reset environment. Returns initial observation."""
        self.state = EnvState(
            task_id=self.task["id"],
            visibility=self.visibility,
            theta_star=self.task["theta_star"],
        )
        self.user_sim = UserSimulator(self.task["theta_star"])

        obs = self._get_obs()
        return obs

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Take an action.
        action = {"type": "question", "content": "..."} 
              or {"type": "submit",   "content": "<code>"}

        Returns: (observation, reward, done, info)
        """
        assert self.state is not None, "Call reset() first."
        assert not self.state.done, "Episode is done. Call reset()."

        action_type = action.get("type")
        content = action.get("content", "")

        if action_type == "question":
            return self._handle_question(content)
        elif action_type == "submit":
            return self._handle_submit(content)
        else:
            raise ValueError(f"Unknown action type: {action_type}. Use 'question' or 'submit'.")

    def _handle_question(self, question: str) -> tuple[dict, float, bool, dict]:
        self.state.num_questions += 1
        self.state.conversation.append({"role": "agent", "content": question})

        response, revealed_dim = self.user_sim.respond(question)
        if revealed_dim:
            self.state.revealed_dimensions.add(revealed_dim)

        self.state.conversation.append({"role": "user", "content": response})

        # Force termination if max turns reached
        if self.state.num_questions >= self.MAX_TURNS:
            self.state.done = True

        info = {
            "revealed_dimension": revealed_dim,
            "user_response": response,
            "num_questions": self.state.num_questions,
        }

        return self._get_obs(), 0.0, self.state.done, info

    def _handle_submit(self, code: str) -> tuple[dict, float, bool, dict]:
        self.state.submitted_code = code
        self.state.done = True

        # Grade the submission
        grade_results = grade_all(code, self.task)
        self.state.grade_results = grade_results

        reward = total_reward(grade_results, self.state.num_questions)
        pref = preference_score(grade_results)
        functional = grade_results["functional"].score

        info = {
            "grade_results": {
                dim: {
                    "score": r.score,
                    "passed": r.passed,
                    "reason": r.reason,
                    "expected": r.expected,
                }
                for dim, r in grade_results.items()
            },
            "functional_score": functional,
            "preference_score": pref,
            "reward": reward,
            "num_questions": self.state.num_questions,
            "revealed_dimensions": list(self.state.revealed_dimensions),
        }

        return self._get_obs(), reward, True, info

    def _get_obs(self) -> dict:
        return {
            "task_prompt": self.task["prompts"][self.visibility],
            "conversation": list(self.state.conversation),
            "revealed_dimensions": list(self.state.revealed_dimensions),
            "num_questions": self.state.num_questions,
            "done": self.state.done,
        }

    def render(self):
        """Print current state for debugging."""
        s = self.state
        print(f"\n{'='*60}")
        print(f"Task: {s.task_id}  |  Visibility: {s.visibility}")
        print(f"Questions asked: {s.num_questions}")
        print(f"Revealed dims: {s.revealed_dimensions}")
        if s.conversation:
            print("\nConversation:")
            for turn in s.conversation:
                prefix = "  Agent: " if turn["role"] == "agent" else "  User:  "
                print(prefix + turn["content"])
        if s.grade_results:
            print("\nGrade Results:")
            for dim, r in s.grade_results.items():
                mark = "✓" if r.passed else "✗"
                print(f"  [{mark}] {dim:20s} {r.reason}")
            reward = total_reward(s.grade_results, s.num_questions)
            print(f"\n  Reward: {reward:.4f}")
        print("="*60)