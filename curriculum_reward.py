"""
curriculum_reward.py
Curriculum-shaped reward function for GRPO training.

Implements a 3-phase curriculum that gradually increases reward complexity:
  Phase 1 (0-20% of training): Survival — just complete waypoints
  Phase 2 (20-60%): Efficiency — add energy + thermal management
  Phase 3 (60-100%): Mastery — full 5-component rubric weights

The reward function replays the environment to the exact state described
in each prompt, executes the model's proposed action, and scores it.
No pre-existing dataset needed — all rewards come from live environment.

Usage:
    from curriculum_reward import CurriculumReward
    reward_fn = CurriculumReward(max_steps=500)
    # Pass reward_fn to GRPOTrainer as reward_funcs=[reward_fn]
"""

import re
import json
import csv
import os
from typing import Dict, Optional

from environment import SolarEVEnvironment
from models import Action


# ═══════════════════════════════════════════════════════════════════════════════
# State registry: maps prompt keys to environment replay info
# ═══════════════════════════════════════════════════════════════════════════════

_STATE_MAP: Dict[str, tuple] = {}


def register_state(key: str, task_id: str, seed: int, prior_actions: list):
    """Register a (task_id, seed, prior_actions) tuple for a prompt key.

    Args:
        key: Unique identifier embedded in the prompt (e.g. "flat_track_easy|42|3")
        task_id: Task to reset to.
        seed: Seed for stochastic perturbation.
        prior_actions: List of action dicts to replay before evaluating.
    """
    _STATE_MAP[key] = (task_id, seed, prior_actions)


def get_state_map_size() -> int:
    return len(_STATE_MAP)


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_action(text: str) -> Optional[Action]:
    """Parse the model's text output into an Action object.

    Handles common LLM output quirks:
      - Markdown code fences (```json ... ```)
      - Leading/trailing whitespace
      - Extra text before/after JSON
    """
    try:
        clean = text.strip()
        # Strip markdown fences
        clean = re.sub(r'```json\s*', '', clean)
        clean = re.sub(r'```\s*', '', clean)
        clean = clean.strip()
        # Find the JSON object
        start = clean.find('{')
        end = clean.rfind('}')
        if start == -1 or end == -1:
            return None
        json_str = clean[start:end + 1]
        d = json.loads(json_str)
        return Action(**d)
    except Exception:
        return None


def extract_key(text: str) -> Optional[str]:
    """Extract EPISODE_KEY from a formatted prompt string."""
    match = re.search(r'EPISODE_KEY=(\S+)', text)
    return match.group(1) if match else None


# ═══════════════════════════════════════════════════════════════════════════════
# Environment replay
# ═══════════════════════════════════════════════════════════════════════════════

def replay_and_evaluate(task_id: str, seed: int, prior_actions: list, model_action: Action) -> tuple:
    """Replay environment to a specific state and evaluate the model's action.

    Returns:
        (score, rubric_dict_or_None, is_done, violated)
    """
    env = SolarEVEnvironment()
    env.reset(task_id=task_id, seed=seed)

    # Replay prior actions to reach the current waypoint
    for pa in prior_actions:
        try:
            env.step(Action(**pa))
        except Exception:
            return 0.0, None, False, True

    # Execute the model's proposed action
    try:
        obs, reward = env.step(model_action)
        rubric = reward.rubric if reward.rubric else None
        violated = reward.is_done and not reward.is_success and reward.score < 0.01
        return reward.score, rubric, reward.is_done, violated
    except Exception:
        return 0.0, None, False, True


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum reward class
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumReward:
    """Callable reward function with curriculum phase tracking.

    Usage:
        reward_fn = CurriculumReward(max_steps=500)
        # In training callback: reward_fn.update_step(state.global_step)
        # GRPOTrainer calls: reward_fn(completions=..., prompts=...)
    """

    def __init__(self, max_steps: int = 500, log_dir: str = "results"):
        self.max_steps = max_steps
        self.current_step = 0
        self.log_dir = log_dir
        self._metrics_log = []

        os.makedirs(log_dir, exist_ok=True)
        self._csv_path = os.path.join(log_dir, "training_metrics.csv")
        self._csv_initialized = False

    def update_step(self, step: int):
        """Called by TrainerCallback to sync the current training step."""
        self.current_step = step

    @property
    def progress(self) -> float:
        return self.current_step / max(self.max_steps, 1)

    @property
    def phase_name(self) -> str:
        if self.progress < 0.2:
            return "survival"
        elif self.progress < 0.6:
            return "efficiency"
        else:
            return "mastery"

    def _get_weights(self) -> Dict[str, float]:
        """Compute curriculum-interpolated rubric weights."""
        p = self.progress

        if p < 0.2:
            # Phase 1: Survival — just complete waypoints
            return {"completion": 1.0, "energy": 0.0, "thermal": 0.0, "solar": 0.0, "time": 0.0}

        elif p < 0.6:
            # Phase 2: Efficiency — smooth transition
            t = (p - 0.2) / 0.4  # 0 → 1 within phase
            return {
                "completion": 1.0 - 0.5 * t,   # 1.0 → 0.5
                "energy": 0.3 * t,               # 0.0 → 0.3
                "thermal": 0.1 * t,              # 0.0 → 0.1
                "solar": 0.1 * t,                # 0.0 → 0.1
                "time": 0.0,
            }

        else:
            # Phase 3: Mastery — approach final rubric weights
            t = (p - 0.6) / 0.4  # 0 → 1 within phase
            return {
                "completion": 0.5 - 0.1 * t,   # 0.5 → 0.4
                "energy": 0.3 - 0.1 * t,        # 0.3 → 0.2
                "thermal": 0.1 + 0.05 * t,      # 0.1 → 0.15
                "solar": 0.1 + 0.05 * t,        # 0.1 → 0.15
                "time": 0.1 * t,                # 0.0 → 0.1
            }

    def _log_metrics(self, metrics: dict):
        """Log metrics to CSV for post-training plots."""
        metrics["step"] = self.current_step
        metrics["phase"] = self.phase_name
        self._metrics_log.append(metrics)

        # Write to CSV
        if not self._csv_initialized:
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
                writer.writeheader()
                writer.writerow(metrics)
            self._csv_initialized = True
        else:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
                writer.writerow(metrics)

    def __call__(self, completions: list, prompts: list = None, **kwargs) -> list:
        """Score model completions by replaying on the environment.

        Args:
            completions: List of generated text strings.
            prompts: List of formatted prompt strings (contains EPISODE_KEY).

        Returns:
            List of float rewards in [0.0, 1.0].
        """
        w = self._get_weights()
        rewards = []

        n_valid = 0
        n_violations = 0
        rubric_sums = {"completion": 0.0, "energy": 0.0, "thermal": 0.0, "solar": 0.0, "time": 0.0}
        raw_scores = []

        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts and i < len(prompts) else ""
            key = extract_key(prompt)

            if key is None or key not in _STATE_MAP:
                rewards.append(0.0)
                continue

            task_id, seed, prior_actions = _STATE_MAP[key]

            # Parse model output
            action = parse_action(completion)
            if action is None:
                rewards.append(0.0)
                continue

            n_valid += 1

            # Replay and evaluate
            score, rubric, is_done, violated = replay_and_evaluate(
                task_id, seed, prior_actions, action
            )

            if violated:
                n_violations += 1

            if rubric and is_done:
                # Final step: use curriculum-weighted rubric
                comp = rubric.get("completion_score", 0)
                ener = rubric.get("energy_efficiency_score", 0)
                ther = rubric.get("thermal_management_score", 0)
                sol = rubric.get("solar_utilization_score", 0)
                tim = rubric.get("time_performance_score", 0)

                curriculum_score = (
                    w["completion"] * comp +
                    w["energy"] * ener +
                    w["thermal"] * ther +
                    w["solar"] * sol +
                    w["time"] * tim
                )

                rubric_sums["completion"] += comp
                rubric_sums["energy"] += ener
                rubric_sums["thermal"] += ther
                rubric_sums["solar"] += sol
                rubric_sums["time"] += tim
            else:
                # Intermediate step: weighted step score
                curriculum_score = score

            raw_scores.append(score)
            rewards.append(max(0.0, min(1.0, curriculum_score)))

        # Log batch metrics
        if n_valid > 0:
            self._log_metrics({
                "mean_reward": sum(rewards) / len(rewards),
                "mean_raw_score": sum(raw_scores) / max(len(raw_scores), 1),
                "valid_action_rate": n_valid / len(completions),
                "violation_rate": n_violations / n_valid,
                "rubric_completion": rubric_sums["completion"] / n_valid,
                "rubric_energy": rubric_sums["energy"] / n_valid,
                "rubric_thermal": rubric_sums["thermal"] / n_valid,
                "rubric_solar": rubric_sums["solar"] / n_valid,
                "rubric_time": rubric_sums["time"] / n_valid,
            })

        return rewards
