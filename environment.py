"""
environment.py
Core OpenEnv environment.
Implements reset(), step(), and state() as required by the OpenEnv spec.

KEY FIX: motor_temp_c and battery_temp_c are clamped before storing in
VehicleState (Pydantic caps at 120 and 80 respectively), but the RAW
unclamped values from physics are used for constraint checking and grading.
This ensures crashes never happen while grading remains correct.

ROUND 2 ADDITIONS:
  - Controlled stochasticity via seed parameter (prevents memorization)
  - Look-ahead window of 2-3 upcoming segments (long-horizon planning)
  - Weather advisor integration (multi-agent interaction)
  - Cumulative energy/solar counters and rubric-based reward breakdowns
"""

import random as _random
from typing import Optional, List

from models import (
    Observation, Action, Reward, VehicleState,
    SegmentAhead, EpisodeResult,
)
from physics import simulate_segment, BATTERY_MAX_TEMP_C, BATTERY_MIN_SOC_PCT, BATTERY_CAPACITY_WH
from tasks import TASKS, TaskDefinition, grade_step, grade_final
from advisor import generate_forecast

# ── Stochasticity bounds ──────────────────────────────────────────────────────
# These are intentionally small so tasks remain solvable but not memorizable.
_SOLAR_NOISE_FRACTION = 0.10     # ±10% solar irradiance
_TEMP_NOISE_C = 3.0              # ±3°C ambient temperature
_INCLINE_NOISE_PCT = 0.5         # ±0.5% incline perturbation

# Number of upcoming segments to show in the look-ahead window
_LOOKAHEAD_WINDOW = 3


class SolarEVEnvironment:
    """OpenEnv-compliant Solar EV PMU Strategist environment.

    Simulates a solar electric vehicle traversing a track of waypoints.
    Each episode, the agent configures the PMU at each waypoint by setting
    speed, cooling, and solar routing. The deterministic physics engine
    computes the resulting vehicle state.

    Round 2 features:
      - Controlled stochasticity: pass seed to reset() for reproducible
        variation in segment conditions. Without seed, episodes are
        fully deterministic (backward compatible).
      - Look-ahead: observation includes next 2-3 upcoming segments.
      - Weather advisor: a second agent that provides structured forecasts
        and strategy recommendations.

    Class Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: Required by OpenEnv for create_app()
            with max_concurrent_envs > 1.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task: Optional[TaskDefinition] = None
        self._segments: List[SegmentAhead] = []  # working copy (may be perturbed)
        self._current_waypoint: int = 0
        self._vehicle_state: Optional[VehicleState] = None
        self._total_time_s: float = 0.0
        self._terminated_early: bool = False
        self._step_scores: list = []
        self._done: bool = False
        self._seed: Optional[int] = None

        # ── Cumulative counters ───────────────────────────────────────────────
        self._cumulative_energy_wh: float = 0.0
        self._cumulative_solar_wh: float = 0.0
        self._last_segment_energy_wh: float = 0.0
        self._last_segment_distance_km: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = "flat_track_easy",
        randomize: bool = False,
        seed: Optional[int] = None,
    ) -> Observation:
        """Start a new episode for the given task.

        Args:
            task_id: Which task to run. Must be a key in TASKS.
            randomize: When True, perturb segment weather/elevation within
                bounded ranges so agents cannot memorize fixed tracks.
            seed: Optional random seed for reproducible stochastic episodes.
                A seed has no effect unless randomize=True, keeping default
                resets fully deterministic and backward compatible.

        Returns:
            Initial Observation with vehicle state and upcoming segments.
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid options: {list(TASKS.keys())}")

        self._task = TASKS[task_id]
        self._seed = seed
        self._current_waypoint = 0
        self._total_time_s = 0.0
        self._terminated_early = False
        self._step_scores = []
        self._done = False

        # Reset cumulative counters
        self._cumulative_energy_wh = 0.0
        self._cumulative_solar_wh = 0.0
        self._last_segment_energy_wh = 0.0
        self._last_segment_distance_km = 0.0

        # Create working copy of segments (deep copy to avoid mutating task defs)
        self._segments = [
            SegmentAhead(**seg.model_dump()) for seg in self._task.segments
        ]

        # Apply bounded stochastic perturbation only when explicitly requested.
        if randomize:
            self._perturb_segments(seed)

        self._vehicle_state = VehicleState(
            battery_soc_pct=self._task.starting_soc_pct,
            battery_temp_c=self._task.starting_battery_temp_c,
            motor_temp_c=self._task.starting_motor_temp_c,
            current_speed_kph=0.0,
            distance_covered_km=0.0,
            solar_power_generated_w=0.0,
        )

        return self._build_observation(last_feedback=None)

    def step(self, action: Action) -> tuple:
        """Execute one agent action (waypoint configuration).

        Simulates the segment physics, updates state, returns
        (Observation, Reward) tuple.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        segment = self._segments[self._current_waypoint]

        # ── Run physics simulation ──────────────────────────────────────────
        result = simulate_segment(
            current_soc_pct=self._vehicle_state.battery_soc_pct,
            current_battery_temp_c=self._vehicle_state.battery_temp_c,
            current_motor_temp_c=self._vehicle_state.motor_temp_c,
            ambient_temp_c=segment.ambient_temp_c,
            segment_distance_km=segment.distance_to_next_waypoint_km,
            segment_incline_pct=segment.average_incline_pct,
            solar_irradiance_wm2=segment.solar_irradiance_wm2,
            target_speed_kph=action.target_cruise_speed_kph,
            cooling_level=action.cooling_system_level,
            solar_routing_mode=action.solar_routing_mode,
        )

        # ── Update cumulative counters ──────────────────────────────────────
        self._cumulative_energy_wh += result.energy_consumed_wh
        solar_total_wh = (result.solar_power_w * result.time_taken_s) / 3600.0
        self._cumulative_solar_wh += solar_total_wh
        self._last_segment_energy_wh = result.energy_consumed_wh
        self._last_segment_distance_km = segment.distance_to_next_waypoint_km

        # ── Update vehicle state ────────────────────────────────────────────
        # IMPORTANT: clamp temps to Pydantic model limits to avoid ValidationError.
        # Raw physics values (result.*) are still used for grading below.
        safe_battery_temp = min(result.new_battery_temp_c, 79.9)
        safe_motor_temp = min(result.new_motor_temp_c, 119.9)

        self._vehicle_state = VehicleState(
            battery_soc_pct=result.new_soc_pct,
            battery_temp_c=safe_battery_temp,
            motor_temp_c=safe_motor_temp,
            current_speed_kph=result.new_speed_kph,
            distance_covered_km=(
                self._vehicle_state.distance_covered_km + segment.distance_to_next_waypoint_km
            ),
            solar_power_generated_w=result.solar_power_w,
        )

        self._total_time_s += result.time_taken_s
        self._current_waypoint += 1

        # ── Check hard constraints using RAW physics values ─────────────────
        constraint_violated = (
            result.new_battery_temp_c >= BATTERY_MAX_TEMP_C
            or result.new_soc_pct <= BATTERY_MIN_SOC_PCT
        )
        is_last_waypoint = self._current_waypoint >= len(self._segments)

        if constraint_violated:
            self._terminated_early = True
            self._done = True
        elif is_last_waypoint:
            self._done = True

        # ── Grade using RAW physics values ──────────────────────────────────
        step_score = grade_step(
            soc_pct=result.new_soc_pct,
            battery_temp_c=result.new_battery_temp_c,
            waypoints_completed=self._current_waypoint,
            total_waypoints=len(self._segments),
            task=self._task,
        )
        self._step_scores.append(step_score)

        if self._done:
            final_score, final_reason, rubric = grade_final(
                task=self._task,
                soc_pct=result.new_soc_pct,
                battery_temp_c=result.new_battery_temp_c,
                total_time_s=self._total_time_s,
                waypoints_completed=self._current_waypoint,
                terminated_early=self._terminated_early,
                cumulative_energy_wh=self._cumulative_energy_wh,
                cumulative_solar_wh=self._cumulative_solar_wh,
                total_distance_km=self._vehicle_state.distance_covered_km,
            )
            reward = Reward(
                score=final_score,
                is_done=True,
                is_success=final_score >= 0.6,
                feedback=final_reason,
                battery_soc_remaining_pct=result.new_soc_pct,
                battery_temp_c=result.new_battery_temp_c,
                distance_covered_km=self._vehicle_state.distance_covered_km,
                rubric=rubric,
            )
        else:
            reward = Reward(
                score=step_score,
                is_done=False,
                is_success=False,
                feedback=result.feedback,
                battery_soc_remaining_pct=result.new_soc_pct,
                battery_temp_c=result.new_battery_temp_c,
                distance_covered_km=self._vehicle_state.distance_covered_km,
            )

        obs = self._build_observation(last_feedback=result.feedback)
        return obs, reward

    def state(self) -> dict:
        """Return full current environment state. Used by the /state endpoint."""
        if self._vehicle_state is None:
            return {"status": "not_initialized"}

        return {
            "task_id": self._task.task_id if self._task else None,
            "waypoint_index": self._current_waypoint,
            "total_waypoints": len(self._segments) if self._segments else 0,
            "vehicle": self._vehicle_state.model_dump(),
            "total_time_s": self._total_time_s,
            "done": self._done,
            "terminated_early": self._terminated_early,
            "step_scores": self._step_scores,
            "cumulative_energy_wh": round(self._cumulative_energy_wh, 2),
            "cumulative_solar_wh": round(self._cumulative_solar_wh, 2),
            "seed": self._seed,
        }

    def get_episode_result(self) -> Optional[EpisodeResult]:
        """Returns full episode summary once episode is done."""
        if not self._done or self._task is None:
            return None

        final_score, reason, rubric = grade_final(
            task=self._task,
            soc_pct=self._vehicle_state.battery_soc_pct,
            battery_temp_c=self._vehicle_state.battery_temp_c,
            total_time_s=self._total_time_s,
            waypoints_completed=self._current_waypoint,
            terminated_early=self._terminated_early,
            cumulative_energy_wh=self._cumulative_energy_wh,
            cumulative_solar_wh=self._cumulative_solar_wh,
            total_distance_km=self._vehicle_state.distance_covered_km,
        )

        return EpisodeResult(
            task_id=self._task.task_id,
            final_score=final_score,
            success=final_score >= 0.6,
            total_steps=self._current_waypoint,
            distance_covered_km=self._vehicle_state.distance_covered_km,
            final_soc_pct=self._vehicle_state.battery_soc_pct,
            final_battery_temp_c=self._vehicle_state.battery_temp_c,
            termination_reason=reason,
            step_scores=self._step_scores,
            rubric=rubric,
        )

    def close(self) -> None:
        """Clean up any resources held by the environment instance.

        Required by the OpenEnv Environment interface for session teardown.
        Our environment is purely in-memory so this is a no-op.
        """
        pass

    # ── Private helpers ────────────────────────────────────────────────────────

    def _perturb_segments(self, seed: Optional[int]) -> None:
        """Apply bounded stochastic perturbation to segment conditions.

        Uses the seed to create reproducible noise within safe bounds:
          - Solar irradiance: ±10% (clamped to [0, 1100])
          - Ambient temperature: ±3°C (clamped to [-10, 50])
          - Incline: ±0.5% (clamped to [-15, 15])

        The perturbation is small enough that every task remains solvable
        under all seeds, but large enough to prevent rote memorization.
        """
        rng = _random.Random(seed)

        for seg in self._segments:
            # Solar: ±10%
            solar_noise = rng.uniform(-_SOLAR_NOISE_FRACTION, _SOLAR_NOISE_FRACTION)
            new_solar = seg.solar_irradiance_wm2 * (1.0 + solar_noise)
            seg.solar_irradiance_wm2 = round(max(0.0, min(1100.0, new_solar)), 1)

            # Temperature: ±3°C
            temp_noise = rng.uniform(-_TEMP_NOISE_C, _TEMP_NOISE_C)
            new_temp = seg.ambient_temp_c + temp_noise
            seg.ambient_temp_c = round(max(-10.0, min(50.0, new_temp)), 1)

            # Incline: ±0.5%
            incline_noise = rng.uniform(-_INCLINE_NOISE_PCT, _INCLINE_NOISE_PCT)
            new_incline = seg.average_incline_pct + incline_noise
            seg.average_incline_pct = round(max(-15.0, min(15.0, new_incline)), 2)

    def _get_upcoming_segments(self) -> List[SegmentAhead]:
        """Return the next few segments for the look-ahead window."""
        start = self._current_waypoint
        end = min(start + _LOOKAHEAD_WINDOW, len(self._segments))
        return self._segments[start:end]

    def _build_observation(self, last_feedback: Optional[str]) -> Observation:
        """Internal helper to construct the Observation object."""
        total_waypoints = len(self._segments)
        steps_remaining = total_waypoints - self._current_waypoint

        if self._current_waypoint >= total_waypoints:
            upcoming = self._segments[-1]
        else:
            upcoming = self._segments[self._current_waypoint]

        # ── Distance and progress ───────────────────────────────────────────
        distance_covered = self._vehicle_state.distance_covered_km
        total_distance = self._task.total_distance_km
        track_progress = min(100.0, (distance_covered / total_distance) * 100.0) if total_distance > 0 else 0.0

        # ── Estimated range remaining ───────────────────────────────────────
        if distance_covered > 0 and self._cumulative_energy_wh > 0:
            avg_wh_per_km = self._cumulative_energy_wh / distance_covered
            remaining_energy_wh = (self._vehicle_state.battery_soc_pct / 100.0) * BATTERY_CAPACITY_WH
            estimated_range = remaining_energy_wh / avg_wh_per_km if avg_wh_per_km > 0 else 999.0
        else:
            remaining_energy_wh = (self._vehicle_state.battery_soc_pct / 100.0) * BATTERY_CAPACITY_WH
            estimated_range = remaining_energy_wh / 30.0

        # ── Last segment efficiency ─────────────────────────────────────────
        if self._last_segment_distance_km > 0:
            last_efficiency = self._last_segment_energy_wh / self._last_segment_distance_km
        else:
            last_efficiency = 0.0

        # ── Look-ahead window ───────────────────────────────────────────────
        upcoming_segments = self._get_upcoming_segments()

        # ── Weather advisor forecast ────────────────────────────────────────
        distance_remaining = total_distance - distance_covered
        weather_forecast = generate_forecast(
            upcoming_segments=upcoming_segments,
            current_soc_pct=self._vehicle_state.battery_soc_pct,
            current_battery_temp_c=self._vehicle_state.battery_temp_c,
            distance_remaining_km=max(0.0, distance_remaining),
            cumulative_energy_wh=self._cumulative_energy_wh,
            distance_covered_km=distance_covered,
        )

        return Observation(
            task_id=self._task.task_id,
            waypoint_index=self._current_waypoint,
            total_waypoints=total_waypoints,
            vehicle=self._vehicle_state,
            segment_ahead=upcoming,
            steps_remaining=steps_remaining,
            last_action_feedback=last_feedback,
            task_objective=self._task.description,
            episode_terminated_early=self._terminated_early,
            cumulative_energy_used_wh=round(self._cumulative_energy_wh, 2),
            cumulative_solar_recovered_wh=round(self._cumulative_solar_wh, 2),
            estimated_range_remaining_km=round(estimated_range, 2),
            track_progress_pct=round(track_progress, 2),
            last_segment_efficiency=round(last_efficiency, 2),
            upcoming_segments=upcoming_segments,
            weather_forecast=weather_forecast,
            episode_seed=self._seed,
        )
