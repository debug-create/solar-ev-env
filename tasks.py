"""
tasks.py
Task definitions for the Solar EV environment.
Defines 5 tasks: easy → medium → medium-hard → hard → expert, each with track
segments and graders. All graders are 100% deterministic — no LLM-as-judge, pure math.
"""

from dataclasses import dataclass, field
from typing import List
from models import SegmentAhead


@dataclass
class TaskDefinition:
    """Full definition of one task including track layout and constraints."""
    task_id: str
    name: str
    difficulty: str
    description: str
    total_distance_km: float
    time_budget_s: float
    segments: List[SegmentAhead]
    starting_soc_pct: float = 95.0
    starting_battery_temp_c: float = 28.0
    starting_motor_temp_c: float = 25.0
    strict_time_limit: bool = False


# ── TASK 1: Easy — Flat Track Endurance ───────────────────────────────────────
TASK_EASY = TaskDefinition(
    task_id="flat_track_easy",
    name="Flat Track Endurance",
    difficulty="easy",
    description=(
        "Maintain safe operation on a flat 20km track under clear skies. "
        "Do not let battery SoC drop below 20%. Complete all waypoints."
    ),
    total_distance_km=20.0,
    time_budget_s=99999,
    strict_time_limit=False,
    starting_soc_pct=95.0,
    starting_battery_temp_c=28.0,
    starting_motor_temp_c=25.0,
    segments=[
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0, solar_irradiance_wm2=950.0, ambient_temp_c=28.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0, solar_irradiance_wm2=980.0, ambient_temp_c=28.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0, solar_irradiance_wm2=970.0, ambient_temp_c=28.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0, solar_irradiance_wm2=960.0, ambient_temp_c=28.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0, solar_irradiance_wm2=940.0, ambient_temp_c=28.0),
    ],
)

# ── TASK 2: Medium — Hills and Clouds Challenge ────────────────────────────────
TASK_MEDIUM = TaskDefinition(
    task_id="dynamic_routing_medium",
    name="Hills and Clouds Challenge",
    difficulty="medium",
    description=(
        "Navigate a 30km track with varying elevation and cloud cover. "
        "Dynamically route solar power and manage speed across changing conditions."
    ),
    total_distance_km=30.0,
    time_budget_s=99999,
    strict_time_limit=False,
    starting_soc_pct=90.0,
    starting_battery_temp_c=28.0,
    starting_motor_temp_c=25.0,
    segments=[
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0,  solar_irradiance_wm2=900.0, ambient_temp_c=22.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=4.5,  solar_irradiance_wm2=400.0, ambient_temp_c=24.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=-3.0, solar_irradiance_wm2=200.0, ambient_temp_c=23.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=2.0,  solar_irradiance_wm2=750.0, ambient_temp_c=26.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0,  solar_irradiance_wm2=950.0, ambient_temp_c=30.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=-2.5, solar_irradiance_wm2=850.0, ambient_temp_c=32.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=1.5,  solar_irradiance_wm2=300.0, ambient_temp_c=28.0),
    ],
)

# ── TASK 3: Hard — Thermal Race Sprint ────────────────────────────────────────
TASK_HARD = TaskDefinition(
    task_id="thermal_race_hard",
    name="Thermal Race Sprint",
    difficulty="hard",
    description=(
        "Complete a 40km aggressive race circuit within 2800 seconds. "
        "Battery temperature must stay below 58C at all times. "
        "High-speed sections stress the thermal system — manage cooling actively."
    ),
    total_distance_km=40.0,
    time_budget_s=2800.0,
    strict_time_limit=True,
    starting_soc_pct=100.0,
    starting_battery_temp_c=30.0,
    starting_motor_temp_c=25.0,
    segments=[
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=0.0,  solar_irradiance_wm2=800.0, ambient_temp_c=28.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=3.0,  solar_irradiance_wm2=750.0, ambient_temp_c=29.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=6.0,  solar_irradiance_wm2=600.0, ambient_temp_c=32.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=-5.0, solar_irradiance_wm2=700.0, ambient_temp_c=30.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=0.0,  solar_irradiance_wm2=900.0, ambient_temp_c=34.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=4.0,  solar_irradiance_wm2=500.0, ambient_temp_c=35.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=-2.0, solar_irradiance_wm2=850.0, ambient_temp_c=33.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=2.0,  solar_irradiance_wm2=780.0, ambient_temp_c=31.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0,  solar_irradiance_wm2=820.0, ambient_temp_c=29.0),
    ],
)

# ── TASK 4: Medium-Hard — Night Run No Solar ──────────────────────────────────
TASK_NIGHT = TaskDefinition(
    task_id="night_run_no_solar",
    name="Night Run: Zero Solar",
    difficulty="medium-hard",
    description=(
        "Complete a 25km nighttime course with zero solar irradiance. "
        "The agent must conserve battery aggressively with no solar recovery. "
        "Mix of flat and hilly terrain tests energy management under scarcity."
    ),
    total_distance_km=25.0,
    time_budget_s=99999,
    strict_time_limit=False,
    starting_soc_pct=85.0,
    starting_battery_temp_c=25.0,
    starting_motor_temp_c=22.0,
    segments=[
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=0.0,  solar_irradiance_wm2=0.0, ambient_temp_c=18.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=3.5,  solar_irradiance_wm2=0.0, ambient_temp_c=17.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=-2.0, solar_irradiance_wm2=0.0, ambient_temp_c=16.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=5.0,  solar_irradiance_wm2=0.0, ambient_temp_c=16.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=-4.0, solar_irradiance_wm2=0.0, ambient_temp_c=15.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=1.0,  solar_irradiance_wm2=0.0, ambient_temp_c=15.0),
    ],
)

# ── TASK 5: Expert — Ultra Endurance Challenge ────────────────────────────────
TASK_EXPERT = TaskDefinition(
    task_id="ultra_endurance_expert",
    name="Ultra Endurance Challenge",
    difficulty="expert",
    description=(
        "Complete a grueling 60km course with 12 segments under mixed conditions. "
        "Hills, cloud cover, heat stress, and a strict 4500s time budget — all at once. "
        "This is the ultimate test for frontier LLM energy management strategies."
    ),
    total_distance_km=60.0,
    time_budget_s=4500.0,
    strict_time_limit=True,
    starting_soc_pct=100.0,
    starting_battery_temp_c=32.0,
    starting_motor_temp_c=28.0,
    segments=[
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=0.0,  solar_irradiance_wm2=850.0,  ambient_temp_c=30.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=3.0,  solar_irradiance_wm2=700.0,  ambient_temp_c=31.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=5.5,  solar_irradiance_wm2=400.0,  ambient_temp_c=33.0),
        SegmentAhead(distance_to_next_waypoint_km=4.0, average_incline_pct=-3.0, solar_irradiance_wm2=600.0,  ambient_temp_c=34.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=0.0,  solar_irradiance_wm2=950.0,  ambient_temp_c=36.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=2.5,  solar_irradiance_wm2=300.0,  ambient_temp_c=35.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=-4.5, solar_irradiance_wm2=500.0,  ambient_temp_c=33.0),
        SegmentAhead(distance_to_next_waypoint_km=6.0, average_incline_pct=4.0,  solar_irradiance_wm2=750.0,  ambient_temp_c=37.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=0.0,  solar_irradiance_wm2=200.0,  ambient_temp_c=38.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=-2.0, solar_irradiance_wm2=800.0,  ambient_temp_c=36.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=6.0,  solar_irradiance_wm2=550.0,  ambient_temp_c=35.0),
        SegmentAhead(distance_to_next_waypoint_km=5.0, average_incline_pct=-1.0, solar_irradiance_wm2=900.0,  ambient_temp_c=34.0),
    ],
)


TASKS = {
    "flat_track_easy": TASK_EASY,
    "dynamic_routing_medium": TASK_MEDIUM,
    "thermal_race_hard": TASK_HARD,
    "night_run_no_solar": TASK_NIGHT,
    "ultra_endurance_expert": TASK_EXPERT,
}


def grade_step(
    soc_pct: float,
    battery_temp_c: float,
    waypoints_completed: int,
    total_waypoints: int,
    task: TaskDefinition,
) -> float:
    """
    Compute a partial reward score for completing one waypoint.
    Returns 0.0 if hard constraints are violated.
    Returns 0.0–1.0 based on progress, SoC efficiency, and thermal health.
    """
    from physics import BATTERY_MAX_TEMP_C, BATTERY_MIN_SOC_PCT

    if battery_temp_c >= BATTERY_MAX_TEMP_C:
        return 0.0001
    if soc_pct <= BATTERY_MIN_SOC_PCT:
        return 0.0001

    progress_score = waypoints_completed / total_waypoints
    soc_bonus = (soc_pct - BATTERY_MIN_SOC_PCT) / (100.0 - BATTERY_MIN_SOC_PCT)
    thermal_bonus = max(0.0, 1.0 - ((battery_temp_c - 25.0) / (BATTERY_MAX_TEMP_C - 25.0)))

    step_score = (progress_score * 0.5) + (soc_bonus * 0.3) + (thermal_bonus * 0.2)
    return max(0.0001, min(0.9999, round(step_score, 4)))


def grade_final(
    task: TaskDefinition,
    soc_pct: float,
    battery_temp_c: float,
    total_time_s: float,
    waypoints_completed: int,
    terminated_early: bool,
    cumulative_energy_wh: float = 0.0,
    cumulative_solar_wh: float = 0.0,
    total_distance_km: float = 0.0,
) -> tuple:
    """
    Compute the final episode score, reason string, and rubric breakdown.
    Returns (score: float, reason: str, rubric: dict)

    The rubric dict is computed by rubric.py and provides interpretable
    sub-scores for judges and training pipelines.
    """
    from physics import BATTERY_MAX_TEMP_C, BATTERY_MIN_SOC_PCT
    from rubric import compute_rubric

    total_waypoints = len(task.segments)

    # Compute rubric regardless of outcome — judges always see it
    rubric = compute_rubric(
        task_id=task.task_id,
        soc_pct=soc_pct,
        battery_temp_c=battery_temp_c,
        total_time_s=total_time_s,
        time_budget_s=task.time_budget_s,
        strict_time_limit=task.strict_time_limit,
        waypoints_completed=waypoints_completed,
        total_waypoints=total_waypoints,
        terminated_early=terminated_early,
        cumulative_energy_wh=cumulative_energy_wh,
        cumulative_solar_wh=cumulative_solar_wh,
        total_distance_km=total_distance_km,
    )

    if terminated_early:
        if battery_temp_c >= BATTERY_MAX_TEMP_C:
            return 0.0001, f"FAIL: Battery overheated at {battery_temp_c:.1f}C (limit: {BATTERY_MAX_TEMP_C}C)", rubric
        if soc_pct <= BATTERY_MIN_SOC_PCT:
            return 0.0001, f"FAIL: Battery depleted to {soc_pct:.1f}% (minimum: {BATTERY_MIN_SOC_PCT}%)", rubric
        return 0.0001, "FAIL: Episode terminated early due to constraint violation", rubric

    if waypoints_completed < total_waypoints:
        partial = waypoints_completed / total_waypoints
        return max(0.0001, min(0.9999, round(partial * 0.4, 4))), f"PARTIAL: Completed {waypoints_completed}/{total_waypoints} waypoints", rubric

    soc_bonus = (soc_pct - BATTERY_MIN_SOC_PCT) / (100.0 - BATTERY_MIN_SOC_PCT)
    thermal_bonus = max(0.0, 1.0 - ((battery_temp_c - 25.0) / (BATTERY_MAX_TEMP_C - 25.0)))
    base_score = 0.6
    time_note = ""

    if task.strict_time_limit and total_time_s > task.time_budget_s:
        time_penalty = min(0.3, (total_time_s - task.time_budget_s) / task.time_budget_s)
        base_score -= time_penalty
        time_note = f" Time exceeded budget by {total_time_s - task.time_budget_s:.0f}s."

    final_score = round(min(1.0, max(0.0, base_score + (soc_bonus * 0.25) + (thermal_bonus * 0.15))), 4)
    final_score = max(0.0001, min(0.9999, final_score))

    reason = (
        f"SUCCESS: All {total_waypoints} waypoints complete.{time_note} "
        f"Final SoC: {soc_pct:.1f}%, Battery temp: {battery_temp_c:.1f}C, "
        f"Time: {total_time_s:.0f}s. Score: {final_score}"
    )
    return final_score, reason, rubric