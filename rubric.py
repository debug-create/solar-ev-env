"""
rubric.py
Composable rubric-based reward system for the Solar EV Environment.

Instead of a single opaque score, this module decomposes the final reward into
interpretable sub-components that judges (and training pipelines) can inspect.

Sub-reward components:
  - energy_efficiency_score   (0-1): Wh per km vs an optimal baseline
  - thermal_management_score  (0-1): how well battery temps were managed
  - solar_utilization_score   (0-1): how much available solar was harvested
  - time_performance_score    (0-1): time vs budget (for timed tasks)
  - completion_score          (0-1): fraction of waypoints completed
  - constraint_penalty        (0 or -1): hard constraint violations

Final weighted_total = weighted sum of components (clipped to [0.0001, 0.9999])
"""

from physics import BATTERY_MAX_TEMP_C, BATTERY_MIN_SOC_PCT, BATTERY_CAPACITY_WH


# ── Optimal baselines (derived from physics analysis) ─────────────────────────
# A perfectly efficient solar race car on flat ground at 60kph uses ~18 Wh/km.
# We use 25 Wh/km as the "good" target and 80 Wh/km as "terrible".
OPTIMAL_WH_PER_KM = 25.0
WORST_WH_PER_KM = 80.0

# ── Rubric weights ────────────────────────────────────────────────────────────
RUBRIC_WEIGHTS = {
    "energy_efficiency_score": 0.20,
    "thermal_management_score": 0.15,
    "solar_utilization_score": 0.15,
    "time_performance_score": 0.10,
    "completion_score": 0.40,
}


def _energy_efficiency_score(
    cumulative_energy_wh: float,
    total_distance_km: float,
) -> float:
    """
    Score based on Wh consumed per km driven.
    Lower is better. Score 1.0 = optimal, 0.0 = terrible or worse.
    """
    if total_distance_km <= 0:
        return 0.0

    wh_per_km = cumulative_energy_wh / total_distance_km

    if wh_per_km <= OPTIMAL_WH_PER_KM:
        return 1.0
    if wh_per_km >= WORST_WH_PER_KM:
        return 0.0

    # Linear interpolation between optimal and worst
    return 1.0 - (wh_per_km - OPTIMAL_WH_PER_KM) / (WORST_WH_PER_KM - OPTIMAL_WH_PER_KM)


def _thermal_management_score(
    battery_temp_c: float,
) -> float:
    """
    Score based on how far battery temp is from the danger zone.
    25C = perfect (1.0), >=58C = failure (0.0).
    """
    if battery_temp_c >= BATTERY_MAX_TEMP_C:
        return 0.0
    if battery_temp_c <= 25.0:
        return 1.0

    return max(0.0, 1.0 - ((battery_temp_c - 25.0) / (BATTERY_MAX_TEMP_C - 25.0)))


def _solar_utilization_score(
    cumulative_solar_wh: float,
    cumulative_energy_wh: float,
) -> float:
    """
    Score based on fraction of total energy that came from solar recovery.
    Higher solar contribution = better score.
    Capped at 1.0 — if solar covered >50% of consumption, perfect score.
    """
    if cumulative_energy_wh <= 0:
        return 1.0  # no energy used = trivially perfect

    ratio = cumulative_solar_wh / cumulative_energy_wh
    # If solar recovered >= 50% of consumed energy, that's excellent
    return min(1.0, ratio / 0.5)


def _time_performance_score(
    total_time_s: float,
    time_budget_s: float,
    strict_time_limit: bool,
) -> float:
    """
    Score based on time performance vs budget.
    If no strict time limit, return 1.0 (not penalized).
    Under budget = 1.0, over budget = decays linearly.
    """
    if not strict_time_limit or time_budget_s <= 0:
        return 1.0

    if total_time_s <= time_budget_s:
        return 1.0

    overshoot_fraction = (total_time_s - time_budget_s) / time_budget_s
    return max(0.0, 1.0 - overshoot_fraction)


def _completion_score(
    waypoints_completed: int,
    total_waypoints: int,
) -> float:
    """
    Fraction of waypoints completed. Simple linear measure.
    """
    if total_waypoints <= 0:
        return 0.0
    return waypoints_completed / total_waypoints


def _constraint_penalty(
    terminated_early: bool,
    soc_pct: float,
    battery_temp_c: float,
) -> float:
    """
    Returns -1.0 if any hard constraint was violated, 0.0 otherwise.
    Hard constraints:
      - battery_temp_c >= 58C (overheat)
      - soc_pct <= 20% (battery depleted)
    """
    if terminated_early:
        return -1.0
    if battery_temp_c >= BATTERY_MAX_TEMP_C:
        return -1.0
    if soc_pct <= BATTERY_MIN_SOC_PCT:
        return -1.0
    return 0.0


def compute_rubric(
    task_id: str,
    soc_pct: float,
    battery_temp_c: float,
    total_time_s: float,
    time_budget_s: float,
    strict_time_limit: bool,
    waypoints_completed: int,
    total_waypoints: int,
    terminated_early: bool,
    cumulative_energy_wh: float,
    cumulative_solar_wh: float,
    total_distance_km: float,
) -> dict:
    """
    Compute the full rubric breakdown for an episode.

    Returns a dict with individual sub-scores plus a final weighted_total.
    All sub-scores are in [0.0, 1.0] except constraint_penalty which is 0 or -1.
    weighted_total is clipped to [0.0001, 0.9999] per OpenEnv convention.
    """
    energy = round(_energy_efficiency_score(cumulative_energy_wh, total_distance_km), 4)
    thermal = round(_thermal_management_score(battery_temp_c), 4)
    solar = round(_solar_utilization_score(cumulative_solar_wh, cumulative_energy_wh), 4)
    time_perf = round(_time_performance_score(total_time_s, time_budget_s, strict_time_limit), 4)
    completion = round(_completion_score(waypoints_completed, total_waypoints), 4)
    penalty = _constraint_penalty(terminated_early, soc_pct, battery_temp_c)

    # Weighted sum of the 5 positive components
    weighted = (
        RUBRIC_WEIGHTS["energy_efficiency_score"] * energy
        + RUBRIC_WEIGHTS["thermal_management_score"] * thermal
        + RUBRIC_WEIGHTS["solar_utilization_score"] * solar
        + RUBRIC_WEIGHTS["time_performance_score"] * time_perf
        + RUBRIC_WEIGHTS["completion_score"] * completion
    )

    # Apply constraint penalty: if violated, collapse to near-zero
    if penalty < 0:
        weighted_total = 0.0001
    else:
        weighted_total = max(0.0001, min(0.9999, round(weighted, 4)))

    return {
        "energy_efficiency_score": energy,
        "thermal_management_score": thermal,
        "solar_utilization_score": solar,
        "time_performance_score": time_perf,
        "completion_score": completion,
        "constraint_penalty": penalty,
        "weights": RUBRIC_WEIGHTS,
        "weighted_total": weighted_total,
    }
