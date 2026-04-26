"""
advisor.py
Weather & Strategy Advisor — the 'second agent' in the Solar EV environment.

This module implements a structured advisory system that provides the PMU
strategist with forward-looking intelligence. It simulates a multi-agent
interaction where a weather/route advisor analyses upcoming conditions
and recommends optimal strategies.

The advisor is deterministic given the same inputs, so it does not break
reproducibility. It uses physics-aware heuristics to generate intelligence
that a language model agent can reason about.

Touches hackathon themes:
  - Multi-Agent Interactions (advisor ↔ PMU strategist)
  - Long-Horizon Planning (route-level analysis)
"""

from typing import List
from models import SegmentAhead, WeatherForecast
from physics import (
    BATTERY_MAX_TEMP_C,
    BATTERY_MIN_SOC_PCT,
    BATTERY_CAPACITY_WH,
)


def generate_forecast(
    upcoming_segments: List[SegmentAhead],
    current_soc_pct: float,
    current_battery_temp_c: float,
    distance_remaining_km: float,
    cumulative_energy_wh: float,
    distance_covered_km: float,
) -> WeatherForecast:
    """Generate a structured weather and strategy forecast.

    Analyses the upcoming segments and current vehicle state to produce
    actionable recommendations for the PMU strategist.

    Args:
        upcoming_segments: The next 1-3 visible segments.
        current_soc_pct: Current battery state of charge (%).
        current_battery_temp_c: Current battery temperature (°C).
        distance_remaining_km: Total remaining distance on the track.
        cumulative_energy_wh: Energy consumed so far.
        distance_covered_km: Distance covered so far.

    Returns:
        A WeatherForecast with risk assessments and speed/cooling/solar
        recommendations.
    """
    if not upcoming_segments:
        return _empty_forecast()

    # ── Aggregate upcoming conditions ──────────────────────────────────────
    n = len(upcoming_segments)
    avg_solar = sum(s.solar_irradiance_wm2 for s in upcoming_segments) / n
    avg_temp = sum(s.ambient_temp_c for s in upcoming_segments) / n
    total_dist = sum(s.distance_to_next_waypoint_km for s in upcoming_segments)

    # Elevation gain: convert incline % over distance to approximate meters
    total_elevation = sum(
        s.average_incline_pct / 100.0 * s.distance_to_next_waypoint_km * 1000.0
        for s in upcoming_segments
    )

    # ── Thermal risk assessment ────────────────────────────────────────────
    thermal_margin = BATTERY_MAX_TEMP_C - current_battery_temp_c
    has_steep_uphill = any(s.average_incline_pct > 4.0 for s in upcoming_segments)
    has_hot_ambient = any(s.ambient_temp_c > 33.0 for s in upcoming_segments)

    if thermal_margin < 5.0 or (has_steep_uphill and has_hot_ambient):
        thermal_risk = "critical"
    elif thermal_margin < 10.0 or has_steep_uphill:
        thermal_risk = "high"
    elif thermal_margin < 18.0 or has_hot_ambient:
        thermal_risk = "moderate"
    else:
        thermal_risk = "low"

    # ── Energy risk assessment ─────────────────────────────────────────────
    remaining_energy_wh = (current_soc_pct / 100.0) * BATTERY_CAPACITY_WH
    soc_margin = current_soc_pct - BATTERY_MIN_SOC_PCT

    # Estimate energy needed for remaining distance
    if distance_covered_km > 0 and cumulative_energy_wh > 0:
        avg_wh_per_km = cumulative_energy_wh / distance_covered_km
    else:
        avg_wh_per_km = 30.0  # conservative default

    energy_needed_wh = avg_wh_per_km * distance_remaining_km
    energy_buffer_ratio = remaining_energy_wh / max(energy_needed_wh, 1.0)

    if soc_margin < 10.0 or energy_buffer_ratio < 0.9:
        energy_risk = "critical"
    elif soc_margin < 20.0 or energy_buffer_ratio < 1.1:
        energy_risk = "high"
    elif soc_margin < 35.0 or energy_buffer_ratio < 1.3:
        energy_risk = "moderate"
    else:
        energy_risk = "low"

    # ── Generate speed recommendation ──────────────────────────────────────
    next_seg = upcoming_segments[0]
    base_speed = 60.0

    # Adjust for energy risk
    if energy_risk == "critical":
        base_speed = 35.0
    elif energy_risk == "high":
        base_speed = 45.0
    elif energy_risk == "moderate":
        base_speed = 55.0

    # Adjust for incline
    if next_seg.average_incline_pct > 4.0:
        base_speed = min(base_speed, 45.0)
    elif next_seg.average_incline_pct > 2.0:
        base_speed = min(base_speed, 55.0)
    elif next_seg.average_incline_pct < -2.0:
        # Downhill = go faster for regen + time savings
        base_speed = max(base_speed, 70.0)

    # Adjust for thermal risk
    if thermal_risk == "critical":
        base_speed = min(base_speed, 40.0)
    elif thermal_risk == "high":
        base_speed = min(base_speed, 50.0)

    recommended_speed = max(10.0, min(120.0, round(base_speed, 1)))

    # ── Cooling recommendation ─────────────────────────────────────────────
    if thermal_risk == "critical":
        recommended_cooling = 2
    elif thermal_risk == "high":
        recommended_cooling = 2
    elif thermal_risk == "moderate" or current_battery_temp_c > 40.0:
        recommended_cooling = 1
    else:
        recommended_cooling = 0

    # ── Solar routing recommendation ───────────────────────────────────────
    if avg_solar > 600.0 and energy_risk in ("low", "moderate"):
        recommended_solar = "direct_to_motor"
    elif avg_solar < 200.0:
        recommended_solar = "charge_battery"
    elif energy_risk in ("high", "critical"):
        recommended_solar = "direct_to_motor"  # maximize immediate savings
    else:
        recommended_solar = "direct_to_motor"

    # ── Build reasoning string ─────────────────────────────────────────────
    reasoning_parts = []

    if thermal_risk in ("high", "critical"):
        reasoning_parts.append(
            f"THERMAL WARNING: Battery at {current_battery_temp_c:.1f}°C with "
            f"{'steep climbs' if has_steep_uphill else 'hot conditions'} ahead. "
            f"Reduce speed and activate cooling."
        )

    if energy_risk in ("high", "critical"):
        reasoning_parts.append(
            f"ENERGY WARNING: SoC at {current_soc_pct:.1f}% with "
            f"{distance_remaining_km:.1f}km remaining. "
            f"Buffer ratio: {energy_buffer_ratio:.2f}x. Conserve aggressively."
        )

    if avg_solar < 200.0:
        reasoning_parts.append(
            f"LOW SOLAR: Average irradiance ahead is {avg_solar:.0f} W/m². "
            f"Expect reduced solar recovery."
        )
    elif avg_solar > 700.0:
        reasoning_parts.append(
            f"GOOD SOLAR: Average irradiance ahead is {avg_solar:.0f} W/m². "
            f"Route solar direct to motor to offset battery drain."
        )

    if next_seg.average_incline_pct > 3.0:
        reasoning_parts.append(
            f"UPHILL: Next segment has {next_seg.average_incline_pct:.1f}% grade. "
            f"Reduce speed to limit energy and heat spike."
        )
    elif next_seg.average_incline_pct < -2.0:
        reasoning_parts.append(
            f"DOWNHILL: Next segment has {next_seg.average_incline_pct:.1f}% grade. "
            f"Increase speed for regenerative braking energy recovery."
        )

    if not reasoning_parts:
        reasoning_parts.append(
            f"CONDITIONS NOMINAL: Thermal margin {thermal_margin:.1f}°C, "
            f"energy buffer {energy_buffer_ratio:.2f}x. Maintain steady pace."
        )

    reasoning = " | ".join(reasoning_parts)

    return WeatherForecast(
        segments_ahead_count=n,
        avg_solar_irradiance_ahead=round(avg_solar, 1),
        avg_ambient_temp_ahead=round(avg_temp, 1),
        total_distance_ahead_km=round(total_dist, 2),
        total_elevation_gain_m=round(total_elevation, 1),
        thermal_risk=thermal_risk,
        energy_risk=energy_risk,
        recommended_speed_kph=recommended_speed,
        recommended_cooling=recommended_cooling,
        recommended_solar_mode=recommended_solar,
        reasoning=reasoning,
    )


def _empty_forecast() -> WeatherForecast:
    """Return a neutral forecast when no segments are visible."""
    return WeatherForecast(
        segments_ahead_count=0,
        avg_solar_irradiance_ahead=0.0,
        avg_ambient_temp_ahead=25.0,
        total_distance_ahead_km=0.0,
        total_elevation_gain_m=0.0,
        thermal_risk="low",
        energy_risk="low",
        recommended_speed_kph=55.0,
        recommended_cooling=1,
        recommended_solar_mode="direct_to_motor",
        reasoning="No upcoming segments visible. Maintain steady defaults.",
    )
