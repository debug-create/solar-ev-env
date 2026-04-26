"""
models.py
Pydantic models for the Solar EV Environment.
Defines the strict typed contracts for Observation, Action, and Reward
that form the OpenEnv specification interface.

Round 2 additions:
  - WeatherForecast: structured advisory from the onboard weather system
  - upcoming_segments: look-ahead window for long-horizon planning
  - advisor_recommendation: strategy guidance from the weather advisor
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class VehicleState(BaseModel):
    """Real-time telemetry of the vehicle at the current waypoint."""
    battery_soc_pct: float = Field(..., ge=0.0, le=100.0, description="Battery State of Charge as a percentage (0-100)")
    battery_temp_c: float = Field(..., ge=-10.0, le=80.0, description="Battery temperature in Celsius")
    motor_temp_c: float = Field(..., ge=-10.0, le=120.0, description="Motor temperature in Celsius")
    current_speed_kph: float = Field(..., ge=0.0, le=130.0, description="Current vehicle speed in km/h")
    distance_covered_km: float = Field(..., ge=0.0, description="Total distance covered so far in km")
    solar_power_generated_w: float = Field(..., ge=0.0, description="Solar power being generated right now in Watts")


class SegmentAhead(BaseModel):
    """Data about the upcoming track segment the agent must plan for."""
    distance_to_next_waypoint_km: float = Field(..., gt=0.0, description="Distance to the next waypoint in km")
    average_incline_pct: float = Field(..., ge=-15.0, le=15.0, description="Average gradient. Negative = downhill.")
    solar_irradiance_wm2: float = Field(..., ge=0.0, le=1100.0, description="Expected solar irradiance in W/m2")
    ambient_temp_c: float = Field(..., ge=-10.0, le=50.0, description="Ambient air temperature in Celsius")


class WeatherForecast(BaseModel):
    """Structured forecast from the onboard weather advisory system.

    This represents a 'second agent' — a weather/route advisor that the
    PMU strategist can consult before making decisions. Provides
    forward-looking intelligence about upcoming conditions and risks.
    """
    segments_ahead_count: int = Field(..., ge=0, description="Number of segments visible in the forecast")
    avg_solar_irradiance_ahead: float = Field(..., ge=0.0, description="Mean solar irradiance over upcoming segments (W/m²)")
    avg_ambient_temp_ahead: float = Field(..., ge=-10.0, description="Mean ambient temperature ahead (°C)")
    total_distance_ahead_km: float = Field(..., ge=0.0, description="Total distance remaining in km")
    total_elevation_gain_m: float = Field(..., description="Net elevation change ahead (positive = uphill dominant)")
    thermal_risk: str = Field(..., description="'low', 'moderate', 'high', or 'critical' — based on upcoming heat + incline stress")
    energy_risk: str = Field(..., description="'low', 'moderate', 'high', or 'critical' — based on remaining range vs distance")
    recommended_speed_kph: float = Field(..., ge=10.0, le=120.0, description="Advisor's suggested cruise speed for next segment")
    recommended_cooling: int = Field(..., ge=0, le=2, description="Advisor's suggested cooling level")
    recommended_solar_mode: str = Field(..., description="Advisor's suggested solar routing mode")
    reasoning: str = Field(..., description="Plain-English explanation of the advisory recommendation")


class Observation(BaseModel):
    """Full observation returned by reset() and step()."""
    task_id: str = Field(..., description="Which task is currently running")
    waypoint_index: int = Field(..., ge=0, description="Current waypoint number (0-indexed)")
    total_waypoints: int = Field(..., gt=0, description="Total waypoints in this task")
    vehicle: VehicleState = Field(..., description="Current vehicle telemetry")
    segment_ahead: SegmentAhead = Field(..., description="Upcoming segment data")
    steps_remaining: int = Field(..., ge=0, description="Waypoint decisions left")
    last_action_feedback: Optional[str] = Field(None, description="Result of previous action, or None on first step")
    task_objective: str = Field(..., description="Plain-English description of what the agent must accomplish")
    episode_terminated_early: bool = Field(False, description="True if a hard constraint was violated")

    # ── Round 2: Richer observation space ─────────────────────────────────────
    cumulative_energy_used_wh: float = Field(0.0, ge=0.0, description="Total energy consumed so far in Wh")
    cumulative_solar_recovered_wh: float = Field(0.0, ge=0.0, description="Total solar energy recovered so far in Wh")
    estimated_range_remaining_km: float = Field(0.0, ge=0.0, description="Estimated range remaining based on current SoC and avg consumption")
    track_progress_pct: float = Field(0.0, ge=0.0, le=100.0, description="How far through the track (0-100%)")
    last_segment_efficiency: float = Field(0.0, ge=0.0, description="Energy efficiency of last segment in Wh/km (0 on first step)")

    # ── Round 2: Long-horizon planning (look-ahead window) ────────────────────
    upcoming_segments: List[SegmentAhead] = Field(
        default_factory=list,
        description="Preview of next 2-3 upcoming segments for long-horizon planning"
    )

    # ── Round 2: Multi-agent — Weather advisor forecast ───────────────────────
    weather_forecast: Optional[WeatherForecast] = Field(
        None,
        description="Structured advisory from the onboard weather/route advisor agent"
    )

    # ── Round 2: Episode seed (for reproducibility) ───────────────────────────
    episode_seed: Optional[int] = Field(None, description="Random seed used for this episode (None = deterministic base)")


class Action(BaseModel):
    """The agent's configuration decision for the upcoming track segment."""
    target_cruise_speed_kph: float = Field(..., ge=10.0, le=120.0, description="Target cruise speed in km/h")
    cooling_system_level: int = Field(..., ge=0, le=2, description="0=off, 1=moderate, 2=maximum")
    solar_routing_mode: str = Field(..., description="'direct_to_motor' or 'charge_battery'")

    @field_validator("solar_routing_mode")
    @classmethod
    def validate_routing_mode(cls, v: str) -> str:
        allowed = {"direct_to_motor", "charge_battery"}
        if v not in allowed:
            raise ValueError(f"solar_routing_mode must be one of {allowed}, got '{v}'")
        return v


class Reward(BaseModel):
    """Scoring output after each step and at episode end."""
    score: float = Field(..., ge=0.0, le=1.0, description="Score for this step or final episode score")
    is_done: bool = Field(..., description="True if episode has ended")
    is_success: bool = Field(..., description="True only if task completed without constraint violations")
    feedback: str = Field(..., description="Human-readable explanation of the score")
    battery_soc_remaining_pct: float = Field(..., description="Battery SoC at this point")
    battery_temp_c: float = Field(..., description="Battery temperature at this point")
    distance_covered_km: float = Field(..., description="Total distance covered")

    # ── New field: rubric breakdown (populated on final step) ──────────────────
    rubric: Optional[dict] = Field(None, description="Rubric sub-score breakdown (only on final reward)")


class EpisodeResult(BaseModel):
    """Full episode summary returned when an episode completes."""
    task_id: str
    final_score: float = Field(..., ge=0.0, le=1.0)
    success: bool
    total_steps: int
    distance_covered_km: float
    final_soc_pct: float
    final_battery_temp_c: float
    termination_reason: str
    step_scores: List[float] = Field(default_factory=list)

    # ── New field: rubric breakdown ───────────────────────────────────────────
    rubric: Optional[dict] = Field(None, description="Rubric sub-score breakdown")