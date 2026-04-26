"""
physics.py
Physics engine for the Solar EV environment.
Includes:
- Dynamic motor efficiency curve (speed-interpolated)
- Temperature-dependent rolling resistance
- Solar thermal degradation model
- Regenerative braking
- Active cooling system
All calculations are deterministic — same inputs always produce same outputs.
"""

from dataclasses import dataclass

# ── Vehicle constants ──────────────────────────────────────────────────────────
VEHICLE_MASS_KG = 300
DRAG_COEFFICIENT = 0.2
FRONTAL_AREA_M2 = 1.1
AIR_DENSITY = 1.225
BASE_ROLLING_RESISTANCE = 0.015
GRAVITY = 9.81

# ── Battery constants ──────────────────────────────────────────────────────────
BATTERY_CAPACITY_WH = 4960.0
BATTERY_MAX_TEMP_C = 58.0       # hard limit — episode fails above this
BATTERY_MIN_SOC_PCT = 20.0      # hard limit — episode fails below this
BATTERY_THERMAL_MASS = 800.0    # J/K

# ── Solar panel constants ──────────────────────────────────────────────────────
SOLAR_PANEL_AREA_M2 = 3.51
SOLAR_EFFICIENCY_BASE = 0.18    # 18% base efficiency

# ── Drivetrain ─────────────────────────────────────────────────────────────────
REGEN_EFFICIENCY = 0.65         # regenerative braking energy recovery
DRIVETRAIN_LOSS = 0.97          # mechanical drivetrain efficiency

# ── Cooling system ─────────────────────────────────────────────────────────────
COOLING_POWER_W = {0: 0, 1: 50, 2: 150}
COOLING_EFFECTIVENESS = {0: 0.0, 1: 0.4, 2: 0.9}

# ── Motor efficiency curve (speed in kph → efficiency fraction) ───────────────
# At low speeds motor is inefficient (44% at 10kph), peaks around 60-80kph (91%)
# This creates a realistic trap: going too slow wastes energy
MOTOR_EFFICIENCY_MAP = {
    10: 0.44,
    20: 0.64,
    30: 0.74,
    40: 0.81,
    50: 0.85,
    60: 0.90,
    80: 0.91,
    100: 0.91,
    120: 0.90,
}


@dataclass
class SegmentResult:
    """Output from simulating one track segment."""
    energy_consumed_wh: float
    energy_recovered_wh: float
    regen_energy_wh: float
    time_taken_s: float
    new_soc_pct: float
    new_battery_temp_c: float
    new_motor_temp_c: float
    new_speed_kph: float
    solar_power_w: float
    feedback: str


def interpolate_motor_efficiency(speed_kph: float) -> float:
    """
    Interpolate motor efficiency from the lookup table.
    Linear interpolation between known speed points.
    """
    speeds = sorted(MOTOR_EFFICIENCY_MAP.keys())
    if speed_kph <= speeds[0]:
        return MOTOR_EFFICIENCY_MAP[speeds[0]]
    if speed_kph >= speeds[-1]:
        return MOTOR_EFFICIENCY_MAP[speeds[-1]]
    for i in range(len(speeds) - 1):
        lo, hi = speeds[i], speeds[i + 1]
        if lo <= speed_kph < hi:
            t = (speed_kph - lo) / (hi - lo)
            return MOTOR_EFFICIENCY_MAP[lo] + t * (MOTOR_EFFICIENCY_MAP[hi] - MOTOR_EFFICIENCY_MAP[lo])
    return 0.90


def compute_rolling_resistance_coeff(ambient_temp_c: float) -> float:
    """
    Rolling resistance increases slightly with temperature deviation from 25C.
    Hotter road = more tire deformation = more resistance.
    """
    return BASE_ROLLING_RESISTANCE * (1 + 0.01 * abs(ambient_temp_c - 25))


def compute_solar_power(irradiance_wm2: float, ambient_temp_c: float) -> float:
    """
    Calculate solar power generated in Watts with thermal degradation.
    Panels lose 0.4% efficiency for every degree above 25C.
    """
    temp_penalty = 1 - 0.004 * max(0, ambient_temp_c - 25)
    actual_efficiency = SOLAR_EFFICIENCY_BASE * temp_penalty
    return max(0.0, irradiance_wm2 * SOLAR_PANEL_AREA_M2 * actual_efficiency)


def compute_traction_power(speed_kph: float, incline_pct: float, ambient_temp_c: float) -> float:
    """
    Calculate power needed to maintain speed against:
    - Aerodynamic drag
    - Temperature-adjusted rolling resistance
    - Gravity component on incline
    Returns power in Watts.
    """
    speed_ms = speed_kph / 3.6
    if speed_ms <= 0:
        return 0.0

    drag_force = 0.5 * DRAG_COEFFICIENT * FRONTAL_AREA_M2 * AIR_DENSITY * (speed_ms ** 2)
    rolling_coeff = compute_rolling_resistance_coeff(ambient_temp_c)
    rolling_force = rolling_coeff * VEHICLE_MASS_KG * GRAVITY
    gravity_force = VEHICLE_MASS_KG * GRAVITY * (incline_pct / 100.0)

    total_force = drag_force + rolling_force + gravity_force
    motor_eff = interpolate_motor_efficiency(speed_kph)
    power_watts = (total_force * speed_ms) / (motor_eff * DRIVETRAIN_LOSS)

    return max(power_watts, 0.0)


def compute_regen_power(speed_kph: float, incline_pct: float) -> float:
    """
    On downhills, calculate regenerative braking energy recovery.
    Only active when incline is negative (downhill).
    """
    if incline_pct >= 0:
        return 0.0
    speed_ms = speed_kph / 3.6
    gravity_force = VEHICLE_MASS_KG * GRAVITY * abs(incline_pct / 100.0)
    return gravity_force * speed_ms * REGEN_EFFICIENCY


def compute_battery_temperature(
    current_temp_c: float,
    ambient_temp_c: float,
    net_power_draw_w: float,
    cooling_level: int,
    time_s: float
) -> float:
    """
    Model battery temperature change over a time period.
    - Heat generated is proportional to power draw (I2R losses approximated)
    - Cooling system removes a fraction of generated heat
    - Passive cooling toward ambient temperature (Newton's law)
    """
    heat_generated = (net_power_draw_w * 0.05) * time_s
    cooling_removed = COOLING_EFFECTIVENESS[cooling_level] * heat_generated
    net_heat = heat_generated - cooling_removed
    delta_t = net_heat / BATTERY_THERMAL_MASS
    passive_cooling = (current_temp_c - ambient_temp_c) * 0.002 * time_s
    new_temp = current_temp_c + delta_t - passive_cooling
    return max(-10.0, round(new_temp, 2))


def simulate_segment(
    current_soc_pct: float,
    current_battery_temp_c: float,
    current_motor_temp_c: float,
    ambient_temp_c: float,
    segment_distance_km: float,
    segment_incline_pct: float,
    solar_irradiance_wm2: float,
    target_speed_kph: float,
    cooling_level: int,
    solar_routing_mode: str
) -> SegmentResult:
    """
    Main simulation function.
    Given current vehicle state and agent action,
    compute the resulting state after traversing a track segment.
    """
    distance_m = segment_distance_km * 1000.0
    speed_ms = target_speed_kph / 3.6
    time_s = distance_m / speed_ms if speed_ms > 0 else 9999

    traction_power_w = compute_traction_power(target_speed_kph, segment_incline_pct, ambient_temp_c)
    regen_power_w = compute_regen_power(target_speed_kph, segment_incline_pct)
    solar_power_w = compute_solar_power(solar_irradiance_wm2, ambient_temp_c)
    cooling_draw_w = COOLING_POWER_W[cooling_level]

    gross_power_demand_w = traction_power_w + cooling_draw_w

    if solar_routing_mode == "direct_to_motor":
        net_battery_draw_w = max(0.0, gross_power_demand_w - solar_power_w)
        solar_to_battery_w = 0.0
    else:
        net_battery_draw_w = gross_power_demand_w
        solar_to_battery_w = solar_power_w

    net_battery_draw_w = max(0.0, net_battery_draw_w - regen_power_w)

    energy_consumed_wh = (net_battery_draw_w * time_s) / 3600.0
    energy_from_solar_wh = (solar_to_battery_w * time_s) / 3600.0
    regen_energy_wh = (regen_power_w * time_s) / 3600.0

    current_energy_wh = (current_soc_pct / 100.0) * BATTERY_CAPACITY_WH
    new_energy_wh = max(0.0, min(BATTERY_CAPACITY_WH,
                                  current_energy_wh - energy_consumed_wh + energy_from_solar_wh))
    new_soc_pct = round((new_energy_wh / BATTERY_CAPACITY_WH) * 100.0, 2)

    new_battery_temp_c = compute_battery_temperature(
        current_battery_temp_c, ambient_temp_c, net_battery_draw_w, cooling_level, time_s
    )

    motor_heat = (traction_power_w * 0.08 * time_s) / BATTERY_THERMAL_MASS
    motor_cool = (current_motor_temp_c - ambient_temp_c) * 0.003 * time_s
    new_motor_temp_c = max(-10.0, round(current_motor_temp_c + motor_heat - motor_cool, 2))

    feedback = (
        f"Segment complete. Speed: {target_speed_kph}kph, "
        f"Energy used: {energy_consumed_wh:.1f}Wh, "
        f"Solar recovered: {(solar_power_w * time_s / 3600):.1f}Wh, "
        f"Regen: {regen_energy_wh:.1f}Wh, "
        f"SoC: {new_soc_pct:.1f}%, "
        f"Battery temp: {new_battery_temp_c:.1f}C"
    )

    return SegmentResult(
        energy_consumed_wh=round(energy_consumed_wh, 2),
        energy_recovered_wh=round(energy_from_solar_wh, 2),
        regen_energy_wh=round(regen_energy_wh, 2),
        time_taken_s=round(time_s, 1),
        new_soc_pct=new_soc_pct,
        new_battery_temp_c=new_battery_temp_c,
        new_motor_temp_c=new_motor_temp_c,
        new_speed_kph=target_speed_kph,
        solar_power_w=round(solar_power_w, 2),
        feedback=feedback,
    )