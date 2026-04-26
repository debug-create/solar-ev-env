"""
server/app.py
FastAPI server for the Solar EV PMU Strategist Environment.

Uses the official OpenEnv create_app() factory pattern to enable
WebSocket-based concurrent sessions for GRPOTrainer parallel rollouts.

Custom endpoints (/health, /tasks, /grader, /baseline, /train)
are added to the app after create_app() returns it.

Legacy HTTP endpoints (/api/reset, /api/step) are provided alongside
the OpenEnv built-in ones for backward compatibility with baseline.py
and inference.py.

Round 2 — OpenEnv Architecture Migration.
"""

import os
import sys
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action, Observation
from environment import SolarEVEnvironment
from tasks import TASKS
from advisor import generate_forecast

# ═══════════════════════════════════════════════════════════════════════════════
# OpenEnv create_app() — Official pattern for concurrent WebSocket sessions
# ═══════════════════════════════════════════════════════════════════════════════

# Try the official OpenEnv create_app path first.
# If openenv-core is not installed (e.g. local dev), fall back to raw FastAPI.

_USE_OPENENV = False
try:
    from openenv.core.env_server.http_server import create_app

    max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "64"))

    app = create_app(
        SolarEVEnvironment,          # factory: class is callable
        Action,                       # our Pydantic Action is model_dump()-compatible
        Observation,                  # our Pydantic Observation is model_dump()-compatible
        env_name="solar-ev-env",
        max_concurrent_envs=max_concurrent,
    )
    _USE_OPENENV = True

except Exception:
    # Fallback: raw FastAPI — used when openenv-core is not installed
    from fastapi import FastAPI

    app = FastAPI(
        title="Solar EV Environment",
        description=(
            "Solar Electric Vehicle Power Management environment. "
            "Fallback mode — openenv-core not available."
        ),
        version="2.0.0",
    )

    # Shared instance for fallback mode only
    _fallback_env = SolarEVEnvironment()

    @app.post("/reset")
    def _fallback_reset(
        task_id: str = "flat_track_easy",
        randomize: bool = False,
        seed: Optional[int] = None,
    ):
        try:
            obs = _fallback_env.reset(task_id=task_id, randomize=randomize, seed=seed)
            return obs.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post("/step")
    def _fallback_step(action: Action):
        try:
            obs, reward = _fallback_env.step(action)
            return {"observation": obs.model_dump(), "reward": reward.model_dump()}
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.get("/state")
    def _fallback_state():
        return _fallback_env.state()


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy HTTP endpoints — backward compatible with baseline.py / inference.py
# These always create a FRESH environment per request (stateless HTTP).
# ═══════════════════════════════════════════════════════════════════════════════

# Shared instance for legacy endpoints (stateful across /api/reset → /api/step)
_legacy_env = SolarEVEnvironment()

if os.getenv("ENABLE_DEMO_INTERFACE", "true").lower() in {"1", "true", "yes"}:
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount("/demo", StaticFiles(directory="frontend", html=True), name="frontend")
    except Exception as exc:
        print(f"[WARN] Arcade frontend not mounted: {exc}")


@app.post("/api/reset")
def api_reset(
    task_id: str = "flat_track_easy",
    randomize: bool = False,
    seed: Optional[int] = None,
):
    """Legacy HTTP reset. Used by baseline.py and inference.py.

    Pass randomize=true and seed for reproducible stochastic episodes.
    Without randomize, episodes are fully deterministic.
    """
    try:
        obs = _legacy_env.reset(task_id=task_id, randomize=randomize, seed=seed)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Reset Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}\n{traceback.format_exc()}")


@app.post("/advisor")
def advisor_forecast():
    """Get weather advisor forecast for current environment state.

    Returns a structured forecast with risk assessments and strategy
    recommendations from the weather/route advisor agent.
    Multi-agent interaction: the PMU strategist consults the weather advisor.
    """
    if _legacy_env._vehicle_state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /api/reset first.")

    upcoming = _legacy_env._get_upcoming_segments()
    distance_remaining = _legacy_env._task.total_distance_km - _legacy_env._vehicle_state.distance_covered_km

    forecast = generate_forecast(
        upcoming_segments=upcoming,
        current_soc_pct=_legacy_env._vehicle_state.battery_soc_pct,
        current_battery_temp_c=_legacy_env._vehicle_state.battery_temp_c,
        distance_remaining_km=max(0.0, distance_remaining),
        cumulative_energy_wh=_legacy_env._cumulative_energy_wh,
        distance_covered_km=_legacy_env._vehicle_state.distance_covered_km,
    )
    return forecast.model_dump()


@app.post("/api/step")
def api_step(action: Action):
    """Legacy HTTP step. Used by baseline.py and inference.py."""
    try:
        obs, reward = _legacy_env.step(action)
        return {"observation": obs.model_dump(), "reward": reward.model_dump()}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=f"Step Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}\n{traceback.format_exc()}")


@app.get("/api/state")
def api_state():
    """Legacy HTTP state."""
    return _legacy_env.state()


# ═══════════════════════════════════════════════════════════════════════════════
# Custom endpoints — added to the app after create_app() returns it
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name": "solar-ev-env",
        "version": "2.0.0",
        "status": "running",
        "openenv_mode": _USE_OPENENV,
    }


@app.get("/health")
def health():
    """OpenEnv-compliant health endpoint returning environment metadata."""
    return {
        "name": "solar-ev-env",
        "version": "2.0.0",
        "status": "healthy",
        "openenv_mode": _USE_OPENENV,
        "environment_type": "sequential",
        "observation_type": "structured",
        "action_type": "structured",
        "reward_type": "continuous",
        "supports_concurrent_sessions": True,
        "max_concurrent_envs": int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
        "tasks_available": len(TASKS),
        "task_ids": list(TASKS.keys()),
        "endpoints": [
            "/reset", "/step", "/state", "/ws",
            "/api/reset", "/api/step", "/api/state",
            "/tasks", "/grader", "/baseline", "/health", "/train",
        ],
    }


@app.get("/tasks")
def tasks():
    """Return all tasks and the action schema."""
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "total_waypoints": len(t.segments),
                "total_distance_km": t.total_distance_km,
                "strict_time_limit": t.strict_time_limit,
                "time_budget_s": t.time_budget_s,
                "segments": [
                    {
                        "distance_km": s.distance_to_next_waypoint_km,
                        "incline_pct": s.average_incline_pct,
                        "solar_wm2": s.solar_irradiance_wm2,
                        "temp_c": s.ambient_temp_c
                    }
                    for s in t.segments
                ]
            }
            for t in TASKS.values()
        ],
        "action_schema": {
            "target_cruise_speed_kph": "float, range 10.0 to 120.0",
            "cooling_system_level": "int — 0=off, 1=moderate (50W), 2=maximum (150W)",
            "solar_routing_mode": "string — 'direct_to_motor' or 'charge_battery'",
        },
    }


@app.post("/grader")
def grader():
    """Score the completed episode with rubric breakdown."""
    result = _legacy_env.get_episode_result()
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="No completed episode found. Run /api/reset + /api/step through all waypoints first.",
        )
    return result.model_dump()


@app.post("/baseline")
async def baseline():
    """Run the LLM baseline agent on all 5 tasks."""
    try:
        from baseline import run_baseline

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            scores = await loop.run_in_executor(pool, run_baseline)
        return {"baseline_scores": scores}
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/train")
async def train():
    """
    Self-improvement endpoint: runs heuristic strategies across all tasks,
    identifies the hardest segments, and returns actionable insights.
    """
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, _run_training_analysis)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


def _run_training_analysis() -> dict:
    """
    Run multiple heuristic strategies on each task using a private
    environment instance. Identifies hardest segments and recommends
    optimal actions for training pipeline guidance.
    """
    strategies = [
        {"name": "conservative", "speed": 45.0, "cooling": 1, "routing": "direct_to_motor"},
        {"name": "moderate", "speed": 60.0, "cooling": 1, "routing": "direct_to_motor"},
        {"name": "aggressive", "speed": 80.0, "cooling": 2, "routing": "charge_battery"},
        {"name": "eco_crawl", "speed": 30.0, "cooling": 0, "routing": "direct_to_motor"},
        {"name": "balanced_charge", "speed": 55.0, "cooling": 1, "routing": "charge_battery"},
    ]

    train_env = SolarEVEnvironment()
    all_results = []

    for task_id in TASKS:
        task_analysis = {
            "task_id": task_id,
            "strategy_scores": {},
            "segment_analysis": [],
            "best_strategy": None,
            "best_score": 0.0,
        }

        for strat in strategies:
            action = Action(
                target_cruise_speed_kph=strat["speed"],
                cooling_system_level=strat["cooling"],
                solar_routing_mode=strat["routing"],
            )

            obs = train_env.reset(task_id=task_id)
            segment_data = []

            while obs.steps_remaining > 0 and not obs.episode_terminated_early:
                obs, reward = train_env.step(action)
                segment_data.append({
                    "waypoint": obs.waypoint_index,
                    "score": reward.score,
                    "soc": reward.battery_soc_remaining_pct,
                    "temp": reward.battery_temp_c,
                })
                if reward.is_done:
                    break

            ep = train_env.get_episode_result()
            final_score = ep.final_score if ep else 0.0001
            task_analysis["strategy_scores"][strat["name"]] = final_score

            if final_score > task_analysis["best_score"]:
                task_analysis["best_score"] = final_score
                task_analysis["best_strategy"] = strat["name"]

            if strat["name"] == "moderate":
                task_analysis["segment_analysis"] = segment_data

        if task_analysis["segment_analysis"]:
            sorted_segs = sorted(task_analysis["segment_analysis"], key=lambda s: s["score"])
            task_analysis["hardest_segments"] = sorted_segs[:3]

        task_analysis["lessons"] = _generate_lessons(task_analysis)
        all_results.append(task_analysis)

    train_env.close()

    return {
        "training_analysis": all_results,
        "total_tasks_analyzed": len(all_results),
        "note": "Use these insights to guide LLM fine-tuning. Hardest segments need speed reduction and aggressive cooling.",
    }


def _generate_lessons(analysis: dict) -> list:
    """Generate actionable lessons from training analysis."""
    lessons = []
    scores = analysis.get("strategy_scores", {})
    best = analysis.get("best_strategy", "moderate")
    worst_strat = min(scores, key=scores.get) if scores else "unknown"

    if best == "conservative":
        lessons.append("This task rewards energy conservation. Keep speeds under 50kph.")
    elif best == "aggressive":
        lessons.append("This task has a tight time budget. Higher speeds are needed despite energy cost.")
    elif best == "eco_crawl":
        lessons.append("Extreme conservation works here. Minimize speed and cooling.")
    elif best == "balanced_charge":
        lessons.append("Routing solar to battery storage helps on this track profile.")
    else:
        lessons.append("A balanced moderate strategy works best for this task.")

    if worst_strat == "aggressive" and scores.get("aggressive", 1.0) < 0.01:
        lessons.append("WARNING: Aggressive speed causes thermal failure on this track.")
    if worst_strat == "eco_crawl" and scores.get("eco_crawl", 1.0) < 0.01:
        lessons.append("WARNING: Crawling speed may exceed time budget on timed tracks.")

    hardest = analysis.get("hardest_segments", [])
    if hardest:
        worst_wp = hardest[0].get("waypoint", "?")
        lessons.append(f"Hardest segment is around waypoint {worst_wp}. Consider reducing speed there.")

    return lessons


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
