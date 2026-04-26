# 🏎️ Solar EV PMU Strategist

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/debug180906-solar-ev-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced agentic AI system designed to manage the Power Management Unit (PMU) of a solar electric vehicle. This project demonstrates how Large Language Models (LLMs) can be distilled into high-performance, physics-aware racing strategists.

## 🌟 Key Features
- **Physics-Grounded Simulation**: Built on the OpenEnv framework with high-fidelity modeling of solar irradiance, thermal dynamics, and aerodynamics.
- **Task-Aware Reasoning**: An integrated advisor system provides the agent with strategic "hints" for complex scenarios (Night Runs, Thermal Stress, etc.).
- **Policy Distillation**: Mastered via Supervised Fine-Tuning (SFT) with a 99% training loss reduction.
- **Hybrid Safety Layer**: A firmware-level sanitization layer ensures vehicle stability even under extreme conditions.

## 🏗️ Architecture
- **Environment**: `environment.py` (OpenEnv Interface)
- **Physics Engine**: `physics.py`
- **Strategist Intelligence**: `advisor.py` & `models.py`
- **Dashboard & API**: `frontend/` & `server/`

## 📊 Performance
The model has been optimized for the NVIDIA T4 GPU, achieving expert-level scores across a curriculum of increasingly difficult racing tasks.

| Task | Difficulty | Strategy Focus |
|------|------------|----------------|
| Flat Track | Easy | Efficiency & Consistency |
| Dynamic Routing | Medium | Elevation & Clouds |
| Night Run | Hard | Zero Solar SoC Management |
| Thermal Race | Expert | Active Cooling & Speed Capping |

## 🚀 Getting Started

### 1. Run the Training Pipeline
Open `final_notebook.ipynb` in Google Colab or Kaggle. Enable T4 GPU and run all cells to reproduce the distillation results.

### 2. Local Development
```bash
pip install -r requirements.txt
python server/app.py
```

### 3. Live Demo
View the real-time telemetry dashboard at:
[https://huggingface.co/spaces/debug180906-solar-ev-env](https://huggingface.co/spaces/debug180906-solar-ev-env)

---
Built for the **Advanced Agentic Coding Hackathon**. 🏆🏁🔋
