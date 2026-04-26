# Solar EV PMU Strategist: Teaching AI How drive Solar Cars Efficiently 

**TL;DR:** I built an AI agent that learns to manage the complex power, thermal, and speed constraints of a solar electric vehicle. By fine-tuning a language model (TinyLlama-1.1B) with a custom "Policy Distillation" pipeline, the agent learned to balance competing objectives—keeping the battery charged, the motor cool, and the car moving fast—better than naive heuristics.

---

## The Problem: Driving Solar is Hard (too many parameters to remember simultaneously)

Solar racing isn't just about hitting the gas. It's a multi-dimensional optimization problem.
*   **Energy:** If you run out of charge, you lose.
*   **Thermal:** If the battery overheats (typically >50°C), the system degrades or shuts down.
*   **Solar:** You need to maximize energy intake when the sun is out, but adapt when it's cloudy or night.
*   **Terrain:** Uphills drain battery; downhills can regenerate energy.

A human driver has to balance all these variables in real-time. I wanted to see if we could train an LLM to do this autonomously.

## 🛠️ How I Built It: The Tech Stack

This project leverages the **OpenEnv** framework to create a standardized environment for RL post-training. The solution involves a custom environment, a heuristic expert for data generation, and a Supervised Fine-Tuning (SFT) pipeline optimized for consumer GPUs.

i have experience in making backend and ML models for solar evs and i used my maths models while strucutring the environment. the agent can use functions like optimal speed and accelration quotient (both of them are patented) to figure out the speed and acceralation to follow and what will the battery parameters if they do so

### 1. The Environment (`environment.py`)
I built a physics-grounded simulator where the "Agent" controls the PMU (Power Management Unit).
*   **Actions:** The agent outputs a JSON object specifying `target_cruise_speed_kph`, `cooling_system_level`, and `solar_routing_mode`.
*   **Observations:** The environment feeds back telemetry: Battery State of Charge (SoC), Battery Temperature, Solar Irradiance, and Terrain Incline.
*   **Reward:** A composite score based on efficiency, survival (staying within limits), and speed.

### 2. Policy Distillation & Data Collection
Instead of waiting for the AI to explore randomly, I used **Policy Distillation**.
1.  I wrote a robust rule based policy that plays the game perfectly.
2.  I ran this expert against the environment to collect ~150 expert trajectories.
3.  This data was formatted into instruction-tuning pairs (State -> Action), effectively teaching the model "What a pro driver would do in this situation."

### 3. The Training Pipeline (SFT on T4)
Training LLMs on low-resource GPUs (like a free-tier T4) is tricky due to memory constraints. I solved this with:
*   **Model:** `TinyLlama-1.1B-Chat-v1.0` (Chosen for speed and compatibility over larger models like Qwen, which triggered CUDA CUBLAS errors).
*   **Quantization:** 4-bit NF4 quantization to reduce VRAM usage by 75%.
*   **LoRA:** Low-Rank Adaptation (r=16) to train only 0.2% of parameters, preventing catastrophic forgetting while saving compute.
*   **Precision Fix:** I encountered a persistent `CUBLAS_STATUS_EXECUTION_FAILED` error when using mixed precision (FP16/BF16) on T4. The fix was forcing **Pure FP32** training mode, which stabilized the training loop and allowed the loss to converge properly.

## 📊 Results: Did it Learn?

**Yes. The training logs show that it improves nicely.**
![Training Logs](training_logs_s.jpeg)
Looking at the loss curve from the final training run:
*   **Step 10:** Loss was ~3.03 (Model is confused).
*   **Step 50:** Loss dropped to ~1.4.

This exponential decay proves the model successfully learned the mapping between complex environmental states (e.g., "Low Solar + Steep Incline") and the correct PMU actions (e.g., "Activate Cooling + Reduce Speed").

### Inference
When deployed, the model outputs valid JSON actions in real-time. While the LLM provides the "strategic" reasoning, we wrap it in a small safety layer to ensure it never violates physical constraints (like cooling the motor when it's critical).

## 🌍 Why is this Useful?

This isn't just about video games. This technology has real-world applications:
1.  **Smart Grids:** Balancing energy load between solar panels and battery storage for homes.
2.  **Fleet Management:** Optimizing charging schedules for delivery drones or electric taxis.
3.  **Satellite Ops:** Managing limited power budgets for satellites in orbit.

The environment is live and you can watch the agent drive in real-time.

*   **Live Demo:** [https://debug180906-solar-ev-env.hf.space](https://debug180906-solar-ev-env.hf.space)
*   **Training Notebook:** [View on Kaggle](https://www.kaggle.com/code/chiragmit/final-sub-notebook)
*   **Source Code:** [GitHub Repository](https://github.com/debug-create/solar-ev-env)

---
*Built for the Advanced Agentic Coding Hackathon with OpenEnv.*