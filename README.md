# LHC Beam Steering Optimization with Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Reinforcement Learning](https://img.shields.io/badge/RL-PPO-orange)
![Library](https://img.shields.io/badge/Gymnasium-Enabled-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A physics-informed reinforcement learning system for optimizing particle beam steering in the Large Hadron Collider (LHC). This project demonstrates how **PPO (Proximal Policy Optimization)** can solve the complex, multi-objective challenge of maintaining beam stability while minimizing power consumption and preventing beam degradation.

---

## 🎯 Problem Statement

The LHC operates at extreme scales—27 km in circumference with beams of trillions of protons traveling at 99.9999% the speed of light. Precise beam control requires real-time steering through hundreds of magnets while balancing competing objectives:

* **Beam Stability:** Keep particle beams centered within the vacuum pipe (±10mm tolerance).
* **Energy Efficiency:** Minimize power consumption to reduce operational costs and heat dissipation.
* **Beam Quality:** Prevent emittance growth (beam blow-up) which degrades luminosity and particle collision rates.

Traditional PID controllers struggle with these competing goals and the nonlinear physics. This project applies deep reinforcement learning to find optimal steering strategies.

---

## ✨ Key Features

### Realistic Physics Simulation
* **Magnet Hysteresis:** Nonlinear response based on magnet saturation history.
* **Cross-Plane Coupling:** Misalignments induce coupling between horizontal and vertical planes.
* **Dispersive Effects:** Position-dependent momentum focusing.
* **Beam Blow-Up:** Emittance growth from high-amplitude oscillations and power consumption.
* **Measurement Noise:** Realistic sensor noise (±0.05mm position, ±0.02 mrad angle).

### Multi-Objective Reward Design
The agent optimizes a weighted combination of three objectives:
> **Total Reward = w₁ × Stability + w₂ × Efficiency + Emittance Control**

* **Tunable weighting:** Default is 60% stability, 40% efficiency.
* **Soft penalty:** For emittance exceeding 4 μm nominal.
* **Hard termination:** At 8 μm (beam quality limit).

### Agent & Baselines
* **PPO Agent:** 50k timestep training with live progress visualization.
* **PID Controller:** Classical tuned baseline (`Kp=0.3`, `Ki=0.05`, `Kd=0.1`).
* **Random Controller:** Lower-bound performance reference.

---

## 📊 Results

The trained PPO agent achieves significantly better performance than classical baselines:

| Metric | PPO | PID | Random |
| :--- | :--- | :--- | :--- |
| **Mean Episode Reward** | **~150–200** | ~50–80 | ~-200 |
| **Avg Max Beam Position** | **~1.5 mm** | ~3.2 mm | ~8+ mm |
| **Final Emittance (μm)** | **~4.2** | ~4.8 | ~6.5+ |

**Key Findings:**
* PPO stabilizes beams within **±2mm** (half the pipe tolerance) while maintaining low power.
* Achieves **~40% better reward** than PID through learned trade-offs.
* Learns to apply corrective kicks efficiently early, reducing energy waste later in the sequence.
* Prevents catastrophic beam loss (>10mm deflection or >8 μm emittance growth).

---

## 🛠️ Tech Stack
* **Python 3.8+**
* **Gymnasium:** Reinforcement learning environment framework.
* **Stable-Baselines3:** PPO and SAC implementations.
* **NumPy:** Numerical computations.
* **Matplotlib:** Real-time and post-training visualization.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd lhc-beam-steering

# Install dependencies
pip install -r requirements.txt
