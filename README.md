# LHC Beam Steering Optimization with Reinforcement Learning

## 🚀 Project Overview
This project utilizes **Reinforcement Learning (PPO)** to optimize particle beam steering in a simulated Large Hadron Collider (LHC) environment. It addresses the multi-objective challenge of maintaining beam stability while minimizing power consumption and preventing beam blow-up.

## 🧠 Key Features
* **Realistic Physics Simulation:** Models magnet hysteresis, cross-plane coupling, and emittance growth.
* **Multi-Objective Reward Function:** Balances position error, energy efficiency, and beam quality.
* **Comparative Analysis:** Benchmarks PPO agent against classical PID controllers and random baselines.
* **Live Visualization:** Includes real-time plotting of training progress and beam trajectory.

## 🛠️ Tech Stack
* **Python**
* **Stable Baselines 3** (PPO Algorithm)
* **Gymnasium** (Custom Environment)
* **Matplotlib** (Data Visualization)

## 📊 Results
The PPO agent successfully stabilizes the beam within the ±10mm pipe limits while maintaining lower emittance growth compared to classical PID control.

## 💻 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

python Beam_train.py
