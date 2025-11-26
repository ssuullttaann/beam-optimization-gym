"""
LHC Beam Steering Optimization with Reinforcement Learning
===========================================================
Multi-objective optimization: Beam stability + Energy efficiency
Realistic physics: Magnet hysteresis, power constraints, beam blow-up

Author: Sultan Bahomaid
Project: CERN High-Luminosity LHC Optimization
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
import json


# ==========================================
# PART 1: Realistic LHC Beam Physics
# ==========================================

class RealisticBeamDynamics:
    """
    Simulates LHC beam optics with realistic physics:
    - Coupling between position and angle
    - Magnet hysteresis effects
    - Dispersive effects
    - Beam blow-up (emittance growth)
    """

    def __init__(self, num_magnets=20, coupling_strength=0.15):
        self.num_magnets = num_magnets
        self.coupling_strength = coupling_strength  # Cross-plane coupling
        self.max_emittance = 5.0  # μm (beam blow-up limit)
        self.last_kick = 0.0  # For hysteresis

    def apply_magnet_dynamics(self, state, action):
        """
        state: [x, x', y, y', emittance]
        action: [horizontal_kick, vertical_kick, power_level]
        """
        x, xp, y, yp, emittance = state
        h_kick, v_kick, power = action

        # 1. Magnet Hysteresis (nonlinear response)
        # Stronger magnets have harder-to-saturate response
        h_kick_effective = h_kick * (1.0 - 0.3 * np.tanh(abs(self.last_kick)))
        self.last_kick = h_kick

        # 2. Apply kicks
        xp += h_kick_effective * 0.5
        yp += v_kick * 0.5

        # 3. Coupling effects (misalignment induces cross-plane motion)
        xp += y * self.coupling_strength * 0.01
        yp += x * self.coupling_strength * 0.01

        # 4. Drift space (1 meter to next magnet)
        drift_distance = 1.0
        x += xp * drift_distance
        y += yp * drift_distance

        # 5. Realistic noise (measurement + magnet jitter)
        x += np.random.normal(0, 0.05)
        y += np.random.normal(0, 0.05)
        xp += np.random.normal(0, 0.02)
        yp += np.random.normal(0, 0.02)

        # 6. Beam blow-up (emittance growth from high-amplitude oscillations)
        # High-amplitude beams lose more particles
        max_amplitude = np.sqrt(x**2 + y**2)
        blow_up_factor = 1.0 + 0.05 * max(0, (max_amplitude - 2.0) ** 2)
        emittance *= blow_up_factor

        # 7. Add power efficiency penalty (higher power → more emittance growth)
        emittance += 0.01 * (power ** 2)

        return np.array([x, xp, y, yp, emittance], dtype=np.float32)


# ==========================================
# PART 2: Multi-Objective LHC Environment
# ==========================================

class LHCBeamEnv(gym.Env):
    """
    Multi-objective beam steering environment:
    - Minimize beam offset (stability)
    - Minimize energy consumption (power efficiency)
    - Prevent beam blow-up (emittance control)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_magnets=20, multi_objective_weight=0.5):
        super().__init__()

        self.num_magnets = num_magnets
        self.multi_objective_weight = multi_objective_weight  # 0.5 = equal stability/efficiency
        self.dynamics = RealisticBeamDynamics(num_magnets=num_magnets)

        # Action: [horizontal_kick, vertical_kick, power_level]
        # All normalized to [-1, 1], scaled to physical units
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # State: [x, x', y, y', emittance]
        self.observation_space = spaces.Box(
            low=-20.0, high=20.0, shape=(5,), dtype=np.float32
        )

        self.state = None
        self.step_count = 0
        self.max_steps = num_magnets
        self.episode_history = {"positions": [], "power": [], "emittance": []}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Inject beam with realistic misalignment
        # Position: ±2mm, Angle: ±1 mrad, Emittance: initial 3.5 μm
        self.state = np.array([
            np.random.uniform(-2.0, 2.0),  # x
            np.random.uniform(-1.0, 1.0),  # x'
            np.random.uniform(-2.0, 2.0),  # y
            np.random.uniform(-1.0, 1.0),  # y'
            3.5  # initial emittance
        ], dtype=np.float32)

        self.step_count = 0
        self.episode_history = {"positions": [], "power": [], "emittance": []}

        return self.state, {}

    def step(self, action):
        # Scale actions to physical units
        h_kick = action[0] * 0.5  # mrad
        v_kick = action[1] * 0.5  # mrad
        power = np.abs(action[2])  # [0, 1] normalized power

        # Apply realistic beam dynamics
        self.state = self.dynamics.apply_magnet_dynamics(
            self.state,
            [h_kick, v_kick, power]
        )

        # Unpack state
        x, xp, y, yp, emittance = self.state

        # Track history
        self.episode_history["positions"].append(np.sqrt(x**2 + y**2))
        self.episode_history["power"].append(power)
        self.episode_history["emittance"].append(emittance)

        # ====== MULTI-OBJECTIVE REWARD ======

        # Objective 1: Beam Stability (position + angle control)
        position_error = x**2 + y**2
        angle_error = 0.1 * (xp**2 + yp**2)
        stability_reward = -(position_error + angle_error)

        # Objective 2: Energy Efficiency (minimize power consumption)
        efficiency_reward = -power

        # Objective 3: Emittance Control (prevent beam blow-up)
        emittance_penalty = -0.5 * max(0, (emittance - 4.0) ** 2)

        # Combined reward with tunable weighting
        reward = (
            self.multi_objective_weight * stability_reward +
            (1.0 - self.multi_objective_weight) * efficiency_reward +
            emittance_penalty
        )

        # Add control penalty (minimize magnet kicks)
        reward -= 0.01 * (action[0]**2 + action[1]**2)

        # ====== TERMINATION CONDITIONS ======

        terminated = False

        # Beam loss (hits pipe wall at ±10mm)
        if abs(x) > 10.0 or abs(y) > 10.0:
            terminated = True
            reward -= 50.0

        # Excessive emittance growth (beam quality degraded)
        if emittance > 8.0:
            terminated = True
            reward -= 30.0

        truncated = (self.step_count >= self.max_steps - 1)
        self.step_count += 1

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Step {self.step_count} | x={self.state[0]:.3f}mm | "
              f"y={self.state[2]:.3f}mm | emittance={self.state[4]:.2f}μm")


# ==========================================
# PART 3: Baseline Controllers
# ==========================================

class PIDController:
    """Classical PID controller for comparison"""

    def __init__(self, kp=0.3, ki=0.05, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0

    def compute_action(self, state):
        x, xp, y, yp, emittance = state

        # Simple PID on position
        error = np.sqrt(x**2 + y**2)
        self.integral += error
        derivative = error - self.last_error
        self.last_error = error

        pid_output = (self.kp * error +
                     self.ki * self.integral +
                     self.kd * derivative)

        # Clip and return as action
        h_kick = np.clip(pid_output / 0.5, -1.0, 1.0)
        return np.array([h_kick, h_kick * 0.5, 0.2], dtype=np.float32)


class RandomController:
    """Random baseline"""

    def compute_action(self, state):
        return np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)


# ==========================================
# PART 4: Logging & Evaluation
# ==========================================

class LivePlottingCallback(BaseCallback):
    """Track learning progress with LIVE real-time plotting"""

    def __init__(self, update_freq=500):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.update_freq = update_freq
        self.timesteps_list = []

        # Setup live plotting
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Episode Reward')
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2, label='Moving Avg (10 eps)')

        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Episode Reward')
        self.ax1.set_title('PPO Learning Curve - Episode Rewards')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()

        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Moving Average Reward')
        self.ax2.set_title('PPO Learning Curve - Trend (10-Episode MA)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()

        self.fig.suptitle('LIVE Training Progress', fontsize=14, fontweight='bold')
        self.fig.tight_layout()

    def _on_step(self) -> bool:
        # Log reward from info dict
        if "episode" in self.model.env.buf_infos[0]:
            episode_reward = self.model.env.buf_infos[0]["episode"]["r"]
            episode_length = self.model.env.buf_infos[0]["episode"]["l"]
            self.episode_rewards.append(float(episode_reward))
            self.episode_lengths.append(int(episode_length))
            self.timesteps_list.append(self.num_timesteps)

            # Update plot every 5 episodes
            if len(self.episode_rewards) % 5 == 0:
                episodes = np.arange(len(self.episode_rewards))

                # Plot 1: Raw rewards
                self.line1.set_xdata(episodes)
                self.line1.set_ydata(self.episode_rewards)
                self.ax1.set_xlim(0, len(self.episode_rewards))
                self.ax1.set_ylim(min(self.episode_rewards) - 10, max(self.episode_rewards) + 10)

                # Plot 2: Moving average
                if len(self.episode_rewards) >= 10:
                    moving_avg = np.convolve(self.episode_rewards,
                                            np.ones(10)/10, mode='valid')
                    self.line2.set_xdata(episodes[9:])
                    self.line2.set_ydata(moving_avg)
                    self.ax2.set_xlim(0, len(self.episode_rewards))
                    self.ax2.set_ylim(min(moving_avg) - 10, max(moving_avg) + 10)

                # Redraw
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

                # Print progress
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"  Episodes: {len(self.episode_rewards):>3d} | "
                      f"Timesteps: {self.num_timesteps:>6d} | "
                      f"Latest Reward: {episode_reward:>8.2f} | "
                      f"Avg (last 10): {avg_reward:>8.2f}")

        return True


def evaluate_controller(controller, env, num_episodes=10, name="Controller"):
    """Run evaluation and return metrics"""

    rewards_list = []
    max_positions = []
    final_emittances = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        max_pos = 0.0

        for step in range(env.max_steps):
            if hasattr(controller, 'predict'):  # RL model
                action, _ = controller.predict(obs, deterministic=True)
            else:  # Classical controller
                action = controller.compute_action(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            max_pos = max(max_pos, np.sqrt(obs[0]**2 + obs[2]**2))

            if terminated or truncated:
                break

        rewards_list.append(episode_reward)
        max_positions.append(max_pos)
        final_emittances.append(obs[4])

    return {
        "name": name,
        "mean_reward": np.mean(rewards_list),
        "std_reward": np.std(rewards_list),
        "max_position_mean": np.mean(max_positions),
        "final_emittance_mean": np.mean(final_emittances),
    }


# ==========================================
# PART 5: Main Training & Visualization
# ==========================================

def main():
    print("=" * 70)
    print("LHC BEAM STEERING OPTIMIZATION WITH REINFORCEMENT LEARNING")
    print("=" * 70)

    # Initialize environment
    print("\n[1] Initializing LHC Beam Environment...")
    env = LHCBeamEnv(num_magnets=20, multi_objective_weight=0.6)

    # Train PPO agent
    print("[2] Training PPO Agent (50,000 timesteps)...")
    print("    (Watch progress below - episodes increase as training continues)\n")

    ppo_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=0  # Suppress verbose output, use callback instead
    )

    callback = LivePlottingCallback(update_freq=500)
    ppo_model.learn(total_timesteps=50000, callback=callback)
    print("\n    ✓ Training Complete!")

    # Evaluate all controllers
    print("\n[3] Evaluating Controllers...")
    results = []

    # PPO
    print("    - PPO Agent...", end=" ", flush=True)
    ppo_result = evaluate_controller(ppo_model, env, num_episodes=20, name="PPO")
    results.append(ppo_result)
    print("✓")

    # PID Baseline
    print("    - PID Controller...", end=" ", flush=True)
    pid_controller = PIDController(kp=0.3, ki=0.05, kd=0.1)
    pid_result = evaluate_controller(pid_controller, env, num_episodes=20, name="PID")
    results.append(pid_result)
    print("✓")

    # Random Baseline
    print("    - Random Controller...", end=" ", flush=True)
    random_controller = RandomController()
    random_result = evaluate_controller(random_controller, env, num_episodes=20, name="Random")
    results.append(random_result)
    print("✓")

    # Print results table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Controller':<15} {'Mean Reward':<15} {'Max Position':<15} {'Emittance':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<15} {r['mean_reward']:>10.2f}±{r['std_reward']:>3.2f} "
              f"{r['max_position_mean']:>10.2f}mm      {r['final_emittance_mean']:>10.2f}μm")

    # Visualization
    print("\n[4] Generating Visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test one episode with PPO
    obs, _ = env.reset()
    positions = []
    emittances = []
    powers = []

    for step in range(env.max_steps):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        positions.append(np.sqrt(obs[0]**2 + obs[2]**2))
        emittances.append(obs[4])
        powers.append(np.abs(action[2]))

        if terminated or truncated:
            break

    steps = np.arange(len(positions))

    # Plot 1: Beam Position
    axes[0, 0].plot(steps, positions, 'b-', linewidth=2, label='Beam Offset')
    axes[0, 0].axhline(y=4, color='r', linestyle='--', label='Pipe Wall (±4mm)')
    axes[0, 0].axhline(y=-4, color='r', linestyle='--')
    axes[0, 0].axhline(y=0, color='g', linestyle=':', alpha=0.5, label='Target')
    axes[0, 0].set_xlabel('Magnet Index')
    axes[0, 0].set_ylabel('Beam Offset (mm)')
    axes[0, 0].set_title('Beam Trajectory (PPO Control)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Emittance Growth
    axes[0, 1].plot(steps, emittances, 'r-', linewidth=2, label='Emittance')
    axes[0, 1].axhline(y=4.0, color='orange', linestyle='--', label='Nominal (4μm)')
    axes[0, 1].axhline(y=8.0, color='darkred', linestyle='--', label='Limit (8μm)')
    axes[0, 1].set_xlabel('Magnet Index')
    axes[0, 1].set_ylabel('Emittance (μm)')
    axes[0, 1].set_title('Beam Quality (Emittance)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Power Consumption
    axes[1, 0].bar(steps, powers, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Magnet Index')
    axes[1, 0].set_ylabel('Normalized Power')
    axes[1, 0].set_title('Magnet Power Usage')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Comparison
    controller_names = [r['name'] for r in results]
    mean_rewards = [r['mean_reward'] for r in results]
    colors = ['green', 'orange', 'red']

    axes[1, 1].bar(controller_names, mean_rewards, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Mean Episode Reward')
    axes[1, 1].set_title('Controller Performance Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('lhc_beam_steering_results.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: lhc_beam_steering_results.png")

    # Save metrics to JSON (convert numpy types to native Python)
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    metrics = {
        "project": "LHC Beam Steering Optimization",
        "environment": {
            "num_magnets": 20,
            "state_dim": 5,
            "action_dim": 3,
            "features": ["position", "angle", "coupling", "hysteresis", "emittance_growth"]
        },
        "results": [convert_to_native(r) for r in results],
        "training": {
            "algorithm": "PPO",
            "timesteps": 50000,
            "learning_rate": 0.0003
        }
    }

    with open('lhc_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("    ✓ Saved: lhc_metrics.json")

    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    plt.show()


if __name__ == "__main__":
    main()