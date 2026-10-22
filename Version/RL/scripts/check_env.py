"""Run gymnasium's env checker plus a quick random-rollout summary."""
import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401


def main():
    env = gym.make("BarrierNav-v0")
    check_env(env.unwrapped, skip_render_check=True)
    print("gymnasium check_env: passed")
    print("observation space:", env.observation_space.shape, " action space:", env.action_space)

    for level in [0.0, 1.0, 2.0]:
        env = gym.make("BarrierNav-v0", noise_level=level)
        env.action_space.seed(0)
        outcomes, ndets = [], []
        for seed in range(50):
            env.reset(seed=seed)
            done = False
            while not done:
                _, _, term, trunc, info = env.step(env.action_space.sample())
                ndets.append(info["num_detections"])
                done = term or trunc
            outcomes.append("success" if info["is_success"] else
                            "collision" if info["collision"] else "timeout")
        counts = {k: outcomes.count(k) for k in ["success", "collision", "timeout"]}
        print(f"noise={level:.1f} random policy over 50 eps: {counts}, "
              f"mean detections/frame={np.mean(ndets):.2f}")


if __name__ == "__main__":
    main()
