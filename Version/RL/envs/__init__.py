from gymnasium.envs.registration import register

from .barrier_nav_env import ACTION_NAMES, BarrierNavEnv

register(id="BarrierNav-v0", entry_point="envs.barrier_nav_env:BarrierNavEnv")

__all__ = ["BarrierNavEnv", "ACTION_NAMES"]
