"""BarrierNav-v0: drive through a 2D area to a goal, seeing barriers only
through (noisy) YOLO-style detections from a forward camera."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .camera import DetectionNoise, SimCamera
from .config import deep_update, load_env_config
from .observation import build_observation, obs_bounds

FORWARD, LEFT, RIGHT, SLOW, STOP = range(5)
ACTION_NAMES = ["forward", "left", "right", "slow", "stop"]


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BarrierNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: dict | None = None, render_mode: str | None = None,
                 noise_level: float | None = None):
        cfg = load_env_config() if config is None else deep_update(load_env_config(), config)
        if noise_level is not None:
            cfg["noise"]["level"] = float(noise_level)
        self.cfg = cfg
        self.class_names = [c["name"] for c in cfg["classes"]]
        self.num_classes = len(self.class_names)
        self.max_dets = cfg["observation"]["max_detections"]

        self.camera = SimCamera(cfg["camera"], [c["height"] for c in cfg["classes"]])
        self.noise = DetectionNoise(cfg["noise"], self.num_classes)

        low, high = obs_bounds(self.num_classes, self.max_dets)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = None

        w = cfg["world"]
        self.goal = np.array([w["width"] / 2, w["length"] - w["goal_radius"] - 0.5])

    # ------------------------------------------------------------------ core
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        w = self.cfg["world"]
        self.pos = np.array([w["width"] / 2, w["start_y"]], dtype=np.float64)
        self.heading = np.pi / 2  # facing +y, toward the goal
        self.speed = 0.0
        self.steps = 0
        self.last_action = None
        self.last_reward = 0.0
        self.episode_return = 0.0
        if options and "barriers" in options:
            self.barriers = self._barriers_from_options(options["barriers"])
        else:
            self.barriers = self._sample_barriers()
        obs = self._observe()
        if self.render_mode == "human":
            self.render()
        return obs, self._info(collision=False, success=False)

    def step(self, action):
        a = self.cfg["agent"]
        r = self.cfg["reward"]
        dt = a["dt"]
        action = int(action)
        prev_dist = self._goal_dist()
        near_dist = self._nearest_surface_dist()

        if action == FORWARD:
            self.speed = min(a["max_speed"], self.speed + a["accel"] * dt)
        elif action == LEFT:
            self.heading += np.deg2rad(a["turn_rate"]) * dt
        elif action == RIGHT:
            self.heading -= np.deg2rad(a["turn_rate"]) * dt
        elif action == SLOW:
            self.speed = max(0.0, self.speed - a["decel"] * dt)
        elif action == STOP:
            self.speed = 0.0
        self.heading = wrap_angle(self.heading)
        self.pos += self.speed * dt * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.steps += 1
        self.last_action = action

        collision = self._collided()
        success = (not collision) and self._goal_dist() <= self.cfg["world"]["goal_radius"]

        reward = r["progress"] * (prev_dist - self._goal_dist()) + r["step"]
        if action == STOP and near_dist > r["stop_safe_dist"]:
            reward += r["unnecessary_stop"]
        if collision:
            reward += r["collision"]
        elif success:
            reward += r["goal"]

        terminated = collision or success
        truncated = (not terminated) and self.steps >= a["max_steps"]
        self.last_reward = float(reward)
        self.episode_return += reward
        obs = self._observe()
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, self._info(collision, success)

    # --------------------------------------------------------------- helpers
    def _observe(self):
        clean = self.camera.project(self.pose, self.barriers)
        self.detections = self.noise(clean, self.np_random)
        return build_observation(self.detections, self.speed / self.cfg["agent"]["max_speed"],
                                 self.goal_bearing, self.num_classes, self.max_dets)

    @property
    def pose(self):
        return (self.pos[0], self.pos[1], self.heading)

    @property
    def goal_bearing(self):
        d = self.goal - self.pos
        return float(wrap_angle(np.arctan2(d[1], d[0]) - self.heading))

    def _goal_dist(self):
        return float(np.linalg.norm(self.goal - self.pos))

    def _nearest_surface_dist(self):
        if len(self.barriers) == 0:
            return np.inf
        d = np.linalg.norm(self.barriers[:, :2] - self.pos, axis=1) - self.barriers[:, 2]
        return float(d.min() - self.cfg["agent"]["radius"])

    def _collided(self):
        ar = self.cfg["agent"]["radius"]
        w = self.cfg["world"]
        x, y = self.pos
        if x < ar or x > w["width"] - ar or y < 0 or y > w["length"]:
            return True
        return self._nearest_surface_dist() <= 0

    def _info(self, collision, success):
        return {"is_success": bool(success), "collision": bool(collision),
                "steps": self.steps, "goal_dist": self._goal_dist(),
                "num_detections": int(len(self.detections))}

    def _sample_barriers(self):
        w = self.cfg["world"]
        rng = self.np_random
        n = int(rng.integers(w["num_barriers"][0], w["num_barriers"][1] + 1))
        out = []
        for _ in range(n * 20):  # rejection sampling
            if len(out) == n:
                break
            c = int(rng.integers(self.num_classes))
            rad = float(rng.uniform(*self.cfg["classes"][c]["radius"]))
            x = rng.uniform(rad, w["width"] - rad)
            y = rng.uniform(*w["barrier_zone"])
            if all(np.hypot(x - b[0], y - b[1]) > rad + b[2] + w["min_gap"] for b in out):
                out.append([x, y, rad, c])
        return np.array(out, dtype=np.float64).reshape(-1, 4)

    def _barriers_from_options(self, items):
        """items: list of (x, y, class_id) or (x, y, radius, class_id)."""
        out = []
        for it in items:
            if len(it) == 3:
                x, y, c = it
                rad = float(np.mean(self.cfg["classes"][int(c)]["radius"]))
            else:
                x, y, rad, c = it
            out.append([x, y, rad, int(c)])
        return np.array(out, dtype=np.float64).reshape(-1, 4)

    # -------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .render import Renderer
            self._renderer = Renderer(self, human=self.render_mode == "human")
        return self._renderer.draw()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
