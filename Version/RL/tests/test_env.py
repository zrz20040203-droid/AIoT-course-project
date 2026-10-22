import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401
from envs.barrier_nav_env import FORWARD, STOP, BarrierNavEnv
from envs.camera import SimCamera
from envs.config import load_env_config
from envs.observation import build_observation, decode_observation, obs_size

CFG = load_env_config()
NC = len(CFG["classes"])
N = CFG["observation"]["max_detections"]


def clean_env(**kw):
    return BarrierNavEnv(noise_level=0.0, **kw)


def test_check_env():
    check_env(BarrierNavEnv())


def test_reset_step_shapes():
    env = gym.make("BarrierNav-v0")
    obs, info = env.reset(seed=1)
    assert obs.shape == (obs_size(NC, N),) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(r, float)
        if term or trunc:
            break


def test_seed_determinism():
    a, b = BarrierNavEnv(), BarrierNavEnv()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    np.testing.assert_array_equal(oa, ob)
    np.testing.assert_array_equal(a.barriers, b.barriers)
    for _ in range(10):
        oa = a.step(FORWARD)[0]
        ob = b.step(FORWARD)[0]
    np.testing.assert_array_equal(oa, ob)


def test_forward_progress_reward_positive():
    env = clean_env()
    env.reset(seed=0, options={"barriers": []})
    rewards = [env.step(FORWARD)[1] for _ in range(10)]
    assert sum(rewards) > 0


def test_driving_into_barrier_collides():
    env = clean_env()
    env.reset(seed=0, options={"barriers": [(4.0, 3.0, 1)]})  # wall straight ahead
    for _ in range(100):
        _, r, term, trunc, info = env.step(FORWARD)
        if term:
            break
    assert info["collision"] and not info["is_success"]
    assert r <= CFG["reward"]["collision"] + 1.0


def test_reaching_goal_succeeds():
    env = clean_env()
    env.reset(seed=0, options={"barriers": []})
    for _ in range(CFG["agent"]["max_steps"]):
        _, r, term, trunc, info = env.step(FORWARD)
        if term or trunc:
            break
    assert info["is_success"] and not info["collision"]
    assert r >= CFG["reward"]["goal"]


def test_unnecessary_stop_penalized():
    env = clean_env()
    env.reset(seed=0, options={"barriers": []})
    _, r, *_ = env.step(STOP)
    assert r == pytest.approx(CFG["reward"]["step"] + CFG["reward"]["unnecessary_stop"])
    # stopping right next to a barrier is not penalized
    env.reset(seed=0, options={"barriers": [(4.0, 2.0, 0)]})
    _, r, *_ = env.step(STOP)
    assert r == pytest.approx(CFG["reward"]["step"])


def test_camera_geometry():
    cam = SimCamera(CFG["camera"], [c["height"] for c in CFG["classes"]])
    pose = (0.0, 0.0, np.pi / 2)  # facing +y
    near = cam.project(pose, np.array([[0.0, 2.0, 0.2, 0]]))
    far = cam.project(pose, np.array([[0.0, 6.0, 0.2, 0]]))
    assert near[0, 1] == pytest.approx(0.5, abs=1e-6)              # dead ahead -> centered
    assert near[0, 3] > far[0, 3] and near[0, 4] > far[0, 4]        # closer -> bigger
    assert near[0, 2] + near[0, 4] / 2 > far[0, 2] + far[0, 4] / 2  # closer -> lower bottom edge
    right = cam.project(pose, np.array([[1.0, 4.0, 0.2, 0]]))
    assert right[0, 1] > 0.5                                         # to the right -> cx > 0.5
    assert len(cam.project(pose, np.array([[0.0, -3.0, 0.2, 0]]))) == 0   # behind
    assert len(cam.project(pose, np.array([[0.0, 20.0, 0.2, 0]]))) == 0   # out of range
    assert len(cam.project(pose, np.array([[5.0, 1.0, 0.2, 0]]))) == 0    # outside FOV


def test_occlusion():
    cam = SimCamera(CFG["camera"], [c["height"] for c in CFG["classes"]])
    pose = (0.0, 0.0, np.pi / 2)
    dets = cam.project(pose, np.array([[0.0, 2.0, 0.8, 1], [0.0, 5.0, 0.15, 0]]))
    assert len(dets) == 1 and int(dets[0, 0]) == 1


def test_noise_zero_is_clean_and_noise_changes_dets():
    env0 = clean_env()
    env0.reset(seed=3)
    clean = env0.camera.project(env0.pose, env0.barriers)
    np.testing.assert_array_equal(env0.detections, clean)

    env = BarrierNavEnv(noise_level=2.0)
    diffs = 0
    for s in range(20):
        env.reset(seed=s)
        clean = env.camera.project(env.pose, env.barriers)
        diffs += len(clean) != len(env.detections) or not np.allclose(clean, env.detections)
    assert diffs > 10


def test_observation_roundtrip_and_padding():
    dets = np.array([[2, 0.3, 0.6, 0.1, 0.2, 0.9],
                     [0, 0.5, 0.8, 0.3, 0.4, 0.7]], dtype=np.float32)
    obs = build_observation(dets, 0.5, 0.3, NC, N)
    out, speed, bearing = decode_observation(obs, NC, N)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], dets[1], atol=1e-6)  # lower/bigger box sorted first
    assert speed == pytest.approx(0.5) and bearing == pytest.approx(0.3, abs=1e-6)
    empty = build_observation(np.zeros((0, 6)), 0.0, 0.0, NC, N)
    assert empty[:-3].sum() == 0
    many = np.tile(dets, (10, 1))
    assert len(decode_observation(build_observation(many, 0, 0, NC, N), NC, N)[0]) == N
