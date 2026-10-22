"""Watch or record episodes of BarrierNav-v0.

  python -m scripts.render_episode --policy keyboard          # drive it yourself
  python -m scripts.render_episode --policy random --out runs/random.mp4
  python -m scripts.render_episode --policy forward --montage runs/forward.png

Keyboard: up = forward, left/right = turn, down = slow, space = stop, q = quit.
"""
import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import envs  # noqa: F401  (registers BarrierNav-v0)
from envs.barrier_nav_env import FORWARD, LEFT, RIGHT, SLOW, STOP


def keyboard_action():
    import pygame
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                keymap = {pygame.K_UP: FORWARD, pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
                          pygame.K_DOWN: SLOW, pygame.K_SPACE: STOP}
                if event.key == pygame.K_q:
                    return None
                if event.key in keymap:
                    return keymap[event.key]


def make_policy(name, env):
    if name == "random":
        return lambda obs: env.action_space.sample()
    if name == "forward":
        return lambda obs: FORWARD
    if name == "keyboard":
        return lambda obs: keyboard_action()
    raise ValueError(f"unknown policy {name}")


def save_montage(frames, path, n=6):
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    row = np.concatenate([frames[i] for i in idx[:3]], axis=1)
    rows = [row]
    if len(idx) > 3:
        rest = [frames[i] for i in idx[3:]]
        rest += [np.full_like(frames[0], 255)] * (3 - len(rest))
        rows.append(np.concatenate(rest, axis=1))
    img = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="random", choices=["random", "forward", "keyboard"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--noise", type=float, default=None, help="override noise level")
    p.add_argument("--out", type=Path, help="save an .mp4 of the episode(s)")
    p.add_argument("--montage", type=Path, help="save a PNG grid of 6 frames")
    args = p.parse_args()

    record = args.out is not None or args.montage is not None
    mode = "rgb_array" if record and args.policy != "keyboard" else "human"
    env = gym.make("BarrierNav-v0", render_mode=mode, noise_level=args.noise)
    env.action_space.seed(args.seed)
    policy = make_policy(args.policy, env)

    frames = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if mode == "rgb_array":
            frames.append(env.render())
        done = False
        while not done:
            action = policy(obs)
            if action is None:
                env.close()
                return
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if mode == "rgb_array":
                frames.append(env.render())
        outcome = "success" if info["is_success"] else "collision" if info["collision"] else "timeout"
        print(f"episode {ep}: {outcome}, steps={info['steps']}, "
              f"return={env.unwrapped.episode_return:.1f}")
    env.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"saved {args.out}")
    if args.montage:
        args.montage.parent.mkdir(parents=True, exist_ok=True)
        save_montage(frames, args.montage)
        print(f"saved {args.montage}")


if __name__ == "__main__":
    main()
