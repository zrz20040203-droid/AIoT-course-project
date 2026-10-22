# AIoT course project: RL barrier navigation

An RL agent that uses YOLO barrier detections as its observation and picks
movement actions (forward / left / right / slow / stop) to reach a goal
without hitting anything. It is trained in a 2D simulator, then run on real YOLO output.

```
configs/     env.yaml: world, camera, classes, noise, reward
envs/        BarrierNav-v0 Gymnasium env, simulated camera + YOLO noise, renderer
scripts/     check_env.py, render_episode.py
tests/       pytest tests
(run all commands from Version/RL)
```

## Setup
```bash
conda activate AIoT
pip install -r Version/requirements.txt   # run from the repo root
```

## Phase 1: simulator
```bash
python -m pytest                                    # tests
python -m scripts.check_env                         # gymnasium env checker + random rollouts
python -m scripts.render_episode --policy keyboard  # drive it yourself (arrows, space = stop, q = quit)
python -m scripts.render_episode --policy random --out runs/random.mp4 --montage runs/random.png
python -m scripts.render_episode --noise 2.0 ...    # harsher detector noise
```

**Observation** (83 floats with the default config): the 8 closest detections, sorted by
how low their bottom edge is in the image. Each detection is
`one-hot class (4) | cx cy w h conf | mask`, followed by
`speed/max_speed, sin(goal bearing), cos(goal bearing)`. The same builder
(`envs/observation.py`) will be used by the real-YOLO adapter.

**Noise** (`configs/env.yaml -> noise`): box jitter, misses (more likely for tiny boxes),
Poisson false positives, class confusion, confidence noise, whole-frame drops, and
YOLO's confidence threshold. `level` scales all of them together.

**Reward**: +1 per meter of progress, −50 collision (barriers or leaving the area),
+50 goal, −0.01 per step, −0.1 for STOP when nothing is within 1.5 m.
