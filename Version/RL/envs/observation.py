"""Observation format shared by the simulator and the real-YOLO adapter.

A detection is a row [class_id, cx, cy, w, h, conf] with box values
normalized to [0, 1] (YOLO xywhn). The observation vector is:

    for each of the top-N closest detections:
        one_hot(class_id) (num_classes) | cx cy w h conf (5) | mask (1)
    then agent state:
        speed / max_speed | sin(goal_bearing) | cos(goal_bearing)

Padding slots are all zeros (mask = 0). goal_bearing is the angle from the
agent's heading to the goal, positive = goal is to the left.
"""
import numpy as np

BOX_FIELDS = 5  # cx, cy, w, h, conf
STATE_SIZE = 3


def det_feature_size(num_classes: int) -> int:
    return num_classes + BOX_FIELDS + 1


def obs_size(num_classes: int, max_dets: int) -> int:
    return max_dets * det_feature_size(num_classes) + STATE_SIZE


def obs_bounds(num_classes: int, max_dets: int):
    n = obs_size(num_classes, max_dets)
    low = np.zeros(n, dtype=np.float32)
    high = np.ones(n, dtype=np.float32)
    low[-2:] = -1.0  # sin / cos of goal bearing
    return low, high


def sort_by_closeness(dets: np.ndarray) -> np.ndarray:
    """Closest first. In a ground-level camera a closer object has its bottom
    edge lower in the image, so sort by bottom edge, then by box area."""
    if len(dets) == 0:
        return dets
    bottom = dets[:, 2] + dets[:, 4] / 2
    area = dets[:, 3] * dets[:, 4]
    order = np.lexsort((-area, -bottom))
    return dets[order]


def build_observation(dets, speed_norm: float, goal_bearing: float,
                      num_classes: int, max_dets: int) -> np.ndarray:
    dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
    dets = sort_by_closeness(dets)[:max_dets]
    f = det_feature_size(num_classes)
    obs = np.zeros(obs_size(num_classes, max_dets), dtype=np.float32)
    for i, (cls, cx, cy, w, h, conf) in enumerate(dets):
        base = i * f
        c = int(cls)
        if 0 <= c < num_classes:
            obs[base + c] = 1.0
        obs[base + num_classes: base + num_classes + BOX_FIELDS] = np.clip(
            [cx, cy, w, h, conf], 0.0, 1.0)
        obs[base + f - 1] = 1.0
    obs[-3] = np.clip(speed_norm, 0.0, 1.0)
    obs[-2] = np.sin(goal_bearing)
    obs[-1] = np.cos(goal_bearing)
    return obs


def decode_observation(obs: np.ndarray, num_classes: int, max_dets: int):
    """Inverse of build_observation (for rule-based agents and debugging).
    Returns (dets [M, 6], speed_norm, goal_bearing)."""
    f = det_feature_size(num_classes)
    dets = []
    for i in range(max_dets):
        slot = obs[i * f:(i + 1) * f]
        if slot[-1] < 0.5:
            continue
        cls = int(np.argmax(slot[:num_classes]))
        dets.append([cls, *slot[num_classes:num_classes + BOX_FIELDS]])
    dets = np.array(dets, dtype=np.float32).reshape(-1, 6)
    return dets, float(obs[-3]), float(np.arctan2(obs[-2], obs[-1]))
