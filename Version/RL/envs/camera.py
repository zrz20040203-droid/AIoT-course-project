"""Simulated forward camera that turns barriers into YOLO-style detections.

Pinhole model with the camera at a fixed height looking straight ahead.
For a barrier at forward distance d, lateral offset `lat` (right = +),
radius r and height H:
    x1, x2   = 0.5 + fx * (lat -/+ r) / d
    y_bottom = 0.5 + fy * cam_h / d          (ground contact point)
    y_top    = 0.5 - fy * (H - cam_h) / d
so closer barriers produce bigger boxes whose bottom edge is lower.
"""
import numpy as np


class SimCamera:
    def __init__(self, cam_cfg: dict, class_heights):
        self.hfov = np.deg2rad(cam_cfg["hfov"])
        self.vfov = np.deg2rad(cam_cfg["vfov"])
        self.cam_h = cam_cfg["height"]
        self.max_range = cam_cfg["max_range"]
        self.occlusion = cam_cfg["occlusion"]
        self.fx = 0.5 / np.tan(self.hfov / 2)
        self.fy = 0.5 / np.tan(self.vfov / 2)
        self.class_heights = np.asarray(class_heights, dtype=np.float32)

    def project(self, pose, barriers: np.ndarray) -> np.ndarray:
        """pose = (x, y, heading). barriers = [K, 4] rows (x, y, radius, class_id).
        Returns clean detections [M, 6] (class_id, cx, cy, w, h, conf=1),
        nearest first."""
        x, y, th = pose
        if len(barriers) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        dx = barriers[:, 0] - x
        dy = barriers[:, 1] - y
        fwd = dx * np.cos(th) + dy * np.sin(th)
        lat = dx * np.sin(th) - dy * np.cos(th)  # right of heading is positive
        r = barriers[:, 2]

        boxes = []
        for i in np.argsort(fwd):
            d = fwd[i]
            if d <= 0.05 or np.hypot(dx[i], dy[i]) - r[i] > self.max_range:
                continue
            d = max(d, 0.1)
            x1 = 0.5 + self.fx * (lat[i] - r[i]) / d
            x2 = 0.5 + self.fx * (lat[i] + r[i]) / d
            h_obj = self.class_heights[int(barriers[i, 3])]
            y2 = 0.5 + self.fy * self.cam_h / d
            y1 = 0.5 - self.fy * (h_obj - self.cam_h) / d
            x1, x2 = np.clip([x1, x2], 0.0, 1.0)
            y1, y2 = np.clip([y1, y2], 0.0, 1.0)
            if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
                continue  # outside the field of view
            box = (x1, y1, x2, y2)
            if any(_covered_fraction(box, b) > self.occlusion for b, _ in boxes):
                continue
            boxes.append((box, int(barriers[i, 3])))

        out = [[c, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1], 1.0]
               for b, c in boxes]
        return np.array(out, dtype=np.float32).reshape(-1, 6)


def _covered_fraction(box, occluder) -> float:
    ix = max(0.0, min(box[2], occluder[2]) - max(box[0], occluder[0]))
    iy = max(0.0, min(box[3], occluder[3]) - max(box[1], occluder[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / area if area > 0 else 1.0


class DetectionNoise:
    """Makes clean detections look like real YOLO output: jitter, misses
    (more likely for small boxes), false positives, class confusion,
    confidence noise and occasional whole-frame drops."""

    def __init__(self, noise_cfg: dict, num_classes: int):
        self.cfg = noise_cfg
        self.num_classes = num_classes

    def __call__(self, dets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        lvl = c["level"]
        if lvl <= 0:
            return dets.copy()
        if rng.random() < c["frame_drop_prob"] * lvl:
            return np.zeros((0, 6), dtype=np.float32)

        out = []
        for cls, cx, cy, w, h, _ in dets:
            area = w * h
            small = max(0.0, 1.0 - area / c["small_area"])
            if rng.random() < min(1.0, (c["miss_prob"] + c["miss_prob_small"] * small) * lvl):
                continue
            cx += rng.normal(0, c["center_jitter"] * lvl)
            cy += rng.normal(0, c["center_jitter"] * lvl)
            w *= max(0.1, 1 + rng.normal(0, c["size_jitter"] * lvl))
            h *= max(0.1, 1 + rng.normal(0, c["size_jitter"] * lvl))
            if rng.random() < c["class_confusion"] * lvl:
                cls = rng.integers(self.num_classes)
            conf = rng.normal(c["conf_mean"], c["conf_std"] * lvl) - 0.3 * small
            out.append([cls, cx, cy, w, h, conf])

        for _ in range(rng.poisson(c["false_pos_rate"] * lvl)):
            w, h = rng.uniform(0.02, 0.15), rng.uniform(0.03, 0.25)
            out.append([rng.integers(self.num_classes), rng.uniform(0, 1),
                        rng.uniform(0.4, 1.0), w, h,
                        rng.normal(c["fp_conf_mean"], 0.1)])

        out = np.array(out, dtype=np.float32).reshape(-1, 6)
        out[:, 1:6] = np.clip(out[:, 1:6], 0.0, 1.0)
        return out[out[:, 5] >= c["conf_threshold"]]
