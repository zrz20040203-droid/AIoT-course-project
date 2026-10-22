"""Pygame renderer: top-down map (left) + simulated camera view with the
noisy detections the agent actually sees (right)."""
import os

import numpy as np

CLASS_COLORS = [(255, 140, 0), (120, 120, 140), (60, 160, 255), (220, 60, 120),
                (80, 200, 120), (200, 200, 60)]
PPM = 20          # pixels per meter on the map
MARGIN = 10
CAM_W, CAM_H = 384, 288


class Renderer:
    def __init__(self, env, human: bool):
        if not human:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        self.env = env
        self.human = human
        w = env.cfg["world"]
        self.map_w = int(w["width"] * PPM)
        self.map_h = int(w["length"] * PPM)
        self.size = (self.map_w + CAM_W + 3 * MARGIN, max(self.map_h, CAM_H + 200) + 2 * MARGIN)
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)
        self.big_font = pygame.font.Font(None, 22)
        if human:
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("BarrierNav-v0")
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface(self.size)

    def _to_px(self, x, y):
        # world y grows toward the goal (up on screen)
        return int(MARGIN + x * PPM), int(MARGIN + self.map_h - y * PPM)

    def draw(self):
        pg, env, s = self.pg, self.env, self.screen
        if self.human:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    env.close()
                    return None
        s.fill((245, 245, 240))
        self._draw_map()
        self._draw_camera()
        self._draw_text()
        if self.human:
            pg.display.flip()
            self.clock.tick(env.metadata["render_fps"])
            return None
        return np.transpose(pg.surfarray.array3d(s), (1, 0, 2)).copy()

    def _draw_map(self):
        pg, env, s = self.pg, self.env, self.screen
        pg.draw.rect(s, (255, 255, 255), (MARGIN, MARGIN, self.map_w, self.map_h))
        pg.draw.rect(s, (40, 40, 40), (MARGIN, MARGIN, self.map_w, self.map_h), 2)
        gx, gy = self._to_px(*env.goal)
        pg.draw.circle(s, (120, 220, 120), (gx, gy), int(env.cfg["world"]["goal_radius"] * PPM))

        # field of view wedge
        x, y, th = env.pose
        cam = env.camera
        rng = cam.max_range
        p0 = self._to_px(x, y)
        pts = [p0]
        for a in np.linspace(th + cam.hfov / 2, th - cam.hfov / 2, 12):
            pts.append(self._to_px(x + rng * np.cos(a), y + rng * np.sin(a)))
        fov = pg.Surface(self.size, pg.SRCALPHA)
        pg.draw.polygon(fov, (255, 220, 80, 60), pts)
        s.blit(fov, (0, 0))

        for bx, by, br, c in env.barriers:
            pg.draw.circle(s, CLASS_COLORS[int(c) % len(CLASS_COLORS)],
                           self._to_px(bx, by), max(2, int(br * PPM)))

        ar = env.cfg["agent"]["radius"]
        tip = self._to_px(x + 2.2 * ar * np.cos(th), y + 2.2 * ar * np.sin(th))
        left = self._to_px(x + ar * np.cos(th + 2.4), y + ar * np.sin(th + 2.4))
        right = self._to_px(x + ar * np.cos(th - 2.4), y + ar * np.sin(th - 2.4))
        pg.draw.polygon(s, (20, 20, 20), [tip, left, right])

    def _draw_camera(self):
        pg, env, s = self.pg, self.env, self.screen
        ox, oy = self.map_w + 2 * MARGIN, MARGIN
        pg.draw.rect(s, (200, 225, 245), (ox, oy, CAM_W, CAM_H // 2))       # sky
        pg.draw.rect(s, (205, 200, 185), (ox, oy + CAM_H // 2, CAM_W, CAM_H // 2))  # ground
        for cls, cx, cy, w, h, conf in env.detections:
            col = CLASS_COLORS[int(cls) % len(CLASS_COLORS)]
            rect = pg.Rect(int(ox + (cx - w / 2) * CAM_W), int(oy + (cy - h / 2) * CAM_H),
                           max(1, int(w * CAM_W)), max(1, int(h * CAM_H)))
            pg.draw.rect(s, col, rect, 2)
            label = f"{env.class_names[int(cls)]} {conf:.2f}"
            s.blit(self.font.render(label, True, col), (rect.x, max(oy, rect.y - 14)))
        pg.draw.rect(s, (40, 40, 40), (ox, oy, CAM_W, CAM_H), 2)
        s.blit(self.font.render("simulated camera (noisy YOLO detections)", True, (40, 40, 40)),
               (ox, oy + CAM_H + 4))

    def _draw_text(self):
        from .barrier_nav_env import ACTION_NAMES
        env, s = self.env, self.screen
        ox, oy = self.map_w + 2 * MARGIN, MARGIN + CAM_H + 30
        act = "-" if env.last_action is None else ACTION_NAMES[env.last_action]
        lines = [
            f"step {env.steps}   action: {act}",
            f"speed {env.speed:.2f} m/s   goal dist {env._goal_dist():.1f} m",
            f"goal bearing {np.rad2deg(env.goal_bearing):+.0f} deg",
            f"reward {env.last_reward:+.2f}   return {env.episode_return:+.1f}",
            f"detections {len(env.detections)}   noise level {env.cfg['noise']['level']:.1f}",
        ]
        for i, line in enumerate(lines):
            s.blit(self.big_font.render(line, True, (20, 20, 20)), (ox, oy + i * 22))
        for i, name in enumerate(env.class_names):
            col = CLASS_COLORS[i % len(CLASS_COLORS)]
            y = oy + len(lines) * 22 + 10 + i * 18
            self.pg.draw.circle(s, col, (ox + 6, y + 6), 6)
            s.blit(self.font.render(name, True, (20, 20, 20)), (ox + 18, y))

    def close(self):
        if self.human:
            self.pg.display.quit()
