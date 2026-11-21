"""Obstacle-free Stage B1 target trajectories for ideal quadrotors."""

import math

from quad_tracking_env import IdealQuadTracking
from tracking_env import CONTROL_HZ, DT

B1_ROUTES = (
    "single-100", "single-120", "single-150",
    "alternate-100", "alternate-120", "alternate-150",
    "circle", "eight",
)
CLOSED_ROUTES = ("circle", "eight")
B1_LOOP_RADIUS_M = 3.0
LOOP_STEPS = round(math.tau * B1_LOOP_RADIUS_M / (0.8 * DT))
LOOP_SPEED_MPS = math.tau * B1_LOOP_RADIUS_M / (LOOP_STEPS * DT)
LEAD_IN_STEPS = 30
FINISH_STEPS = CONTROL_HZ
REVERSE_BLEND_END = -0.20


class B1Tracking(IdealQuadTracking):
    def __init__(self, template, directory, stage, *, routes, mirror_episodes=False, reverse_speed_mps=0.0, **kwargs):
        if stage != "B1" or not routes or any(route not in B1_ROUTES for route in routes):
            raise ValueError("Stage B1 requires valid target routes")
        self.routes = routes
        self.reverse_speed_mps = reverse_speed_mps
        self.mirror_episodes = mirror_episodes
        self.first_side = kwargs.get("fixed_route_side") or 1
        self.episode_index = 0
        kwargs["b_weave"] = True
        super().__init__(template, directory, "B", **kwargs)

    def reset(self, *, seed=None, options=None):
        pair, phase = divmod(self.episode_index, 2)
        self.route = self.routes[pair % len(self.routes)]
        if self.mirror_episodes:
            self.fixed_route_side = self.first_side if phase == 0 else -self.first_side
        observation, info = super().reset(seed=seed, options=options)
        self.episode_index += 1
        if self.route in CLOSED_ROUTES:
            self.loops = 1 if self.route == "circle" else 2
            self.duration = LEAD_IN_STEPS + self.loops * LOOP_STEPS + FINISH_STEPS
        else:
            profile, angle = self.route.split("-")
            self.turn_legs = 1 if profile == "single" else 4
            self.turn = math.copysign(math.radians(int(angle)) / (self.b_turn_leg * DT), self.turn)
        return observation, info

    def target_command(self):
        if self.route in CLOSED_ROUTES:
            loop_step = self.elapsed - LEAD_IN_STEPS
            if not 0 <= loop_step < self.loops * LOOP_STEPS:
                return LOOP_SPEED_MPS, 0.0
            turn = math.copysign(LOOP_SPEED_MPS / B1_LOOP_RADIUS_M, self.turn)
            # The second lobe has opposite curvature and the same tangent at the crossover.
            return LOOP_SPEED_MPS, turn if loop_step < LOOP_STEPS else -turn

        speed, turn = super().target_command()
        if self.elapsed >= self.b_turn_start + self.turn_legs * self.b_turn_leg:
            turn = 0.0
        return speed, turn

    def pursuer_command(self, action):
        if self.reverse_speed_mps == 0 or action[0] >= REVERSE_BLEND_END:
            return super().pursuer_command(action)
        blend = (float(action[0]) + 1.0) / (REVERSE_BLEND_END + 1.0)
        upper_speed = 0.75 * (REVERSE_BLEND_END + 1.0)
        forward = -self.reverse_speed_mps + blend * (upper_speed + self.reverse_speed_mps)
        lateral = float(action[1])
        yaw_rate = 1.5 * float(action[2])
        self.drive(self.pursuer, forward, yaw_rate, lateral=lateral)
        return {"v": forward, "lateral": lateral, "omega": yaw_rate}

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        elapsed = self.elapsed
        if self.route in CLOSED_ROUTES:
            route_end = LEAD_IN_STEPS + self.loops * LOOP_STEPS
        else:
            route_end = self.b_turn_start + self.turn_legs * self.b_turn_leg
        info.update(b1_route=self.route, b1_planned_steps=self.duration,
                    b1_route_complete=elapsed >= route_end and not terminated)
        return observation, reward, terminated, truncated, info
