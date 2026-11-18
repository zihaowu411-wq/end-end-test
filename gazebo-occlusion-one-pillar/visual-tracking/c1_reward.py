"""Reward for bypassing the C1 pillar and then recovering tracking."""

import math


C1_BYPASS_RECOVERY = "c1-bypass-recovery"
C1_LINEAR_RECOVERY = "c1-bypass-recovery-linear"
C1_RECOVERY_FEEDBACK = "c1-bypass-recovery-feedback"
C1_GAP_RECOVERY = "c1-bypass-recovery-gap"
C1_EARLY_BRAKE = "c1-bypass-recovery-brake"
C1_TARGET_SIDE = "c1-bypass-recovery-target-side"
_BYPASS_SETTINGS = {
    "clearance_threshold_m": 0.20,
    "max_clearance_penalty": 0.04,
    "collision": -12.0,
    "uncrossed_timeout": -2.0,
    "crossing_bonus": 2.0,
    "max_crossing_clearance_bonus": 1.0,
    "pre_cross_time_cost": 0.02,
    "post_cross_time_cost": 0.01,
    "action_change_weight": 0.005,
    "lateral_potential_weight": 2.0,
    "forward_potential_weight": 2.0,
    "forward_gate_start": 0.80,
    "pre_cross_gap_weight": 0.005,
    "gap_scale": 0.95,
}
C1_REWARD_PROFILES = {
    C1_BYPASS_RECOVERY: dict(
        _BYPASS_SETTINGS, post_cross_valid_weight=0.035, post_cross_gap_weight=0.015,
        post_cross_bearing_weight=0.020, bearing_scale_deg=35.0,
    ),
    C1_LINEAR_RECOVERY: dict(
        _BYPASS_SETTINGS, post_cross_valid_weight=0.05, post_cross_bearing_weight=0.06,
        bearing_scale_deg=180.0, post_cross_close_weight=0.06, post_cross_far_weight=0.02,
        min_gap_m=2.5, max_gap_m=3.5, far_gap_scale_m=2.0,
    ),
}
C1_REWARD_PROFILES[C1_RECOVERY_FEEDBACK] = dict(
    C1_REWARD_PROFILES[C1_LINEAR_RECOVERY],
    post_cross_closing_speed_weight=0.03,
    post_cross_closing_speed_scale_mps=1.0,
    post_cross_recenter_potential_weight=2.0,
    post_cross_recenter_deadband_deg=35.0,
)
C1_REWARD_PROFILES[C1_GAP_RECOVERY] = dict(
    C1_REWARD_PROFILES[C1_RECOVERY_FEEDBACK],
    crossing_gap_penalty_weight=0.5,
    crossing_gap_penalty_scale_m=2.0,
    post_cross_far_potential_weight=1.8,
    post_cross_far_potential_scale_m=3.0,
)
C1_REWARD_PROFILES[C1_EARLY_BRAKE] = dict(
    C1_REWARD_PROFILES[C1_GAP_RECOVERY],
    post_cross_closing_speed_weight=0.06,
    post_cross_brake_start_m=3.8,
    post_cross_brake_ramp_m=0.6,
    post_cross_brake_response_s=1.0,
)
C1_REWARD_PROFILES[C1_TARGET_SIDE] = dict(
    C1_REWARD_PROFILES[C1_EARLY_BRAKE],
    route_penalty_weight=0.02,
    route_start_before_pillar_m=3.0,
    route_ramp_m=1.5,
    route_target_deadband_m=0.5,
)


def clip01(value):
    return min(1.0, max(0.0, value))


class C1BypassRecovery:
    def __init__(self, start, pillar_center, pillar_half_size, body_half_size, goal, gamma,
                 profile=C1_BYPASS_RECOVERY):
        self.profile = profile
        self.settings = C1_REWARD_PROFILES[profile]
        self.goal = goal
        self.gamma = gamma
        self.pillar_x, pillar_y = pillar_center
        self.pillar_front_y = pillar_y - pillar_half_size[1]
        body_radius = math.hypot(*body_half_size)
        self.safe_width = pillar_half_size[0] + body_radius + self.settings["clearance_threshold_m"]
        self.start_y = start.y
        self.forward_span = pillar_y + pillar_half_size[1] + body_radius - self.start_y
        self.crossed = False
        self.min_clearance = math.inf
        self.previous_potential = self.potential(start.x, start.y)
        self.previous_recenter_potential = 0.0
        self.previous_far_gap_potential = 0.0
        self.previous_distance = None
        self.previous_time = None

    def potential(self, x, y):
        settings = self.settings
        lateral = clip01(abs(x - self.pillar_x) / self.safe_width)
        forward = clip01((y - self.start_y) / self.forward_span)
        gate_start = settings["forward_gate_start"]
        gate = clip01((lateral - gate_start) / (1.0 - gate_start))
        return (-settings["lateral_potential_weight"] * (1.0 - lateral)
                - settings["forward_potential_weight"] * (1.0 - gate * forward))

    def recovery_feedback(self, info, was_crossed, terminal):
        settings = self.settings
        current_potential = 0.0
        if self.crossed and not terminal:
            deadband = math.radians(settings["post_cross_recenter_deadband_deg"])
            excess = clip01((abs(info["bearing"]) - deadband) / (math.pi - deadband))
            current_potential = -settings["post_cross_recenter_potential_weight"] * excess
        recenter = self.gamma * current_potential - self.previous_recenter_potential
        self.previous_recenter_potential = current_potential

        distance, time = info["distance"], info["time"]
        closing_penalty = 0.0
        if was_crossed and (
            self.profile in (C1_EARLY_BRAKE, C1_TARGET_SIDE)
            or self.previous_distance < settings["min_gap_m"]
        ):
            closing_speed = (self.previous_distance - distance) / (time - self.previous_time)
            activation = 1.0
            if self.profile in (C1_EARLY_BRAKE, C1_TARGET_SIDE):
                activation = clip01(
                    (settings["post_cross_brake_start_m"] - self.previous_distance)
                    / settings["post_cross_brake_ramp_m"]
                )
                allowed_closing_speed = max(
                    0.0, (self.previous_distance - self.goal) / settings["post_cross_brake_response_s"]
                )
                closing_speed -= allowed_closing_speed
            closing_penalty = settings["post_cross_closing_speed_weight"] * activation * clip01(
                closing_speed / settings["post_cross_closing_speed_scale_mps"],
            )
        self.previous_distance, self.previous_time = distance, time
        info.update(reward_recenter_progress=recenter, reward_closing_penalty=closing_penalty)
        return recenter - closing_penalty

    def far_gap_feedback(self, distance, terminal):
        settings = self.settings
        potential = 0.0
        if self.crossed and not terminal:
            excess = clip01((distance - settings["max_gap_m"]) / settings["post_cross_far_potential_scale_m"])
            potential = -settings["post_cross_far_potential_weight"] * excess
        progress = self.gamma * potential - self.previous_far_gap_potential
        self.previous_far_gap_potential = potential
        return progress

    def route_penalty(self, info):
        settings = self.settings
        target_offset = info["target_x"] - self.pillar_x
        if abs(target_offset) < settings["route_target_deadband_m"]:
            return 0.0
        distance_to_front = self.pillar_front_y - info["y"]
        activation = clip01(
            (settings["route_start_before_pillar_m"] - distance_to_front) / settings["route_ramp_m"]
        )
        reference_x = self.pillar_x + math.copysign(self.safe_width, target_offset)
        lateral_error = abs(info["x"] - reference_x)
        return settings["route_penalty_weight"] * activation * clip01(lateral_error / (2 * self.safe_width))

    def step(self, info, clearance, action_change, *, failed, timed_out):
        settings = self.settings
        margin = settings["clearance_threshold_m"]
        was_crossed = self.crossed
        if not self.crossed:
            self.min_clearance = min(self.min_clearance, clearance)
        first_crossing = bool(info["past_pillar"] and not self.crossed and not failed)
        self.crossed = self.crossed or first_crossing

        current_potential = 0.0
        if not (self.crossed or failed or timed_out):
            current_potential = self.potential(info["x"], info["y"])
        progress = self.gamma * current_potential - self.previous_potential
        self.previous_potential = current_potential

        gap = math.exp(-0.5 * ((info["distance"] - self.goal) / settings["gap_scale"]) ** 2)
        visible_gap = gap * info["visible"]
        if self.crossed:
            angle = abs(info["bearing"]) / math.radians(settings["bearing_scale_deg"])
            if self.profile != C1_BYPASS_RECOVERY:
                distance = info["distance"]
                min_gap = settings["min_gap_m"]
                too_close = max(0.0, (min_gap - distance) / min_gap)
                too_far = clip01((distance - settings["max_gap_m"]) / settings["far_gap_scale_m"])
                tracking = (settings["post_cross_valid_weight"] * info["valid"]
                            - settings["post_cross_bearing_weight"] * angle
                            - settings["post_cross_close_weight"] * too_close
                            - settings["post_cross_far_weight"] * too_far)
            else:
                alignment = math.exp(-angle)
                tracking = (settings["post_cross_valid_weight"] * info["valid"]
                            + settings["post_cross_gap_weight"] * visible_gap
                            + settings["post_cross_bearing_weight"] * (alignment - 1.0))
            time_cost = settings["post_cross_time_cost"]
        else:
            tracking = settings["pre_cross_gap_weight"] * visible_gap
            time_cost = settings["pre_cross_time_cost"]

        event = 0.0
        crossing_gap_penalty = 0.0
        if failed:
            event = settings["collision"]
        elif first_crossing:
            event = (settings["crossing_bonus"]
                     + settings["max_crossing_clearance_bonus"] * clip01(self.min_clearance / margin))
            if self.profile in (C1_GAP_RECOVERY, C1_EARLY_BRAKE, C1_TARGET_SIDE):
                excess = (info["distance"] - settings["max_gap_m"]) / settings["crossing_gap_penalty_scale_m"]
                crossing_gap_penalty = settings["crossing_gap_penalty_weight"] * clip01(excess)
                event -= crossing_gap_penalty
        elif timed_out and not self.crossed:
            event = settings["uncrossed_timeout"]

        penalty = settings["max_clearance_penalty"] * clip01(1.0 - clearance / margin)
        action_cost = settings["action_change_weight"] * action_change
        info.update(
            pillar_clearance_m=clearance,
            clearance_penalty=penalty,
            reward_bypass_progress=progress,
            reward_tracking=tracking,
            reward_time_cost=time_cost,
            reward_action_change=action_cost,
            reward_event=event,
            crossed_pillar=self.crossed,
            time_limit=bool(timed_out and not failed),
        )
        reward = progress + tracking - penalty - time_cost - action_cost + event
        terminal = failed or timed_out
        if self.profile in (C1_RECOVERY_FEEDBACK, C1_GAP_RECOVERY, C1_EARLY_BRAKE, C1_TARGET_SIDE):
            reward += self.recovery_feedback(info, was_crossed, terminal)
        if self.profile in (C1_GAP_RECOVERY, C1_EARLY_BRAKE, C1_TARGET_SIDE):
            distance_progress = self.far_gap_feedback(info["distance"], terminal)
            info.update(reward_crossing_gap_penalty=crossing_gap_penalty,
                        reward_far_gap_progress=distance_progress)
            reward += distance_progress
        if self.profile == C1_TARGET_SIDE:
            route_penalty = 0.0 if self.crossed or failed else self.route_penalty(info)
            info["reward_route_penalty"] = route_penalty
            reward -= route_penalty
        return reward
