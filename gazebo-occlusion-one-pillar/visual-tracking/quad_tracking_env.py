from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from gymnasium import spaces
import numpy as np
from gz.msgs10.twist_pb2 import Twist

from tracking_env import (
    PILLAR_HALF_SIZE,
    GAMMA, GOAL,
    CONTACT, CONTROL_HZ, DT, GazeboTracking, HORIZONTAL_FOV_HALF_RAD, IDEAL_VELOCITY_STAGES,
    Pose, pillar_clearance, write_world,
)


ALTITUDE_M = 3.0
TARGET_CUBE_SIDE_M = 0.68
PILLAR_HEIGHT_M = 6.0
X3_DEMO = Path("/usr/share/gz/gz-sim8/worlds/multicopter_velocity_control.sdf")
X3_ASSETS = Path(__file__).resolve().parent / "assets" / "x3"


def add_box_visual(link, name, pose, size, color):
    visual = ET.SubElement(link, "visual", name=name)
    ET.SubElement(visual, "pose").text = pose
    box = ET.SubElement(ET.SubElement(visual, "geometry"), "box")
    ET.SubElement(box, "size").text = size
    set_visual_color(visual, color)


def set_visual_color(visual, color):
    material = visual.find("material")
    if material is not None:
        visual.remove(material)
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = color
    ET.SubElement(material, "diffuse").text = color


def scale_x3_visuals(model, chassis, scale):
    # The X3 asset's link frames are aligned with the chassis in the nominal pose.
    for link in model.findall("link"):
        origin = tuple(map(float, link.findtext("pose").split()[:3]))
        for visual in link.findall("visual"):
            pose = visual.find("pose")
            values = pose.text.split()
            position = [f"{scale * (offset + float(value)):.9g}"
                        for offset, value in zip(origin, values[:3])]
            pose.text = " ".join(position + values[3:])
            dimensions = visual.find("geometry/mesh/scale")
            if dimensions is None:
                dimensions = visual.find("geometry/box/size")
            dimensions.text = " ".join(f"{scale * float(value):.9g}" for value in dimensions.text.split())
            if link is not chassis:
                link.remove(visual)
                chassis.append(visual)


def x3_model(name, pose, color, source, demo, *, cube_visual=False, visual_scale=1.0):
    model = deepcopy(source)
    model.set("name", name)
    model.find("pose").text = pose
    for item in model.iter():
        if item.get("name", "").startswith("X3/"):
            item.set("name", item.get("name").removeprefix("X3/"))
        if item.text and item.text.startswith("X3/"):
            item.text = item.text.removeprefix("X3/")
        if item.tag == "uri" and item.text.startswith("model://x3/"):
            item.text = str(X3_ASSETS / item.text.removeprefix("model://x3/"))
    base = model.find("link[@name='base_link']")
    base.set("name", "chassis")
    base.find("collision").set("name", "body_collision")
    for joint in model.findall("joint"):
        if joint.find("parent").text == "base_link":
            joint.find("parent").text = "chassis"
    if cube_visual:
        for link in model.findall("link"):
            for visual in link.findall("visual"):
                link.remove(visual)
        side = str(TARGET_CUBE_SIDE_M)
        add_box_visual(base, "tracking_cube", "0 0 0 0 0 0", " ".join([side] * 3), color)
    else:
        if name == "target":
            for visual in model.findall("link/visual"):
                set_visual_color(visual, color)
        add_box_visual(base, "tracking_color", "0 0 0.065 0 0 0", "0.32 0.24 0.055", color)
        if visual_scale != 1.0:
            scale_x3_visuals(model, base, visual_scale)
    for plugin in demo.findall("plugin"):
        if "multicopter" not in plugin.get("filename", ""):
            continue
        plugin = deepcopy(plugin)
        for item in plugin.iter():
            if item.text and item.text.startswith("X3/"):
                item.text = item.text.removeprefix("X3/")
        plugin.find("robotNamespace").text = f"/model/{name}"
        if plugin.find("commandSubTopic").text.endswith("/twist"):
            plugin.find("commandSubTopic").text = "cmd_vel"
            plugin.find("comLinkName").text = "chassis"
            plugin.find("velocityGain").text = "5.4 5.4 2.7"
            plugin.find("attitudeGain").text = "5 7.5 0.15"
            plugin.find("angularRateGain").text = "0.70 0.90 0.18"
            acceleration_limit = plugin.find("maximumLinearAcceleration")
            if acceleration_limit is None:
                acceleration_limit = ET.SubElement(plugin, "maximumLinearAcceleration")
            acceleration_limit.text = "2.2 2.2 2"
        model.append(plugin)
    return model


class QuadTracking(GazeboTracking):
    """X3 quadrotors with native thrust dynamics and a shared altitude setpoint."""

    spawn_z = ALTITUDE_M
    camera_x = 0.27
    collision_half_size = (0.25, 0.32)
    neutral_action = (-1.0, 0.0, 0.0)
    reset_attempts = 12
    reset_linear_speed_mps = 0.05
    reset_angular_speed_radps = 0.10
    reset_stable_steps = 3
    reset_settle_steps = 3 * CONTROL_HZ

    def __init__(self, *args, cube_target=False, target_visual_scale=1.0, **kwargs):
        self.cube_target = cube_target
        self.target_visual_scale = target_visual_scale
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.stage == "B":
            rng = self.np_random
            self.target_turn_rates = np.empty(self.duration)
            direction = math.copysign(1.0, self.turn)
            start = 0
            while start < self.duration:
                turn_steps = int(rng.integers(3 * CONTROL_HZ, 4 * CONTROL_HZ + 1))
                turn_angle = math.radians(rng.uniform(20.0, 100.0))
                end = start + turn_steps
                self.target_turn_rates[start:end] = direction * turn_angle / (turn_steps * DT)
                start = end
                # Reverse at the boundary so consecutive arcs cannot merge into a turn over 100 degrees.
                direction = -direction
        return observation, info

    def target_command(self):
        speed, yaw_rate = super().target_command()
        if self.stage == "B":
            yaw_rate = float(self.target_turn_rates[self.elapsed])
        return speed, yaw_rate

    def build_world(self, template, output, rtf):
        tree = ET.parse(template)
        world = tree.getroot().find("world")
        pillar = world.find("model[@name='pillar']")
        pillar.find("pose").text = "0 0 3 0 0 0"
        for size in pillar.findall(".//box/size"):
            size.text = f"1.35 1.45 {PILLAR_HEIGHT_M:g}"
        source = ET.parse(X3_ASSETS / "model.sdf").getroot().find("model")
        demo = ET.parse(X3_DEMO).getroot().find("world/include")
        for name, color in (("pursuer", "0.03 0.24 0.95 1"), ("target", "0.95 0.04 0.02 1")):
            old = world.find(f"model[@name='{name}']")
            values = old.find("pose").text.split()
            values[2] = str(self.spawn_z)
            model = x3_model(name, " ".join(values), color, source, demo,
                             cube_visual=name == "target" and self.cube_target,
                             visual_scale=self.target_visual_scale if name == "target" else 1.0)
            if name == "pursuer":
                camera = deepcopy(old.find("link/sensor[@name='front_rgb']"))
                camera.find("pose").text = f"{self.camera_x} 0 0 0 0 0"
                model.find("link[@name='chassis']").append(camera)
                for link in model.findall("link"):
                    if not link.get("name").startswith("rotor_"):
                        continue
                    sensor = ET.SubElement(link, "sensor", name="rotor_contact", type="contact")
                    contact = ET.SubElement(sensor, "contact")
                    ET.SubElement(contact, "collision").text = link.find("collision").get("name")
                    ET.SubElement(contact, "topic").text = CONTACT
            world.remove(old)
            world.append(model)
        source_world = output.with_name("quad-template.sdf")
        tree.write(source_world, encoding="utf-8", xml_declaration=True)
        write_world(source_world, output, self.stage, self.record, rtf, remove_pillar=self.remove_pillar,
                    pillar_center=self.pillar_center, camera_resolution=self.camera_resolution)

    def settle_after_teleport(self):
        # Take the first sample after teleporting so the pose reset is not
        # interpreted as motion. Refresh altitude commands from this new pose.
        sample = self.advance()
        self._reset_motion_settled = False
        stable_steps = 0
        for _ in range(self.reset_settle_steps):
            self.drive(self.pursuer, 0.0, 0.0)
            self.drive(self.target, 0.0, 0.0)
            previous, previous_stamp = sample, self.sim_stamp
            sample = self.advance()
            dt = (self.sim_stamp - previous_stamp) / 1e9
            if all(self._motion_settled(before, after, dt)
                   for before, after in zip(previous[:2], sample[:2])):
                stable_steps += 1
                if stable_steps >= self.reset_stable_steps:
                    self._reset_motion_settled = True
                    break
            else:
                stable_steps = 0
        return sample

    def _motion_settled(self, before, after, dt):
        displacement = math.hypot(after.x - before.x, after.y - before.y, after.z - before.z)
        angle_change = math.hypot(
            math.remainder(after.roll - before.roll, math.tau),
            math.remainder(after.pitch - before.pitch, math.tau),
            math.remainder(after.yaw - before.yaw, math.tau),
        )
        return (dt > 0.0 and abs(after.z - self.spawn_z) <= 0.06
                and displacement <= self.reset_linear_speed_mps * dt
                and angle_change <= self.reset_angular_speed_radps * dt)

    def settled(self, start, pose):
        return (self._reset_motion_settled
                and math.hypot(start.x - pose.x, start.y - pose.y) <= 0.20
                and abs(pose.z - self.spawn_z) <= 0.15)

    def drive(self, publisher, v, yaw_rate, lateral=0.0):
        with self.condition:
            pose = self.poses[-1][1 if publisher is self.pursuer else 2]
        # Convert yaw-frame horizontal and world-vertical setpoints to body-frame Twist.
        climb = np.clip(2.0 * (self.spawn_z - pose.z), -0.6, 0.6)
        cr, sr = math.cos(pose.roll), math.sin(pose.roll)
        cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
        x = cp * v - sp * climb
        z = sp * v + cp * climb
        twist = Twist()
        twist.linear.x = float(x)
        twist.linear.y = float(cr * lateral + sr * z)
        twist.linear.z = float(-sr * lateral + cr * z)
        twist.angular.z = float(yaw_rate)
        if not publisher.publish(twist):
            raise RuntimeError("Gazebo quadrotor command was not published")

    def pursuer_command(self, action):
        forward, lateral, yaw_rate = 0.75 * (action[0] + 1), action[1], 1.5 * action[2]
        self.drive(self.pursuer, forward, yaw_rate, lateral=lateral)
        return {"v": forward, "lateral": lateral, "omega": yaw_rate}

    def target_in_camera(self, pursuer, target, cx, cy):
        dx, dy, dz = target.x - pursuer.x, target.y - pursuer.y, target.z - pursuer.z
        c, s = math.cos(pursuer.yaw), math.sin(pursuer.yaw)
        cr, sr = math.cos(pursuer.roll), math.sin(pursuer.roll)
        cp, sp = math.cos(pursuer.pitch), math.sin(pursuer.pitch)
        forward = c * dx + s * dy
        lateral = -s * dx + c * dy
        depth = cp * forward - sp * dz - self.camera_x
        vertical = sp * forward + cp * dz
        horizontal = math.atan2(cr * lateral + sr * vertical, depth)
        elevation = math.atan2(-sr * lateral + cr * vertical, depth)
        width, height = self.camera_resolution
        vertical_limit = math.atan(math.tan(HORIZONTAL_FOV_HALF_RAD) * height / width)
        return horizontal, depth > 0 and abs(horizontal) <= HORIZONTAL_FOV_HALF_RAD and abs(elevation) <= vertical_limit

    def measure(self, pursuer, target, collided):
        info = super().measure(pursuer, target, collided)
        info.update(z=pursuer.z, target_z=target.z, yaw=pursuer.yaw)
        return info

    def outside(self, pursuer):
        return super().outside(pursuer) or abs(pursuer.z - self.spawn_z) > 0.50


class IdealQuadTracking(QuadTracking):
    """Fixed-altitude kinematics with Gazebo RGB and contact sensing."""

    def build_world(self, template, output, rtf):
        if self.stage not in IDEAL_VELOCITY_STAGES:
            raise ValueError("Ideal velocity mode requires quadrotor stage A, B, or C1")
        super().build_world(template, output, rtf)
        tree = ET.parse(output)
        for name in ("pursuer", "target"):
            model = tree.getroot().find(f"./world/model[@name='{name}']")
            for plugin in model.findall("plugin"):
                if "multicopter" in plugin.get("filename", ""):
                    model.remove(plugin)
            for link in model.findall("link"):
                gravity = link.find("gravity")
                if gravity is None:
                    gravity = ET.SubElement(link, "gravity")
                gravity.text = "false"
            # Hold zero physical velocity between the poses set by the
            # integration, including after contact and during reset.
            ET.SubElement(model, "plugin", filename="gz-sim-velocity-control-system",
                          name="gz::sim::systems::VelocityControl")
        tree.write(output, encoding="utf-8", xml_declaration=True)

    def drive(self, publisher, v, yaw_rate, lateral=0.0):
        command = (float(v), float(lateral), float(yaw_rate))
        if publisher is self.pursuer:
            self.pursuer_velocity = command
        else:
            self.target_velocity = command

    @staticmethod
    def integrate_pose(pose, command):
        forward, lateral, yaw_rate = command
        half_turn = yaw_rate * DT / 2
        # Integrate a constant body-frame velocity through the whole turn.
        scale = DT * math.sin(half_turn) / half_turn if half_turn else DT
        c, s = math.cos(pose.yaw + half_turn), math.sin(pose.yaw + half_turn)
        return Pose(
            x=pose.x + scale * (c * forward - s * lateral),
            y=pose.y + scale * (s * forward + c * lateral),
            yaw=math.remainder(pose.yaw + 2 * half_turn, math.tau),
            z=ALTITUDE_M,
        )

    def advance_motion(self):
        with self.condition:
            _, pursuer, target = self.poses[-1]
        self._teleport(self.integrate_pose(pursuer, self.pursuer_velocity),
                       self.integrate_pose(target, self.target_velocity))
        return self.advance()


class FreeFlightQuadTracking(IdealQuadTracking):
    """Obstacle-free RGB tracking with world-vertical velocity and 3D distance."""

    neutral_action = (*IdealQuadTracking.neutral_action, 0.0)
    max_vertical_speed_mps = 0.8
    altitude_limits_m = (1.3, 4.7)
    target_amplitude_m = (0.90, 1.10)
    target_period_s = (10.0, 14.0)

    @classmethod
    def free_flight_config(cls):
        return dict(action_order=["forward_mps", "lateral_mps", "yaw_rate_radps", "vertical_mps"],
                    distance_metric="euclidean_3d", spawn_altitude_m=cls.spawn_z,
                    altitude_limits_m=cls.altitude_limits_m, max_vertical_speed_mps=cls.max_vertical_speed_mps,
                    target_altitude_amplitude_m=cls.target_amplitude_m, target_altitude_period_s=cls.target_period_s)

    def tracking_start(self, rng):
        start, target = super().tracking_start(rng)
        dx, dy = target.x - start.x, target.y - start.y
        scale = self.target_gap_m / math.hypot(dx, dy)
        return (replace(start, z=self.spawn_z),
                replace(target, x=start.x + scale * dx, y=start.y + scale * dy, z=self.spawn_z))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        rng = self.np_random
        amplitude = rng.uniform(*self.target_amplitude_m) * rng.choice((-1.0, 1.0))
        period = rng.uniform(*self.target_period_s)
        elapsed_s = np.arange(self.duration + 1) * DT
        self.target_altitudes = self.spawn_z + amplitude * np.sin(math.tau * elapsed_s / period)
        self.pursuer_vertical_speed = 0.0
        self.target_vertical_speed = 0.0
        return obs, info

    def _teleport(self, start, target, *, pillars=()):
        super()._teleport(start, target, pillars=pillars, use_pose_altitude=True)

    def tracking_distance(self, pursuer, target):
        return math.hypot(target.x - pursuer.x, target.y - pursuer.y, target.z - pursuer.z)

    def pursuer_command(self, action):
        command = super().pursuer_command(action)
        self.pursuer_vertical_speed = float(action[3]) * self.max_vertical_speed_mps
        command["vz"] = self.pursuer_vertical_speed
        return command

    def advance_motion(self):
        with self.condition:
            _, pursuer, target = self.poses[-1]
        pursuer_next = self.integrate_pose(pursuer, self.pursuer_velocity)
        target_next = self.integrate_pose(target, self.target_velocity)
        target_z = float(self.target_altitudes[self.elapsed + 1])
        self.target_vertical_speed = (target_z - target.z) / DT
        self._teleport(replace(pursuer_next, z=pursuer.z + DT * self.pursuer_vertical_speed),
                       replace(target_next, z=target_z))
        return self.advance()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["target_vz"] = self.target_vertical_speed
        return obs, reward, terminated, truncated, info

    def outside(self, pursuer):
        lower, upper = self.altitude_limits_m
        return GazeboTracking.outside(self, pursuer) or not lower <= pursuer.z <= upper



class TransverseClearanceShaping:
    """Shape the distance to the target's observed side of an obstacle's lateral envelope."""

    gain = 1.0
    lookahead_m = 4.5
    side_confirmation_m = 0.20
    safety_margin_m = 0.10

    @classmethod
    def config(cls):
        return dict(gain=cls.gain, lookahead_m=cls.lookahead_m,
                    side_confirmation_m=cls.side_confirmation_m,
                    safety_margin_m=cls.safety_margin_m,
                    reference_frame="initial_pursuer_heading",
                    terminal_potential=0.0, time_limit_is_terminal=True)

    def __init__(self, yaw, obstacle_center, obstacle_half_size, body_radius, *, obstacle_yaw=0.0):
        self.forward = (math.cos(yaw), math.sin(yaw))
        self.left = (-self.forward[1], self.forward[0])
        self.center = obstacle_center
        local_left = (-math.sin(yaw-obstacle_yaw), math.cos(yaw-obstacle_yaw))
        self.obstacle_half_width = sum(abs(axis) * half for axis, half in zip(local_left, obstacle_half_size))
        self.safe_half_width = self.obstacle_half_width + body_radius + self.safety_margin_m
        self.side = 0
        self.weight = 0.0
        self.value = 0.0

    def project(self, x, y, axis):
        return (x - self.center[0]) * axis[0] + (y - self.center[1]) * axis[1]

    def step(self, info, finished, *, allow_confirmation=True):
        ahead = -self.project(info["x"], info["y"], self.forward)
        if not finished and allow_confirmation and info["visible"] and 0.0 < ahead < self.lookahead_m:
            target_offset = self.project(info["target_x"], info["target_y"], self.left)
            confirmation = (abs(target_offset) - self.obstacle_half_width) / self.side_confirmation_m
            if confirmation > 0:
                observed_side = 1 if target_offset > 0 else -1
                if self.side in (0, observed_side):
                    self.side = observed_side
                    self.weight = max(self.weight, min(1.0, confirmation))

        lateral_position = self.project(info["x"], info["y"], self.left)
        gap = max(0.0, self.safe_half_width - self.side * lateral_position) if self.side else 0.0
        next_value = 0.0 if finished else -self.gain * self.weight * gap
        added_reward = GAMMA * next_value - self.value
        self.value = next_value
        info.update(transverse_shaping_reward=added_reward, transverse_potential=next_value,
                    transverse_gap_m=gap, transverse_side=self.side, transverse_side_weight=self.weight)
        return added_reward


class RandomizedC1QuadTracking(IdealQuadTracking):
    """Sample pillar locations and a bypass route on every reset."""

    smooth_duration_s = 22
    smooth_start_distance_m = 7.30
    smooth_entry_length_m = 5.0
    smooth_pass_distance_m = 1.55
    smooth_exit_length_m = 5.3
    smooth_speed_min_mps = 0.70
    smooth_speed_max_mps = 0.80

    @classmethod
    def smooth_route_config(cls):
        return dict(duration_s=cls.smooth_duration_s, nominal_target_start_distance_m=cls.smooth_start_distance_m,
                    entry_length_m=cls.smooth_entry_length_m, pass_distance_m=cls.smooth_pass_distance_m,
                    exit_length_m=cls.smooth_exit_length_m,
                    target_speed_mps=[cls.smooth_speed_min_mps, cls.smooth_speed_max_mps])

    def __init__(self, *args, c1_pillar_count=1, transverse_clearance_shaping=False,
                 smooth_c1_route=False, **kwargs):
        if smooth_c1_route and c1_pillar_count != 1:
            raise ValueError("The smooth C1 route requires one pillar")
        self.smooth_c1_route = smooth_c1_route
        if transverse_clearance_shaping and c1_pillar_count != 1:
            raise ValueError("Transverse clearance shaping uses the single active obstacle in single-pillar C1")
        self.transverse_clearance_shaping = transverse_clearance_shaping
        self.transverse_shaping = None
        if c1_pillar_count not in (1, 2):
            raise ValueError("Randomized C1 supports one or two pillars")
        self.c1_pillar_count = c1_pillar_count
        if smooth_c1_route:
            self.longitudinal_limit_m = 15.0
        elif c1_pillar_count == 2:
            self.longitudinal_limit_m = 24.5
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.transverse_clearance_shaping and not self.remove_pillar:
            self.transverse_shaping = TransverseClearanceShaping(
                info["yaw"], self.pillar_center, PILLAR_HALF_SIZE, math.hypot(*self.collision_half_size),
            )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.transverse_clearance_shaping:
            finished = terminated or truncated
            info["unshaped_reward"] = reward
            if self.transverse_shaping is not None:
                reward += self.transverse_shaping.step(info, finished)
            if truncated:
                # The C1 deadline ends the finite-horizon objective. SB3 must not
                # bootstrap a continuation value after the terminal potential was zeroed.
                info["time_limit"] = True
            return obs, reward, finished, False, info
        return obs, reward, terminated, truncated, info

    @property
    def pillar_centers(self):
        return self.episode_pillar_centers

    def build_world(self, template, output, rtf):
        super().build_world(template, output, rtf)
        if self.c1_pillar_count == 1 and not self.smooth_c1_route:
            return
        tree = ET.parse(output)
        world = tree.getroot().find("world")
        floor = world.find("model[@name='floor']")
        floor_length = 32 if self.smooth_c1_route else 52
        for size in floor.findall(".//box/size"):
            width, _, thickness = size.text.split()
            size.text = f"{width} {floor_length} {thickness}"
        if self.c1_pillar_count == 2 and not self.remove_pillar:
            second = deepcopy(world.find("model[@name='pillar']"))
            second.set("name", "pillar_2")
            second.find("pose").text = f"0 4 {self.spawn_z} 0 0 0"
            world.append(second)
        tree.write(output, encoding="utf-8", xml_declaration=True)

    def c1_layout(self, rng, seed):
        self.route_side = int(rng.choice((-1, 1)))
        pillar_x = float(rng.uniform(-0.55, 0.55))
        pillar_y = float(rng.uniform(-1.75, -0.95))
        self.pillar_center = (pillar_x, pillar_y)
        start = Pose(rng.uniform(-0.18, 0.18), rng.uniform(-8.35, -8.05),
                     math.pi / 2 + rng.uniform(-0.02, 0.02), self.spawn_z)
        target = Pose(pillar_x + rng.uniform(-0.18, 0.18), rng.uniform(-5.15, -4.95),
                      math.pi / 2, self.spawn_z)
        lane_x = pillar_x + self.route_side * rng.uniform(1.30, 1.46)
        wobble_side = rng.choice((-1.0, 1.0))
        self.episode_pillar_centers = (self.pillar_center,)
        second_bypass = None
        if self.c1_pillar_count == 2:
            self.duration = 30 * CONTROL_HZ
            second_x = float(rng.uniform(-0.55, 0.55))
            second_y = pillar_y + float(rng.uniform(5.0, 5.8))
            self.episode_pillar_centers += ((second_x, second_y),)
            second_lane_x = second_x - self.route_side * rng.uniform(1.30, 1.46)
            second_bypass = (second_lane_x, second_y)
        if self.smooth_c1_route:
            self.duration = self.smooth_duration_s * CONTROL_HZ
            shift_y = pillar_y - self.smooth_start_distance_m + 5.05
            start = replace(start, y=start.y + shift_y)
            target = replace(target, y=target.y + shift_y)
            if self.target_gap_m != GOAL:
                dx, dy = target.x - start.x, target.y - start.y
                approach_fraction = (GOAL - self.target_gap_m) / math.hypot(dx, dy)
                start = replace(start, x=start.x + dx * approach_fraction, y=start.y + dy * approach_fraction)
            self.plan_smooth_target_route(target, lane_x, pillar_y)
        else:
            self.plan_target_route(target, lane_x, pillar_y, wobble_side, second_bypass)
        names = ("pillar", "pillar_2")
        pillars = () if self.remove_pillar else tuple(
            (name, Pose(x, y, z=self.spawn_z)) for name, (x, y) in zip(names, self.pillar_centers)
        )
        return start, target, pillars

    def plan_target_route(self, target, lane_x, pillar_y, wobble_side, second_bypass=None):
        approach_end = pillar_y - 2.50
        bypass_end = pillar_y - 1.28
        approach_length = approach_end - target.y
        amplitude = min(0.22, 0.22 * approach_length)

        sections = np.minimum(np.arange(self.duration) // (3 * CONTROL_HZ), len(self.target_speeds) - 1)
        travel = np.r_[0.0, np.cumsum(self.target_speeds[sections] * DT)]
        # Parameterize the forward-only curve by distance so turns do not increase target speed.
        y_length = float(travel[-1]) + 1.0
        y = np.linspace(target.y, target.y + y_length, math.ceil(y_length / 0.005) + 1)
        phase = np.clip((y - target.y) / approach_length, 0.0, 1.0)
        blend = np.clip((y - approach_end) / (bypass_end - approach_end), 0.0, 1.0)
        x = (target.x + wobble_side * amplitude * np.sin(math.tau * phase) ** 3
             + (lane_x - target.x) * blend ** 2 * (3.0 - 2.0 * blend))

        if second_bypass is not None:
            second_lane_x, second_y = second_bypass
            start_second = pillar_y + 1.28
            end_second = second_y - 1.28
            blend_second = np.clip((y - start_second) / (end_second - start_second), 0.0, 1.0)
            x += (second_lane_x - lane_x) * blend_second ** 2 * (3.0 - 2.0 * blend_second)

        self.set_target_path(target, x, y, travel)

    def plan_smooth_target_route(self, target, lane_x, pillar_y):
        speed = np.interp(self.target_speeds[0], (0.60, 0.95),
                          (self.smooth_speed_min_mps, self.smooth_speed_max_mps))
        travel = np.arange(self.duration + 1) * speed * DT
        length = travel[-1] + 1.0
        y = np.linspace(target.y, target.y + length, math.ceil(length / 0.005) + 1)
        entry_start = pillar_y - self.smooth_pass_distance_m - self.smooth_entry_length_m
        exit_start = pillar_y + self.smooth_pass_distance_m

        def blend(start, length):
            u = np.clip((y - start) / length, 0.0, 1.0)
            return u**3 * (10.0 + u * (-15.0 + 6.0 * u))

        # The quintic joins the main line and bypass lane with continuous
        # heading and curvature. Sample by arc length for constant speed.
        lateral_fraction = blend(entry_start, self.smooth_entry_length_m) - blend(exit_start, self.smooth_exit_length_m)
        x = target.x + (lane_x - target.x) * lateral_fraction
        self.set_target_path(target, x, y, travel)

    def set_target_path(self, target, x, y, travel):
        arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        path_x = np.interp(travel, arc, x)
        path_y = np.interp(travel, arc, y)
        dx, dy = np.diff(path_x), np.diff(path_y)
        yaw = np.r_[target.yaw, np.arctan2(dy, dx)]
        self.target_path = tuple(Pose(float(px), float(py), float(theta), self.spawn_z)
                                 for px, py, theta in zip(path_x, path_y, yaw))
        self.target_motion = tuple(zip(np.hypot(dx, dy) / DT, np.diff(yaw) / DT))

    def target_command(self):
        return self.target_motion[self.elapsed]

    def advance_motion(self):
        with self.condition:
            _, pursuer, _ = self.poses[-1]
        self._teleport(self.integrate_pose(pursuer, self.pursuer_velocity),
                       self.target_path[self.elapsed + 1])
        return self.advance()

    def measure(self, pursuer, target, collided):
        info = super().measure(pursuer, target, collided)
        info.update(pillar_x=self.pillar_center[0], pillar_y=self.pillar_center[1])
        if self.smooth_c1_route or self.c1_pillar_count == 2:
            half_size = TARGET_CUBE_SIDE_M / 2 if self.smooth_c1_route else 0.42
            target_clearance = min(pillar_clearance(target, center, (half_size, half_size))
                                   for center in self.pillar_centers)
            if not self.remove_pillar and target_clearance <= 0.0:
                raise RuntimeError("The scripted target trajectory overlapped a pillar")
            info["target_clearance_m"] = target_clearance
        if self.c1_pillar_count == 2:
            second_x, second_y = self.pillar_centers[1]
            info.update(pillar_2_x=second_x, pillar_2_y=second_y)
        return info


class TruthQuadTracking(IdealQuadTracking):
    """Use measured simulator state in place of the policy's camera input."""

    fields = (
        "target_forward_m", "target_left_m", "target_relative_yaw_sin", "target_relative_yaw_cos",
        "pillar_forward_m", "pillar_left_m", "pillar_clearance_m", "pillar_present",
        "pursuer_yaw_sin", "pursuer_yaw_cos", "target_visible",
    )

    target_motion_fields = ("target_speed_mps", "target_yaw_rate_radps")

    @classmethod
    def fields_for(cls, observe_target_motion):
        return cls.fields + cls.target_motion_fields if observe_target_motion else cls.fields

    def __init__(self, *args, observe_target_motion=False, **kwargs):
        self.observe_target_motion = observe_target_motion
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(self.fields_for(observe_target_motion)),), dtype=np.float32,
        )

    def measure(self, pursuer, target, collided):
        info = super().measure(pursuer, target, collided)
        cos_yaw, sin_yaw = math.cos(pursuer.yaw), math.sin(pursuer.yaw)

        def relative(x, y):
            dx, dy = x - pursuer.x, y - pursuer.y
            return cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy

        yaw_error = target.yaw - pursuer.yaw
        has_pillar = self.stage == "C1" and not self.remove_pillar
        pillar_xy = relative(*self.pillar_center) if has_pillar else (0.0, 0.0)
        clearance = pillar_clearance(pursuer, self.pillar_center, self.collision_half_size) if has_pillar else 0.0
        values = (
            *relative(target.x, target.y), math.sin(yaw_error), math.cos(yaw_error),
            *pillar_xy, clearance, float(has_pillar), sin_yaw, cos_yaw, float(info["visible"]),
        )
        if self.observe_target_motion:
            # Ideal integration uses this velocity for the step that produced the observation.
            speed, _, yaw_rate = self.target_velocity
            values += (speed, yaw_rate)
        self.truth_observation = np.asarray(values, dtype=np.float32)
        return info

    def observation(self):
        return self.truth_observation
