from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from gymnasium import spaces
import numpy as np
from gz.msgs10.twist_pb2 import Twist

from tracking_env import (
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



class TruthQuadTracking(IdealQuadTracking):
    """Use measured simulator state in place of the policy's camera input."""

    fields = (
        "target_forward_m", "target_left_m", "target_relative_yaw_sin", "target_relative_yaw_cos",
        "pillar_forward_m", "pillar_left_m", "pillar_clearance_m", "pillar_present",
        "pursuer_yaw_sin", "pursuer_yaw_cos", "target_visible",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.fields),), dtype=np.float32)

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
        self.truth_observation = np.asarray(
            (*relative(target.x, target.y), math.sin(yaw_error), math.cos(yaw_error),
             *pillar_xy, clearance, float(has_pillar), sin_yaw, cos_yaw, float(info["visible"])),
            dtype=np.float32,
        )
        return info

    def observation(self):
        return self.truth_observation
