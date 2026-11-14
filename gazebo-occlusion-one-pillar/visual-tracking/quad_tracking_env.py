from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from gz.msgs10.twist_pb2 import Twist

from tracking_env import CONTACT, GazeboTracking, HEIGHT, HORIZONTAL_FOV_HALF_RAD, PHYSICS_STEPS, WIDTH, write_world


ALTITUDE_M = 3.0
PILLAR_HEIGHT_M = 6.0
X3_DEMO = Path("/usr/share/gz/gz-sim8/worlds/multicopter_velocity_control.sdf")
X3_ASSETS = Path(__file__).resolve().parent / "assets" / "x3"


def add_box_visual(link, name, pose, size, color):
    visual = ET.SubElement(link, "visual", name=name)
    ET.SubElement(visual, "pose").text = pose
    box = ET.SubElement(ET.SubElement(visual, "geometry"), "box")
    ET.SubElement(box, "size").text = size
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = color
    ET.SubElement(material, "diffuse").text = color


def x3_model(name, pose, color, source, demo):
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
    add_box_visual(base, "tracking_color", "0 0 0.065 0 0 0", "0.32 0.24 0.055", color)
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
            ET.SubElement(plugin, "maxLinearAcceleration").text = "2 2 2"
        model.append(plugin)
    return model


class QuadTracking(GazeboTracking):
    """X3 quadrotors with native thrust dynamics and a shared altitude setpoint."""

    spawn_z = ALTITUDE_M
    camera_x = 0.27
    collision_half_size = (0.25, 0.32)
    neutral_action = (-1.0, 0.0, 0.0)
    reset_attempts = 12

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
            model = x3_model(name, " ".join(values), color, source, demo)
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
                    pillar_center=self.pillar_center)

    def settle_after_teleport(self):
        # Refresh altitude commands after teleporting. A command based on the
        # pre-teleport height would otherwise persist during stabilization.
        sample = self.advance()
        for _ in range(8):
            if all(abs(pose.z - self.spawn_z) <= 0.06 for pose in sample[:2]):
                break
            self.drive(self.pursuer, 0.0, 0.0)
            self.drive(self.target, 0.0, 0.0)
            sample = self.advance(4 * PHYSICS_STEPS)
        self.drive(self.pursuer, 0.0, 0.0)
        self.drive(self.target, 0.0, 0.0)
        return self.advance(4 * PHYSICS_STEPS)

    def settled(self, start, pose):
        return math.hypot(start.x - pose.x, start.y - pose.y) <= 0.20 and abs(pose.z - self.spawn_z) <= 0.15

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
        vertical_limit = math.atan(math.tan(HORIZONTAL_FOV_HALF_RAD) * HEIGHT / WIDTH)
        return horizontal, depth > 0 and abs(horizontal) <= HORIZONTAL_FOV_HALF_RAD and abs(elevation) <= vertical_limit

    def measure(self, pursuer, target, collided):
        info = super().measure(pursuer, target, collided)
        info.update(z=pursuer.z, target_z=target.z, yaw=pursuer.yaw)
        return info

    def outside(self, pursuer):
        return super().outside(pursuer) or abs(pursuer.z - self.spawn_z) > 0.50
