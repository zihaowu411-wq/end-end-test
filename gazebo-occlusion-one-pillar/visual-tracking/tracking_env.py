"""Gazebo Harmonic environment for RGB-only vehicle following."""

from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Condition
from time import monotonic, sleep
from uuid import uuid4
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node

DT = 1 / 6
PHYSICS_STEPS = 40
STEP_NS = 4_166_666
WIDTH, HEIGHT = 128, 72
GOAL = 3.0
TOLERANCE = 0.5
HORIZON = 72
FRONT = "/occlusion/front/rgb"
OVERHEAD = "/occlusion/overview/rgb"
POSE = "/world/one_pillar/dynamic_pose/info"
CONTROL = "/world/one_pillar/control"
CONTACT = "/tracking/body_contacts"


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nsec


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float = 0.0


def from_message(pose):
    q = pose.orientation
    return Pose(pose.position.x, pose.position.y,
                math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def angle_to(robot, x, y):
    angle = math.atan2(y - robot.y, x - robot.x) - robot.yaw
    return math.atan2(math.sin(angle), math.cos(angle))


def blocked_by_pillar(x1, y1, x2, y2):
    enter, leave = 0.0, 1.0
    for start, end, half in ((x1, x2, 0.675), (y1, y2, 0.725)):
        delta = end - start
        if abs(delta) < 1e-9:
            if abs(start) > half:
                return False
            continue
        edges = sorted(((-half - start) / delta, (half - start) / delta))
        enter, leave = max(enter, edges[0]), min(leave, edges[1])
        if enter > leave:
            return False
    return True


def write_world(template, output, stage, record, rtf):
    tree = ET.parse(template)
    world = tree.getroot().find("world")
    world.find("physics/real_time_factor").text = str(rtf)
    broadcaster = world.find("plugin[@name='gz::sim::systems::SceneBroadcaster']")
    ET.SubElement(broadcaster, "dynamic_pose_hertz").text = "1000"
    ET.SubElement(world, "plugin", filename="gz-sim-contact-system", name="gz::sim::systems::Contact")
    chassis = world.find("model[@name='pursuer']/link[@name='chassis']")
    sensor = ET.SubElement(chassis, "sensor", name="body_contacts", type="contact")
    contact = ET.SubElement(sensor, "contact")
    ET.SubElement(contact, "collision").text = "body_collision"
    ET.SubElement(contact, "topic").text = CONTACT
    if stage != "C":
        world.remove(world.find("model[@name='pillar']"))
    image = chassis.find("sensor[@name='front_rgb']/camera/image")
    image.find("width").text = str(WIDTH)
    image.find("height").text = str(HEIGHT)
    if record:
        camera = world.find("model[@name='overview']/link/sensor/camera")
        world.find("model[@name='overview']/pose").text = "-1.0 0 25 0 1.5707963268 0"
        camera.find("image/width").text = "576"
        camera.find("image/height").text = "432"
    else:
        world.remove(world.find("model[@name='overview']"))
    tree.write(output, encoding="utf-8", xml_declaration=True)


class GazeboTracking(gym.Env):
    """Observations are bytes from the onboard camera, in CHW layout."""

    metadata = {"render_modes": []}

    def __init__(self, template, directory, stage="A", record=False, rtf=8):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, shape=(3, HEIGHT, WIDTH), dtype=np.uint8)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.stage, self.record = stage, record
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.partition = f"visual-follow-{uuid4().hex[:12]}"
        self.condition = Condition()
        self.images = deque(maxlen=16)
        self.overhead = deque(maxlen=16)
        self.poses = deque(maxlen=850)
        self.contacts = deque(maxlen=100)
        self.error = None
        self.node = None
        self.sim = None
        self.requests = None
        self.sim_stamp = None
        self.subscriptions = []

        output = self.directory / "world.sdf"
        write_world(template, output, stage, record, rtf)
        env = dict(os.environ, GZ_PARTITION=self.partition, GZ_IP="127.0.0.1",
                   LIBGL_ALWAYS_SOFTWARE=os.environ.get("LIBGL_ALWAYS_SOFTWARE", "1"))
        env.setdefault("LP_NUM_THREADS", "2")
        env.pop("DISPLAY", None)
        self.old_transport_env = {key: os.environ.get(key) for key in ("GZ_PARTITION", "GZ_IP")}
        os.environ.update(GZ_PARTITION=self.partition, GZ_IP="127.0.0.1")
        try:
            self.node = Node()
            self.pursuer = self.node.advertise("/model/pursuer/cmd_vel", Twist)
            self.target = self.node.advertise("/model/target/cmd_vel", Twist)
            self._listen(Image, FRONT, self.images, self._image)
            self._listen(Pose_V, POSE, self.poses, self._pose)
            self._listen(Contacts, CONTACT, self.contacts, self._contact)
            if self.record:
                self._listen(Image, OVERHEAD, self.overhead, self._image)
            with (self.directory / "gazebo.log").open("w") as log:
                self.sim = subprocess.Popen(
                    ["gz", "sim", "-s", "--headless-rendering", "-v", "3", str(output)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,
                )
            self.requests = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).with_name("world_requests.py"))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
            )
            deadline = monotonic() + 20
            while CONTROL not in self.node.service_list():
                if self.sim.poll() is not None or monotonic() >= deadline:
                    raise RuntimeError(f"Gazebo did not provide {CONTROL}; see {self.directory / 'gazebo.log'}")
                sleep(0.1)
            # Discovery and sensor initialization happen before the first episode.
            sleep(0.35)
            self.advance(3 * PHYSICS_STEPS)
        except BaseException:
            self.close()
            raise

    def _listen(self, message_type, topic, queue, parser):
        def callback(message):
            try:
                sample = parser(message)
            except Exception as exc:
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()
                return
            if sample is not None:
                with self.condition:
                    if not queue or sample[0] > queue[-1][0]:
                        queue.append(sample)
                        self.condition.notify_all()

        if not self.node.subscribe(message_type, topic, callback):
            raise RuntimeError(f"Cannot subscribe to {topic}")
        self.subscriptions.append(topic)

    @staticmethod
    def _image(message):
        if message.step != message.width * 3 or len(message.data) != message.height * message.step:
            raise ValueError("Expected packed RGB8 from Gazebo")
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width, 3)
        return stamp_ns(message.header.stamp), image

    @staticmethod
    def _pose(message):
        models = {pose.name: from_message(pose) for pose in message.pose
                  if pose.name in ("pursuer", "target")}
        if len(models) == 2:
            return stamp_ns(message.header.stamp), models["pursuer"], models["target"]

    @staticmethod
    def _contact(message):
        return stamp_ns(message.header.stamp), True

    def request(self, service, message):
        kind = message.DESCRIPTOR.full_name
        payload = message.SerializeToString().hex()
        self.requests.stdin.write(f"{service} {kind} {payload}\n".encode())
        self.requests.stdin.flush()
        response = self.requests.stdout.readline().decode().strip()
        if response != "OK":
            raise RuntimeError(f"Gazebo rejected {service}: {response or 'request worker exited'}")

    def advance(self, count=PHYSICS_STEPS):
        start = self.sim_stamp or 0
        expected = start + count * STEP_NS
        self.request(CONTROL, WorldControl(pause=True, multi_step=count))
        deadline = monotonic() + 8
        with self.condition:
            while True:
                if self.error:
                    raise RuntimeError("Invalid Gazebo sensor message") from self.error
                has_pose = self.poses and self.poses[-1][0] >= expected - 8
                # A 12 Hz camera may expose its last frame up to one image period
                # before the 6 Hz control boundary.
                has_image = self.images and self.images[-1][0] >= expected - 86_000_000
                has_overhead = not self.record or self.overhead and self.overhead[-1][0] >= expected - 86_000_000
                if has_pose and has_image and has_overhead:
                    end_stamp, pursuer, target = self.poses[-1]
                    self.sim_stamp = end_stamp
                    self.image_stamp, self.front_frame = self.images[-1]
                    if self.record:
                        _, self.overhead_frame = min(
                            self.overhead, key=lambda item: abs(item[0] - self.image_stamp))
                    collided = any(start < stamp <= end_stamp for stamp, _ in self.contacts)
                    return pursuer, target, collided
                remaining = deadline - monotonic()
                if remaining <= 0 or self.sim.poll() is not None:
                    tail = [q[-1][0] if q else None for q in (self.poses, self.images, self.overhead)]
                    raise RuntimeError(f"Gazebo step timed out at {expected}, received stamps {tail}; see {self.directory / 'gazebo.log'}")
                self.condition.wait(min(remaining, 0.25))

    @staticmethod
    def drive(publisher, v, yaw_rate):
        twist = Twist()
        twist.linear.x, twist.angular.z = float(v), float(yaw_rate)
        if not publisher.publish(twist):
            raise RuntimeError("Gazebo actuator message was not published")

    def _teleport(self, start, target):
        pair = Pose_V()
        for name, pose in (("pursuer", start), ("target", target)):
            msg = pair.pose.add(name=name)
            msg.position.x, msg.position.y, msg.position.z = pose.x, pose.y, 0.223
            msg.orientation.z = math.sin(pose.yaw / 2)
            msg.orientation.w = math.cos(pose.yaw / 2)
        self.request("/world/one_pillar/set_pose_vector", pair)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.elapsed = 0
        self.duration = HORIZON
        self.previous_action = np.array([-1.0, 0.0], dtype=np.float32)
        self.target_speeds = rng.uniform(0.60, 0.95, 4)
        self.turn = float(rng.choice([-1, 1]) * rng.uniform(0.11, 0.20))
        if self.stage == "C":
            start = Pose(rng.uniform(-3.8, -3.2), rng.uniform(-1.8, -1.4), rng.uniform(0.10, 0.24))
            target = Pose(rng.uniform(1.65, 1.95), rng.uniform(-2.6, -2.15), math.pi / 2)
            self.duration = 60
        else:
            x = rng.uniform(-6.0, -5.5)
            y = rng.uniform(-1.0, 1.0)
            start = Pose(x, y, rng.uniform(-0.22, 0.22))
            target = Pose(x + rng.uniform(2.1, 4.9), y + rng.uniform(-0.85, 0.85))
        self.drive(self.pursuer, 0, 0)
        self.drive(self.target, 0, 0)
        self.advance()
        self._teleport(start, target)
        pursuer, target_pose, _ = self.advance()
        if math.hypot(start.x - pursuer.x, start.y - pursuer.y) > 0.12:
            raise RuntimeError(f"Pursuer retained velocity or reset failed: {start} → {pursuer}")
        if math.hypot(target.x - target_pose.x, target.y - target_pose.y) > 0.12:
            raise RuntimeError(f"Target retained velocity or reset failed: {target} → {target_pose}")
        info = self.measure(pursuer, target_pose, False)
        return self.observation(), info

    def target_command(self):
        section = min(self.elapsed // 18, 3)
        turn = self.turn if self.stage == "B" and 10 <= self.elapsed < 36 else 0.0
        return self.target_speeds[section], turn

    def measure(self, pursuer, target, collided):
        distance = math.hypot(target.x - pursuer.x, target.y - pursuer.y)
        cx = pursuer.x + 0.50 * math.cos(pursuer.yaw)
        cy = pursuer.y + 0.50 * math.sin(pursuer.yaw)
        angle = angle_to(Pose(cx, cy, pursuer.yaw), target.x, target.y)
        occluded = self.stage == "C" and blocked_by_pillar(cx, cy, target.x, target.y)
        visible = abs(angle) <= math.radians(35) and not occluded
        valid = visible and abs(distance - GOAL) <= TOLERANCE and not collided
        return {"time": self.elapsed * DT, "distance": distance, "bearing": angle,
                "visible": visible, "occluded": occluded, "valid": valid, "collision": collided,
                "x": pursuer.x, "y": pursuer.y, "target_x": target.x, "target_y": target.y,
                "image_stamp_ns": self.image_stamp, "sim_stamp_ns": self.sim_stamp}

    def observation(self):
        return np.ascontiguousarray(self.front_frame.transpose(2, 0, 1))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        v, w = 0.75 * (action[0] + 1), 1.5 * action[1]
        target_v, target_w = self.target_command()
        self.drive(self.target, target_v, target_w)
        self.drive(self.pursuer, v, w)
        pursuer, target, collided = self.advance()
        self.elapsed += 1
        info = self.measure(pursuer, target, collided)
        info.update(v=v, omega=w, target_speed=target_v)
        if info["visible"]:
            gap_error = abs(info["distance"] - GOAL)
            gap_score = 2 * math.exp(-0.5 * (gap_error / 0.95) ** 2) - 1
            center_score = max(0.0, 1.0 - abs(info["bearing"]) / math.radians(35))
            reward = 0.7 * gap_score + 0.3 * center_score
        else:
            reward = -0.35
        reward -= 0.025 * float(np.mean(np.square(action - self.previous_action)))
        self.previous_action = action
        outside = abs(pursuer.x) > 12.15 or abs(pursuer.y) > 8.65
        failed = collided or outside
        if failed:
            info["valid"] = False
            reward -= 5.0
        return self.observation(), float(reward), failed, not failed and self.elapsed >= self.duration, info

    def close(self):
        if self.requests and self.requests.poll() is None:
            self.requests.stdin.close()
            try:
                self.requests.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.requests.kill()
                self.requests.wait()
        if self.sim and self.sim.poll() is None:
            # Stop both the gz wrapper and its child server process.
            os.killpg(self.sim.pid, signal.SIGTERM)
            try:
                self.sim.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(self.sim.pid, signal.SIGKILL)
                self.sim.wait()
        if self.node:
            for topic in self.subscriptions:
                self.node.unsubscribe(topic)
            self.subscriptions.clear()
            sleep(0.15)
        for key, value in self.old_transport_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
