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

CONTROL_HZ = CAMERA_HZ = 20
GAMMA = 0.99 ** (6 / CONTROL_HZ)
DT = 1 / CONTROL_HZ
STEP_NS = 2_000_000
PERIOD_NS = 1_000_000_000 // CONTROL_HZ
PHYSICS_STEPS = PERIOD_NS // STEP_NS
WIDTH, HEIGHT = 128, 72
HORIZONTAL_FOV_HALF_DEG = 35.0
HORIZONTAL_FOV_HALF_RAD = math.radians(HORIZONTAL_FOV_HALF_DEG)
FOV_PENALTY_START_DEG = 20.0
FOV_PENALTY_START_RAD = math.radians(FOV_PENALTY_START_DEG)
FOV_MAX_PENALTY = 0.20
GOAL = 3.0
TOLERANCE = 0.5
HORIZON = 12 * CONTROL_HZ
C1_START_SETBACK_M = 2.0
DISTANCE_WEIGHT = 0.85
CENTER_WEIGHT = 0.15
GAP_SCALE = 0.65
IN_BAND_BONUS = 0.20
APPROACH_WEIGHT = 0.20
LOST_REWARD = -1.85
LUO_REWARD_BASE = 1.0
LUO_POSITION_SCALE_M = 3.0
LUO_HEADING_WEIGHT = 0.1
VISIBLE_GAP_DISTANCE_WEIGHT = 0.60
VISIBLE_GAP_BEARING_WEIGHT = 0.15
VISIBLE_GAP_LOST_REWARD = -0.50
VALID_PROGRESS_IN_BAND_REWARD = 1.0
VALID_PROGRESS_VISIBLE_REWARD = 0.10
VALID_PROGRESS_LOST_REWARD = -0.35
VALID_PROGRESS_GAP_WEIGHT = 4.0
VALID_PROGRESS_GAP_SCALE_M = 3.0
VALID_PROGRESS_BEARING_WEIGHT = 2.0
VALID_PROGRESS_CENTER_PENALTY_WEIGHT = 0.25
VALID_PROGRESS_DISTANCE_REWARD_WEIGHT = 0.30
VALID_PROGRESS_DISTANCE_REWARD_SCALE_M = 0.95
LEGACY_CAR_DISTANCE_SCALE_M = 0.95
LEGACY_CAR_WEIGHTS = {
    "legacy-car": (0.70, 0.30),
    "legacy-car-band": (0.70, 0.30),
    "legacy-car-linear-band": (0.70, 0.30),
    "legacy-car-85": (0.85, 0.15),
    "legacy-car-unweighted": (1.0, 1.0),
    "legacy-car-balanced": (0.50, 0.50),
}
LEGACY_CAR_DISTANCE_BAND_WEIGHTS = {"legacy-car-band": 0.10, "legacy-car-linear-band": 0.10}
LEGACY_CAR_LINEAR_DISTANCE_SCALES = {"legacy-car-linear-band": 3.0}
LEGACY_CAR_LOST_REWARD = -0.35
LEGACY_CAR_ACTION_CHANGE_WEIGHT = 0.025
TRACKING_REWARDS = ("original", "luo", "luo-progress", "visible-gap", "valid-progress", *LEGACY_CAR_WEIGHTS)
COLLISION_PENALTY = 5.0
CLEARANCE_THRESHOLD_M = 0.50
MAX_CLEARANCE_PENALTY = 0.18
C1_LEGACY_CLEARANCE_THRESHOLD_M = 2.0
C1_LEGACY_MAX_CLEARANCE_PENALTY = 0.8
PILLAR_HALF_SIZE = (0.675, 0.725)
CHASSIS_HALF_SIZE = (0.62, 0.315)
IDEAL_VELOCITY_STAGES = ("A", "B", "C1")
PILLAR_STAGES = ("C", "C1")
STAGES = ("A", "B", *PILLAR_STAGES)
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
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0


def from_message(pose):
    q = pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    return Pose(pose.position.x, pose.position.y, yaw, pose.position.z, roll, pitch)


def angle_to(robot, x, y):
    angle = math.atan2(y - robot.y, x - robot.x) - robot.yaw
    return math.atan2(math.sin(angle), math.cos(angle))


def luo_tracking_reward(pursuer, target, position_scale_m=LUO_POSITION_SCALE_M):
    desired_x = pursuer.x + GOAL * math.cos(pursuer.yaw)
    desired_y = pursuer.y + GOAL * math.sin(pursuer.yaw)
    position_error = math.hypot(target.x - desired_x, target.y - desired_y)
    heading_error = math.remainder(target.yaw - pursuer.yaw, math.tau)
    return LUO_REWARD_BASE - position_error / position_scale_m - LUO_HEADING_WEIGHT * abs(heading_error)


def horizontal_fov_penalty(bearing):
    edge_fraction = (abs(bearing) - FOV_PENALTY_START_RAD) / (HORIZONTAL_FOV_HALF_RAD - FOV_PENALTY_START_RAD)
    return FOV_MAX_PENALTY * min(1.0, max(0.0, edge_fraction))


def visible_gap_reward(visible, distance, bearing):
    # Keep a horizontal correction signal outside the camera's field of view.
    bearing_cost = VISIBLE_GAP_BEARING_WEIGHT * min(abs(bearing) / HORIZONTAL_FOV_HALF_RAD, 2.0)
    if not visible:
        return VISIBLE_GAP_LOST_REWARD - bearing_cost
    gap_error = abs(distance - GOAL)
    gap_cost = VISIBLE_GAP_DISTANCE_WEIGHT * gap_error / (gap_error + TOLERANCE)
    return 1.0 - gap_cost - bearing_cost


def valid_progress_potential(info):
    gap_error = min(abs(info["distance"] - GOAL) / VALID_PROGRESS_GAP_SCALE_M, 1.0)
    bearing_error = abs(info["bearing"]) / math.pi
    return -(VALID_PROGRESS_GAP_WEIGHT * gap_error + VALID_PROGRESS_BEARING_WEIGHT * bearing_error)


def valid_progress_reward(info, previous_potential, terminated):
    if info["valid"] and not terminated:
        base = VALID_PROGRESS_IN_BAND_REWARD
    elif info["visible"]:
        base = VALID_PROGRESS_VISIBLE_REWARD
    else:
        base = VALID_PROGRESS_LOST_REWARD
    # SB3 bootstraps time limits. Only a true terminal transition has zero potential.
    potential = 0.0 if terminated else valid_progress_potential(info)
    center_error = min(abs(info["bearing"]) / HORIZONTAL_FOV_HALF_RAD, 1.0)
    center_penalty = VALID_PROGRESS_CENTER_PENALTY_WEIGHT * center_error
    scaled_gap_error = (info["distance"] - GOAL) / VALID_PROGRESS_DISTANCE_REWARD_SCALE_M
    distance_reward = VALID_PROGRESS_DISTANCE_REWARD_WEIGHT * math.exp(-0.5 * scaled_gap_error ** 2)
    reward = base + distance_reward - center_penalty
    return reward + GAMMA * potential - previous_potential, potential


def legacy_car_reward(visible, distance, bearing, weights=LEGACY_CAR_WEIGHTS["legacy-car"],
                      *, band_weight=0.0, linear_scale=None):
    if not visible:
        return LEGACY_CAR_LOST_REWARD
    gap_error = abs(distance - GOAL)
    if linear_scale is None:
        gap_score = 2 * math.exp(-0.5 * (gap_error / LEGACY_CAR_DISTANCE_SCALE_M) ** 2) - 1
    else:
        gap_score = 1.0 - 2.0 * min(gap_error / linear_scale, 1.0)
    if band_weight:
        band_score = 1.0 if gap_error <= TOLERANCE else -1.0
        gap_score = (1.0 - band_weight) * gap_score + band_weight * band_score
    center_score = max(0.0, 1.0 - abs(bearing) / HORIZONTAL_FOV_HALF_RAD)
    distance_weight, center_weight = weights
    return distance_weight * gap_score + center_weight * center_score


def blocked_by_pillar(x1, y1, x2, y2, center=(0.0, 0.0)):
    enter, leave = 0.0, 1.0
    for start, end, half, offset in zip((x1, y1), (x2, y2), PILLAR_HALF_SIZE, center):
        start, end = start - offset, end - offset
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


def pillar_clearance(robot, center=(0.0, 0.0), body_half_size=CHASSIS_HALF_SIZE):
    """Planar separation of the SDF chassis collision box and the pillar."""
    robot = Pose(robot.x - center[0], robot.y - center[1], robot.yaw)
    pillar_x, pillar_y = PILLAR_HALF_SIZE
    body_x, body_y = body_half_size
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    # Separating-axis test over the normals of both rectangles.
    overlap = (
        abs(robot.x) <= pillar_x + body_x * abs(c) + body_y * abs(s)
        and abs(robot.y) <= pillar_y + body_x * abs(s) + body_y * abs(c)
        and abs(c * robot.x + s * robot.y) <= body_x + pillar_x * abs(c) + pillar_y * abs(s)
        and abs(-s * robot.x + c * robot.y) <= body_y + pillar_x * abs(s) + pillar_y * abs(c)
    )
    if overlap:
        return 0.0

    def point_to_box(x, y, half_x, half_y):
        return math.hypot(max(0.0, abs(x) - half_x), max(0.0, abs(y) - half_y))

    body_to_pillar = min(
        point_to_box(robot.x + c * x - s * y, robot.y + s * x + c * y, pillar_x, pillar_y)
        for x in (-body_x, body_x) for y in (-body_y, body_y)
    )
    pillar_to_body = min(
        point_to_box(c * (x - robot.x) + s * (y - robot.y),
                     -s * (x - robot.x) + c * (y - robot.y), body_x, body_y)
        for x in (-pillar_x, pillar_x) for y in (-pillar_y, pillar_y)
    )
    return min(body_to_pillar, pillar_to_body)


def clearance_settings(stage, tracking_reward, enabled, threshold, maximum, *, remove_pillar=False):
    legacy_c1 = (stage == "C1" and not remove_pillar
                 and tracking_reward in ("legacy-car", "legacy-car-band", "legacy-car-linear-band"))
    if threshold is None:
        threshold = C1_LEGACY_CLEARANCE_THRESHOLD_M if legacy_c1 else CLEARANCE_THRESHOLD_M
    if maximum is None:
        maximum = C1_LEGACY_MAX_CLEARANCE_PENALTY if legacy_c1 else MAX_CLEARANCE_PENALTY
    return enabled or legacy_c1, threshold, maximum


def clearance_penalty(distance, threshold=CLEARANCE_THRESHOLD_M, maximum=MAX_CLEARANCE_PENALTY):
    return maximum * max(0.0, 1.0 - distance / threshold)


def write_world(template, output, stage, record, rtf, remove_pillar=False, pillar_center=(0.0, 0.0),
                camera_resolution=(WIDTH, HEIGHT)):
    tree = ET.parse(template)
    world = tree.getroot().find("world")
    world.find("physics/max_step_size").text = "0.002"
    world.find("physics/real_time_factor").text = str(rtf)
    broadcaster = world.find("plugin[@name='gz::sim::systems::SceneBroadcaster']")
    ET.SubElement(broadcaster, "dynamic_pose_hertz").text = "1000"
    ET.SubElement(world, "plugin", filename="gz-sim-contact-system", name="gz::sim::systems::Contact")
    chassis = world.find("model[@name='pursuer']/link[@name='chassis']")
    sensor = ET.SubElement(chassis, "sensor", name="body_contacts", type="contact")
    contact = ET.SubElement(sensor, "contact")
    ET.SubElement(contact, "collision").text = "body_collision"
    ET.SubElement(contact, "topic").text = CONTACT
    pillar = world.find("model[@name='pillar']")
    if stage not in PILLAR_STAGES or remove_pillar:
        world.remove(pillar)
    else:
        pose = pillar.find("pose")
        values = pose.text.split()
        values[:2] = map(str, pillar_center)
        pose.text = " ".join(values)
    front = chassis.find("sensor[@name='front_rgb']")
    front.find("update_rate").text = str(CAMERA_HZ)
    image = front.find("camera/image")
    width, height = camera_resolution
    image.find("width").text = str(width)
    image.find("height").text = str(height)
    if record:
        overhead = world.find("model[@name='overview']/link/sensor")
        overhead.find("update_rate").text = str(CAMERA_HZ)
        camera = overhead.find("camera")
        world.find("model[@name='overview']/pose").text = "-1.0 0 25 0 1.5707963268 0"
        camera.find("image/width").text = "576"
        camera.find("image/height").text = "432"
    else:
        world.remove(world.find("model[@name='overview']"))
    tree.write(output, encoding="utf-8", xml_declaration=True)


class GazeboTracking(gym.Env):
    """Observations are bytes from the onboard camera, in CHW layout."""

    metadata = {"render_modes": []}
    spawn_z = 0.223
    camera_x = 0.50
    collision_half_size = CHASSIS_HALF_SIZE
    neutral_action = (-1.0, 0.0)
    longitudinal_limit_m = 8.65
    reset_attempts = 4

    def __init__(self, template, directory, stage="A", record=False, rtf=8, penalize_clearance=False,
                 remove_pillar=False, pillar_center=(0.0, 0.0), clearance_threshold=None,
                 fixed_route_side=None, max_clearance_penalty=None,
                 tracking_reward="original", luo_position_scale_m=LUO_POSITION_SCALE_M,
                 penalize_horizontal_fov=False, camera_resolution=(WIDTH, HEIGHT)):
        super().__init__()
        self.camera_resolution = tuple(camera_resolution)
        width, height = self.camera_resolution
        self.observation_space = gym.spaces.Box(0, 255, shape=(3, height, width), dtype=np.uint8)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(len(self.neutral_action),), dtype=np.float32)
        self.stage, self.record = stage, record
        self.penalize_clearance, self.clearance_threshold, self.max_clearance_penalty = clearance_settings(
            stage, tracking_reward, penalize_clearance, clearance_threshold, max_clearance_penalty,
            remove_pillar=remove_pillar,
        )
        self.remove_pillar = remove_pillar
        self.pillar_center = pillar_center
        self.fixed_route_side = fixed_route_side
        self.tracking_reward = tracking_reward
        self.luo_position_scale_m = luo_position_scale_m
        self.penalize_horizontal_fov = penalize_horizontal_fov
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
        self.build_world(template, output, rtf)
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

    def build_world(self, template, output, rtf):
        write_world(template, output, self.stage, self.record, rtf, remove_pillar=self.remove_pillar,
                    pillar_center=self.pillar_center, camera_resolution=self.camera_resolution)

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
                # The first exposure may be offset from the control boundary.
                # A frame from the previous action must never satisfy this step.
                has_image = self.images and self.images[-1][0] > max(start, expected - PERIOD_NS)
                has_overhead = (not self.record or
                                self.overhead and self.overhead[-1][0] > max(start, expected - PERIOD_NS))
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

    @property
    def pillar_centers(self):
        return (self.pillar_center,)

    def _teleport(self, start, target, *, pillars=()):
        pair = Pose_V()
        for name, pose in (("pursuer", start), ("target", target), *pillars):
            msg = pair.pose.add(name=name)
            msg.position.x, msg.position.y, msg.position.z = pose.x, pose.y, self.spawn_z
            msg.orientation.z = math.sin(pose.yaw / 2)
            msg.orientation.w = math.cos(pose.yaw / 2)
        self.request("/world/one_pillar/set_pose_vector", pair)

    def c1_layout(self, rng, seed):
        self.route_side = self.fixed_route_side
        if self.route_side is None:
            self.route_side = (1 if seed % 2 == 0 else -1) if seed is not None else int(rng.choice((-1, 1)))
        side = self.route_side
        start = Pose(side * rng.uniform(0.20, 0.42), rng.uniform(-6.1, -5.7) - C1_START_SETBACK_M,
                     math.pi / 2 + side * rng.uniform(0.02, 0.06))
        target = Pose(side * rng.uniform(1.30, 1.46), rng.uniform(-2.60, -2.35) - C1_START_SETBACK_M,
                      math.pi / 2)
        return start, target, ()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.elapsed = 0
        self.duration = HORIZON
        self.previous_action = np.array(self.neutral_action, dtype=np.float32)
        self.target_speeds = rng.uniform(0.60, 0.95, 4)
        self.turn = float(rng.choice([-1, 1]) * rng.uniform(0.11, 0.20))
        pillars = ()
        if self.stage == "C1":
            self.duration = 13 * CONTROL_HZ
            start, target, pillars = self.c1_layout(rng, seed)
        elif self.stage == "C":
            start = Pose(rng.uniform(-3.8, -3.2), rng.uniform(-1.8, -1.4), rng.uniform(0.10, 0.24))
            target = Pose(rng.uniform(1.65, 1.95), rng.uniform(-2.6, -2.15), math.pi / 2)
            self.duration = 10 * CONTROL_HZ
        else:
            x = rng.uniform(-6.0, -5.5)
            y = rng.uniform(-1.0, 1.0)
            start = Pose(x, y, rng.uniform(-0.22, 0.22))
            target = Pose(x + rng.uniform(2.1, 4.9), y + rng.uniform(-0.85, 0.85))
        for attempt in range(self.reset_attempts):
            self.drive(self.pursuer, 0, 0)
            self.drive(self.target, 0, 0)
            self.advance((4 if attempt == 0 else 10) * PHYSICS_STEPS)
            self._teleport(start, target, pillars=pillars)
            pursuer, target_pose, _ = self.settle_after_teleport()
            if self.settled(start, pursuer) and self.settled(target, target_pose):
                break
        else:
            raise RuntimeError(f"Vehicles failed to settle at reset: {start} → {pursuer}, {target} → {target_pose}")
        info = self.measure(pursuer, target_pose, False)
        self.previous_gap_error = abs(info["distance"] - GOAL) if info["visible"] else None
        if self.tracking_reward == "valid-progress":
            self.previous_potential = valid_progress_potential(info)
        return self.observation(), info

    def settle_after_teleport(self):
        return self.advance(4 * PHYSICS_STEPS)

    def settled(self, start, pose):
        return math.hypot(start.x - pose.x, start.y - pose.y) <= 0.12

    def target_command(self):
        section = min(self.elapsed // (3 * CONTROL_HZ), 3)
        turn_starts = round((10 / 6) * CONTROL_HZ)
        turn = self.turn if self.stage == "B" and turn_starts <= self.elapsed < 6 * CONTROL_HZ else 0.0
        return self.target_speeds[section], turn

    def measure(self, pursuer, target, collided):
        distance = math.hypot(target.x - pursuer.x, target.y - pursuer.y)
        cx = pursuer.x + self.camera_x * math.cos(pursuer.yaw)
        cy = pursuer.y + self.camera_x * math.sin(pursuer.yaw)
        horizontal, in_view = self.target_in_camera(pursuer, target, cx, cy)
        occluded = (self.stage in PILLAR_STAGES and not self.remove_pillar
                    and any(blocked_by_pillar(cx, cy, target.x, target.y, center)
                            for center in self.pillar_centers))
        visible = in_view and not occluded
        valid = visible and abs(distance - GOAL) <= TOLERANCE and not collided
        info = {"time": self.elapsed * DT, "distance": distance, "bearing": horizontal,
                "visible": visible, "occluded": occluded, "valid": valid, "collision": collided,
                "x": pursuer.x, "y": pursuer.y, "target_x": target.x, "target_y": target.y,
                "image_stamp_ns": self.image_stamp, "sim_stamp_ns": self.sim_stamp}
        if self.stage == "C1":
            half_x, half_y = self.collision_half_size
            rear_y = pursuer.y - half_x * abs(math.sin(pursuer.yaw)) - half_y * abs(math.cos(pursuer.yaw))
            centers = self.pillar_centers
            passed = sum(rear_y > center_y + PILLAR_HALF_SIZE[1] for _, center_y in centers)
            info.update(route_side=self.route_side, past_pillar=passed == len(centers))
            if len(centers) > 1:
                info["pillars_passed"] = passed
        return info

    def target_in_camera(self, pursuer, target, cx, cy):
        angle = angle_to(Pose(cx, cy, pursuer.yaw), target.x, target.y)
        return angle, abs(angle) <= HORIZONTAL_FOV_HALF_RAD

    def observation(self):
        return np.ascontiguousarray(self.front_frame.transpose(2, 0, 1))

    def pursuer_command(self, action):
        v, w = 0.75 * (action[0] + 1), 1.5 * action[1]
        self.drive(self.pursuer, v, w)
        return {"v": v, "omega": w}

    def outside(self, pursuer):
        return abs(pursuer.x) > 12.15 or abs(pursuer.y) > self.longitudinal_limit_m

    def advance_motion(self):
        return self.advance()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        target_v, target_w = self.target_command()
        self.drive(self.target, target_v, target_w)
        command = self.pursuer_command(action)
        pursuer, target, collided = self.advance_motion()
        self.elapsed += 1
        info = self.measure(pursuer, target, collided)
        info.update(command, target_speed=target_v)
        failed = collided or self.outside(pursuer)
        gap_error = abs(info["distance"] - GOAL) if info["visible"] else None
        legacy_car_weights = LEGACY_CAR_WEIGHTS.get(self.tracking_reward)
        legacy_car = legacy_car_weights is not None
        if self.tracking_reward in ("luo", "luo-progress"):
            reward = luo_tracking_reward(pursuer, target, self.luo_position_scale_m)
        elif self.tracking_reward == "visible-gap":
            reward = visible_gap_reward(info["visible"], info["distance"], info["bearing"])
        elif self.tracking_reward == "valid-progress":
            reward, self.previous_potential = valid_progress_reward(info, self.previous_potential, failed)
        elif legacy_car:
            reward = legacy_car_reward(
                info["visible"], info["distance"], info["bearing"], legacy_car_weights,
                band_weight=LEGACY_CAR_DISTANCE_BAND_WEIGHTS.get(self.tracking_reward, 0.0),
                linear_scale=LEGACY_CAR_LINEAR_DISTANCE_SCALES.get(self.tracking_reward),
            )
        elif gap_error is not None:
            # Keep a gradient at large errors, where a narrow Gaussian saturates.
            gap_score = 1.0 - min(gap_error / GAP_SCALE, 3.0)
            center_score = 1.0 - min(abs(info["bearing"]) / math.radians(18), 2.0)
            reward = DISTANCE_WEIGHT * gap_score + CENTER_WEIGHT * center_score
            reward += IN_BAND_BONUS * info["valid"]
            if info["distance"] < 2.0:
                reward -= 0.5 * (2.0 - info["distance"])
        else:
            reward = LOST_REWARD
        if (self.tracking_reward in ("original", "luo-progress") and gap_error is not None
                and self.previous_gap_error is not None):
            improvement = (self.previous_gap_error - gap_error) / DT
            reward += APPROACH_WEIGHT * float(np.clip(improvement, -1.0, 1.0))
        self.previous_gap_error = gap_error
        if self.penalize_horizontal_fov:
            penalty = horizontal_fov_penalty(info["bearing"])
            info["horizontal_fov_penalty"] = penalty
            reward -= penalty
        action_change_weight = LEGACY_CAR_ACTION_CHANGE_WEIGHT if legacy_car else 0.01
        reward -= action_change_weight * float(np.mean(np.square(action - self.previous_action)))
        if self.stage == "A" and not legacy_car:
            reward -= 0.12 * (float(command["omega"]) / 1.5) ** 2
        self.previous_action = action
        # Keep the size of rewards per simulated second comparable to the 6 Hz run.
        reward *= 6 / CONTROL_HZ
        if self.stage in PILLAR_STAGES:
            distance_to_pillar = min(pillar_clearance(pursuer, center, self.collision_half_size)
                                     for center in self.pillar_centers)
            penalty = (clearance_penalty(distance_to_pillar, self.clearance_threshold, self.max_clearance_penalty)
                       if self.penalize_clearance else 0.0)
            info.update(pillar_clearance_m=distance_to_pillar, clearance_penalty=penalty)
            reward -= penalty
        if failed:
            info["valid"] = False
            reward -= COLLISION_PENALTY
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
