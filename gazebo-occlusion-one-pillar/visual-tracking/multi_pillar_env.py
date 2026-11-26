"""Stage C2 Gazebo environment with randomized smooth paths through a pillar field."""

from copy import deepcopy
import math
import xml.etree.ElementTree as ET

import numpy as np

from c2_layout import (
    COLUMNS_M, FIELD_SIZE_M, MIN_SEPARATION_M, MIN_TARGET_CLEARANCE_M, ROUTE_DURATIONS_S,
    MIN_TURN_RADIUS_M, PILLAR_COUNT, ROWS_M, TARGET_SPEED_MPS, footprint, sample_layout,
)
from quad_tracking_env import (
    IdealQuadTracking, ObstacleVerticalMotion, PILLAR_HEIGHT_M, TARGET_CUBE_SIDE_M, TransverseClearanceShaping,
)
from tracking_env import CONTROL_HZ, DT, GAMMA, Pose, blocked_by_pillar, pillar_clearance


class RouteClearanceShaping:
    """Advance the C1 potential through the planned sequence of C2 obstacles."""

    @classmethod
    def config(cls):
        return dict(TransverseClearanceShaping.config(), reference_frame="local_route_tangent_at_encounter",
                    obstacle_sequence="planned_encounters", obstacle_extent="rotated_footprint",
                    end_encounter="pursuer_cleared_or_target_left_local_range")

    def __init__(self, layout, body_radius):
        self.encounters = []
        for event in layout.encounters:
            pillar = layout.pillars[event.pillar_index]
            yaw = float(layout.yaw[event.sample_index])
            angle = yaw - pillar.yaw
            extent = (abs(math.cos(angle))*pillar.half_size[0] + abs(math.sin(angle))*pillar.half_size[1]
                      + body_radius)
            shape = TransverseClearanceShaping(yaw, (pillar.x, pillar.y), pillar.half_size, body_radius,
                                               obstacle_yaw=pillar.yaw)
            self.encounters.append((event, shape, extent))
        self.index = 0
        self.approached = False

    def step(self, info, elapsed, finished):
        previous_value = self.encounters[self.index][1].value if self.index < len(self.encounters) else 0.0
        if not finished:
            while self.index < len(self.encounters):
                event, shape, extent = self.encounters[self.index]
                pursuer_forward = shape.project(info["x"], info["y"], shape.forward)
                pursuer_range = math.hypot(info["x"]-shape.center[0], info["y"]-shape.center[1])
                target_forward = shape.project(info["target_x"], info["target_y"], shape.forward)
                target_range = math.hypot(info["target_x"]-shape.center[0], info["target_y"]-shape.center[1])
                self.approached |= (-shape.lookahead_m < pursuer_forward < 0.0
                                    and pursuer_range <= shape.lookahead_m)
                cleared = self.approached and pursuer_forward > extent
                unavailable = (elapsed > event.sample_index and target_forward > extent
                               and target_range > shape.lookahead_m)
                if not (cleared or unavailable):
                    break
                self.index += 1
                self.approached = False

        if self.index < len(self.encounters):
            event, shape, _ = self.encounters[self.index]
            target_range = math.hypot(info["target_x"]-shape.center[0], info["target_y"]-shape.center[1])
            # The route identifies the pillar and local frame; a visible target determines its side.
            shape.step(info, finished, allow_confirmation=target_range <= shape.lookahead_m)
            next_value = shape.value
            active_pillar = event.pillar_index + 1
        else:
            info.update(transverse_potential=0.0, transverse_gap_m=0.0,
                        transverse_side=0, transverse_side_weight=0.0)
            next_value, active_pillar = 0.0, 0
        # Keep one potential difference across obstacle changes as well as terminal steps.
        reward = GAMMA * next_value - previous_value
        info.update(transverse_shaping_reward=reward, c2_active_pillar=active_pillar,
                    c2_encounters_advanced=self.index)
        return reward


class MultiPillarQuadTracking(IdealQuadTracking):
    """Keep the C1 camera and actions while varying C2 geometry and route on reset."""

    require_interior_encounters = False

    def __init__(self, *args, c2_route="mixed", transverse_clearance_shaping=False, **kwargs):
        self.c2_route = c2_route
        self.transverse_clearance_shaping = transverse_clearance_shaping
        self.route_shaping = None
        super().__init__(*args, **kwargs)

    @classmethod
    def layout_config(cls, route="mixed"):
        durations = ROUTE_DURATIONS_S if route == "mixed" else {route: ROUTE_DURATIONS_S[route]}
        return dict(field_size_m=list(FIELD_SIZE_M), pillars=PILLAR_COUNT, durations_s=durations,
                    route_obstacles=4, route_selection=route, target_speed_mps=list(TARGET_SPEED_MPS),
                    minimum_pillar_surface_separation_m=MIN_SEPARATION_M,
                    minimum_target_clearance_m=MIN_TARGET_CLEARANCE_M, minimum_turn_radius_m=MIN_TURN_RADIUS_M)

    def build_world(self, template, output, rtf):
        super().build_world(template, output, rtf)
        tree = ET.parse(output)
        world = tree.getroot().find("world")
        floor = world.find("model[@name='floor']")
        for size in floor.findall(".//box/size"):
            thickness = size.text.split()[-1]
            size.text = f"{FIELD_SIZE_M[0]} {FIELD_SIZE_M[1]} {thickness}"
        template_pillar = world.find("model[@name='pillar']")
        if template_pillar is not None:
            world.remove(template_pillar)
            for index in range(PILLAR_COUNT):
                row, col = divmod(index, len(COLUMNS_M))
                pillar = deepcopy(template_pillar)
                pillar.set("name", f"c2_pillar_{index+1:02d}")
                pillar.find("pose").text = f"{COLUMNS_M[col]} {ROWS_M[row]} {self.spawn_z} 0 0 0"
                width, length = footprint(index)
                for size in pillar.findall(".//box/size"):
                    size.text = f"{width} {length} {PILLAR_HEIGHT_M}"
                world.append(pillar)
        if self.record:
            overview = world.find("model[@name='overview']")
            overview.find("pose").text = "0 0 107 0 1.5707963268 0"
            camera = overview.find("link/sensor/camera")
            camera.find("image/width").text = "960"
            camera.find("image/height").text = "540"
            camera.find("clip/far").text = "160"
        tree.write(output, encoding="utf-8", xml_declaration=True)

    def c2_layout(self, rng):
        radius = math.hypot(TARGET_CUBE_SIDE_M/2, TARGET_CUBE_SIDE_M/2)
        self.layout = sample_layout(rng, DT, radius, mode=self.c2_route,
                                    require_interior_encounters=self.require_interior_encounters)
        self.duration = len(self.layout.xy)-1
        self.route_shaping = None
        if self.transverse_clearance_shaping and not self.remove_pillar:
            self.route_shaping = RouteClearanceShaping(self.layout, math.hypot(*self.collision_half_size))
        self.target_path = tuple(Pose(float(x), float(y), float(yaw), self.spawn_z)
                                 for (x, y), yaw in zip(self.layout.xy, self.layout.yaw))
        distances = np.linalg.norm(np.diff(self.layout.xy, axis=0), axis=1)
        self.target_motion = tuple(zip(distances / DT, np.diff(self.layout.yaw) / DT))
        target = self.target_path[0]
        start = Pose(target.x-self.target_gap_m*math.cos(target.yaw),
                     target.y-self.target_gap_m*math.sin(target.yaw), target.yaw, self.spawn_z)
        self.route_checkpoints_reached = 0
        pillars = () if self.remove_pillar else tuple(
            (f"c2_pillar_{index+1:02d}", Pose(p.x, p.y, p.yaw, self.spawn_z))
            for index, p in enumerate(self.layout.pillars)
        )
        return start, target, pillars

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not self.transverse_clearance_shaping:
            return obs, reward, terminated, truncated, info
        finished = terminated or truncated
        info["unshaped_reward"] = reward
        if self.route_shaping is not None:
            reward += self.route_shaping.step(info,self.elapsed,finished)
        if truncated:
            info["time_limit"] = True
        return obs, reward, finished, False, info

    def target_command(self):
        return self.target_motion[self.elapsed]

    def next_target_pose(self, target):
        return self.target_path[self.elapsed + 1]

    def target_occluded(self, cx, cy, target):
        for pillar in self.layout.pillars:
            camera_x, camera_y = pillar.local_xy(cx, cy)
            target_x, target_y = pillar.local_xy(target.x, target.y)
            if blocked_by_pillar(camera_x, camera_y, target_x, target_y, pillar_half_size=pillar.half_size):
                return True
        return False

    def nearest_clearance(self, pose, body_half_size):
        clearance = math.inf
        for pillar in self.layout.pillars:
            x, y = pillar.local_xy(pose.x, pose.y)
            local_pose = Pose(x, y, pose.yaw-pillar.yaw)
            clearance = min(clearance, pillar_clearance(local_pose, body_half_size=body_half_size,
                                                        pillar_half_size=pillar.half_size))
        return clearance

    def minimum_pillar_clearance(self, pursuer):
        return self.nearest_clearance(pursuer, self.collision_half_size)

    def measure(self, pursuer, target, collided):
        info = super().measure(pursuer, target, collided)
        if self.route_checkpoints_reached < len(self.layout.encounters):
            checkpoint = self.layout.encounters[self.route_checkpoints_reached]
            x, y = self.layout.xy[checkpoint.sample_index]
            if math.hypot(pursuer.x-x,pursuer.y-y) <= 1.5:
                self.route_checkpoints_reached += 1
        info["route_checkpoints_reached"] = self.route_checkpoints_reached
        info["route_mode"] = self.layout.kind
        info["episode_duration_steps"] = self.duration
        info["target_clearance_m"] = self.nearest_clearance(target, (TARGET_CUBE_SIDE_M/2,)*2)
        if not self.remove_pillar and info["target_clearance_m"] <= 0.0:
            raise RuntimeError("The Stage C2 target route overlapped a pillar")
        return info

    def outside(self, pursuer):
        width, length = FIELD_SIZE_M
        return (abs(pursuer.x) > width/2 - 0.5 or abs(pursuer.y) > length/2 - 0.5
                or self.outside_altitude(pursuer))


class FreeFlightC2QuadTracking(ObstacleVerticalMotion, MultiPillarQuadTracking):
    require_interior_encounters = True

    def target_crossing_steps(self):
        return tuple(event.sample_index for event in self.layout.encounters)
