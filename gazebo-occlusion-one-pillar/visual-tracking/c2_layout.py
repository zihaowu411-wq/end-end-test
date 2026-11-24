"""Seeded pillar layouts and smooth target trajectories for Stage C2."""

from dataclasses import dataclass
import math

import numpy as np

FIELD_SIZE_M = (42.0, 84.0)
COLUMNS_M = (-14.1, -4.7, 4.7, 14.1)
ROWS_M = (-22.5, -7.5, 7.5, 22.5)
PILLAR_FOOTPRINTS_M = ((1.35, 1.45), (2.1, 1.2), (1.25, 2.0))
PILLAR_COUNT = len(COLUMNS_M) * len(ROWS_M)
ROUTE_DURATIONS_S = {"weave": 96, "figure-eight": 150}
ROUTE_MODES = ("mixed", *ROUTE_DURATIONS_S)
TARGET_SPEED_MPS = (0.70, 0.80)
MIN_SEPARATION_M = 6.0
MIN_TARGET_CLEARANCE_M = 0.45
MIN_TURN_RADIUS_M = 2.5
SAMPLE_SPACING_M = 0.02


@dataclass(frozen=True)
class Pillar:
    x: float
    y: float
    yaw: float
    half_size: tuple[float, float]

    @property
    def radius(self):
        return math.hypot(*self.half_size)

    @property
    def extent_x(self):
        return abs(math.cos(self.yaw)) * self.half_size[0] + abs(math.sin(self.yaw)) * self.half_size[1]

    @property
    def extent_y(self):
        return abs(math.sin(self.yaw)) * self.half_size[0] + abs(math.cos(self.yaw)) * self.half_size[1]

    def local_xy(self, x, y):
        dx, dy = x - self.x, y - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy


@dataclass(frozen=True)
class Encounter:
    pillar_index: int
    side: int
    sample_index: int


@dataclass(frozen=True)
class Layout:
    kind: str
    pillars: tuple[Pillar, ...]
    encounters: tuple[Encounter, ...]
    xy: np.ndarray
    yaw: np.ndarray
    speed_mps: float
    minimum_pillar_separation_m: float
    minimum_target_clearance_m: float
    minimum_turn_radius_m: float


def footprint(index):
    return PILLAR_FOOTPRINTS_M[(index + index // len(COLUMNS_M)) % len(PILLAR_FOOTPRINTS_M)]


def sample_pillars(rng):
    pillars = []
    for row in ROWS_M:
        for column in COLUMNS_M:
            width, length = footprint(len(pillars))
            pillars.append(Pillar(
                float(column + rng.uniform(-0.25, 0.25)),
                float(row + rng.uniform(-0.30, 0.30)),
                math.radians(float(rng.uniform(-30.0, 30.0))),
                (width / 2, length / 2),
            ))
    separation = min(math.hypot(p.x-q.x, p.y-q.y)-p.radius-q.radius
                     for i, p in enumerate(pillars) for q in pillars[i+1:])
    if separation < MIN_SEPARATION_M:
        raise RuntimeError("Pillar circumscribed footprints violate the minimum spacing")

    return tuple(pillars), separation


def sample_layout(rng, dt, target_radius, mode="mixed"):
    """Choose a route independently on each reset, unless a mode is requested."""
    if mode not in ROUTE_MODES:
        raise ValueError(f"Unknown C2 route mode: {mode}")
    if mode == "mixed":
        mode = "weave" if rng.integers(2)==0 else "figure-eight"
    if mode == "weave":
        return sample_weave(rng,dt,target_radius)
    return sample_figure_eight(rng,dt,target_radius)


def sample_figure_eight(rng, dt, target_radius):
    """Sample an asymmetric figure-eight through a block of four pillars."""
    pillars, separation = sample_pillars(rng)
    duration = ROUTE_DURATIONS_S["figure-eight"]
    ticks = round(duration / dt)
    phase = np.linspace(0.0, math.tau, 4097)
    sine, cosine = np.sin(phase), np.cos(phase)
    sin2, cos2 = np.sin(2*phase), np.cos(2*phase)
    sin4, cos4 = np.sin(4*phase), np.cos(4*phase)

    for _ in range(768):
        row = int(rng.integers(len(ROWS_M)-1))
        column = int(rng.integers(len(COLUMNS_M)-1))
        selected = tuple((row+r)*len(COLUMNS_M)+column+c for r in range(2) for c in range(2))
        center = np.mean([(pillars[i].x, pillars[i].y) for i in selected], axis=0)
        longitudinal = float(rng.uniform(12.5, 15.9))
        lateral = float(rng.uniform(7.8, 10.3))
        taper, lateral_skew, longitudinal_skew = rng.uniform(
            (-0.16, -0.45, -0.75), (0.12, 0.45, 0.75),
        )
        sign_x, sign_y = rng.choice((-1, 1), size=2)

        x = lateral*sin2*(1.0+taper*cos2) + lateral_skew*sine**3
        y = longitudinal*sine + longitudinal_skew*sine**4
        dx = 2*lateral*(cos2+taper*cos4) + 3*lateral_skew*sine**2*cosine
        dy = longitudinal*cosine + 4*longitudinal_skew*sine**3*cosine
        ddx = -4*lateral*sin2-8*lateral*taper*sin4+3*lateral_skew*(2*sine*cosine**2-sine**3)
        ddy = -longitudinal*sine+4*longitudinal_skew*(3*sine**2*cosine**2-sine**4)
        curvature = abs(dx*ddy-dy*ddx) / np.power(dx**2+dy**2, 1.5)
        minimum_radius = 1.0 / float(np.max(curvature))
        if minimum_radius < MIN_TURN_RADIUS_M:
            continue

        loop = center + np.column_stack((sign_x*x, sign_y*y))
        tangent = np.array((sign_x*dx[0], sign_y*dy[0]))
        tangent /= np.linalg.norm(tangent)
        lead = float(rng.uniform(7.6, 9.6))
        tail = float(rng.uniform(4.0, 6.0))
        offset = float(rng.choice((-1., 1.)) * rng.uniform(2.8, 4.1))
        normal = np.array((-tangent[1], tangent[0]))
        u = np.linspace(0.0, 1.0, math.ceil((lead+abs(offset))/SAMPLE_SPACING_M)+1)[:-1]
        blend = u**3*(10.0+u*(-15.0+6.0*u))
        approach = center + (u-1.0)[:, None]*lead*tangent + (1.0-blend)[:, None]*offset*normal
        lateral_derivative = -offset*30.0*u**2*(1.0-u)**2
        approach_velocity = lead*tangent + lateral_derivative[:, None]*normal
        approach_heading = np.arctan2(approach_velocity[:,1],approach_velocity[:,0])
        approach_curvature = (lead*abs(offset*60.0*u*(1.0-u)*(1.0-2.0*u))
                              / np.power(lead**2+lateral_derivative**2,1.5))
        minimum_radius = min(minimum_radius,1.0/float(np.max(approach_curvature)))
        if minimum_radius < MIN_TURN_RADIUS_M:
            continue
        departure = np.linspace(center,center+tail*tangent,math.ceil(tail/SAMPLE_SPACING_M)+1)[1:]
        points = np.concatenate((approach, loop, departure))
        starts = approach[0] - np.array((2.0, 3.0))[:, None]*tangent
        bounds = np.asarray(FIELD_SIZE_M)/2 - 0.85
        if np.any(abs(points)>bounds) or np.any(abs(starts)>bounds):
            continue
        if min(float(np.linalg.norm(points-position, axis=1).min()) for position in starts) < 1.0:
            continue

        spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(spacing)]
        speed = float(arc[-1]) / duration
        if not TARGET_SPEED_MPS[0] <= speed <= TARGET_SPEED_MPS[1]:
            continue

        clearances = np.array([rectangle_distance(points,p)-target_radius for p in pillars])
        min_target = float(clearances.min() - spacing.max())
        if min_target < MIN_TARGET_CLEARANCE_M or max(clearances[i].min() for i in selected) > 1.9:
            continue
        if min(rectangle_distance(starts,p).min() for p in pillars) < 0.8:
            continue

        travel = np.linspace(0.0, arc[-1], ticks+1)
        path = np.column_stack([np.interp(travel,arc,points[:,axis]) for axis in range(2)])
        alignment = math.atan2(tangent[1],tangent[0])
        loop_heading = np.arctan2(sign_y*dy,sign_x*dx)
        headings = np.unwrap(np.r_[approach_heading,loop_heading,np.full(len(departure),alignment)])
        yaw = np.interp(travel,arc,headings)

        encounters = []
        for index in selected:
            sample = int(np.argmin(rectangle_distance(path,pillars[index])))
            relative = path[sample]-np.array((pillars[index].x,pillars[index].y))
            left = np.array((-math.sin(yaw[sample]), math.cos(yaw[sample])))
            encounters.append(Encounter(index,-1 if np.dot(relative,left)>0.0 else 1,sample))
        encounters.sort(key=lambda event:event.sample_index)
        if min(b.sample_index-a.sample_index for a,b in zip(encounters,encounters[1:]))*dt < 3.0:
            continue
        return Layout("figure-eight", pillars, tuple(encounters), path, yaw, speed, separation, min_target, minimum_radius)
    raise RuntimeError("Unable to sample a feasible Stage C2 figure-eight route")


def rectangle_distance(points, pillar):
    u, v = pillar.local_xy(points[:,0], points[:,1])
    return np.hypot(np.maximum(0,abs(u)-pillar.half_size[0]),np.maximum(0,abs(v)-pillar.half_size[1]))


def sample_weave(rng, dt, target_radius):
    """Sample an S-shaped route through four local obstacles, one per row."""
    pillars, separation = sample_pillars(rng)
    ticks = round(ROUTE_DURATIONS_S["weave"] / dt)
    for _ in range(160):
        columns = [int(rng.integers(len(COLUMNS_M)))]
        for _ in ROWS_M[1:]:
            first = max(0, columns[-1] - 1)
            last = min(len(COLUMNS_M) - 1, columns[-1] + 1)
            columns.append(int(rng.integers(first, last + 1)))
        if sum(a != b for a, b in zip(columns, columns[1:])) not in (1, 2):
            continue
        sides = rng.choice((-1, 1), size=len(ROWS_M))
        if not np.any(sides[1:] != sides[:-1]):
            continue
        encounters = tuple(Encounter(row * len(COLUMNS_M) + column, int(side), 0)
                           for row, (column, side) in enumerate(zip(columns, sides)))
        selected = [pillars[e.pillar_index] for e in encounters]
        first, last = selected[0], selected[-1]
        start_x = first.x + float(rng.uniform(-0.15, 0.15))
        start_y = first.y - float(rng.uniform(7.6, 8.2))
        points = [(start_y, start_x), (first.y-6.9, start_x)]
        for pillar, encounter in zip(selected, encounters):
            offset = pillar.extent_x + float(rng.uniform(1.0, 1.2))
            lane_x = pillar.x + encounter.side * offset
            points.extend(((pillar.y-1.7, lane_x), (pillar.y+1.7, lane_x)))
        finish_x = last.x + float(rng.uniform(-0.35, 0.35))
        return_y = last.y + 7.4
        points.extend(((return_y, finish_x), (FIELD_SIZE_M[1]/2-1.0, finish_x)))
        span = points[-1][0] - start_y
        y = np.linspace(start_y, points[-1][0], math.ceil(span/SAMPLE_SPACING_M)+1)
        x, slopes, curvature = smooth_lanes(y, points)
        radius = 1.0 / max(1e-12, float(np.max(curvature)))
        if radius < MIN_TURN_RADIUS_M:
            continue
        arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        speed = float(rng.uniform(*TARGET_SPEED_MPS))
        travel = np.arange(ticks+1) * dt * speed
        if not np.interp(return_y, y, arc) + 0.8 <= travel[-1] <= arc[-1]:
            continue
        if np.interp(last.y + 2.0, y, arc) + 2.0 > speed * (ROUTE_DURATIONS_S["weave"] - 4.0):
            continue
        clearance_guard = float(np.diff(arc).max())
        active = arc <= travel[-1] + clearance_guard
        # Distance to each rotated rectangle minus the cube's planar circumradius.
        min_target = math.inf
        for p in pillars:
            u, v = p.local_xy(x[active], y[active])
            d = np.hypot(np.maximum(0, abs(u)-p.half_size[0]), np.maximum(0, abs(v)-p.half_size[1]))
            min_target = min(min_target, float(d.min()) - target_radius)
        if min_target - clearance_guard < MIN_TARGET_CLEARANCE_M:
            continue
        path_x, path_y = np.interp(travel, arc, x), np.interp(travel, arc, y)
        yaw = np.arctan2(1.0, np.interp(travel, arc, slopes))
        encounters = tuple(Encounter(e.pillar_index,e.side,int(np.searchsorted(path_y,p.y)))
                           for e,p in zip(encounters,selected))
        return Layout("weave", pillars, encounters, np.column_stack((path_x, path_y)), yaw,
                      speed, separation, min_target - clearance_guard, radius)
    raise RuntimeError("Unable to sample a feasible Stage C2 route")


def smooth_lanes(y, points):
    """Join straight passes with quintic transitions continuous through curvature."""
    station, offsets = np.asarray(points).T
    index = np.minimum(np.searchsorted(station, y, side="right") - 1, len(station)-2)
    length = station[index+1] - station[index]
    delta = offsets[index+1] - offsets[index]
    u = (y - station[index]) / length
    x = offsets[index] + delta * u**3 * (10 + u * (-15 + 6*u))
    dx = (delta / length) * 30 * u**2 * (1-u)**2
    ddx = (delta / length**2) * 60 * u * (1-u) * (1-2*u)
    curvature = abs(ddx) / np.power(1 + dx**2, 1.5)
    return x, dx, curvature
