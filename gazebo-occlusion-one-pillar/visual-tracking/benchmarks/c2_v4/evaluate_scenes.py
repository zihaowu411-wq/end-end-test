"""Run a frozen C2 RGB policy on materialized V4 wall routes in Gazebo."""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from c2_layout import Pillar
from multi_pillar_env import MultiPillarQuadTracking
from quad_tracking_env import IdealQuadTracking, PILLAR_HEIGHT_M, TARGET_CUBE_SIDE_M
from tracking_env import CONTROL_HZ, DT, Pose, SAFE_EFFECTIVE_TRACKING


class V4WallTracking(MultiPillarQuadTracking):
    """Use the frozen policy's original RGB and vehicle interface on V4 geometry."""

    overview_height = 46.0

    def __init__(self, world, directory, scene, *, record=False):
        self.scene = scene
        self.walls = tuple(Pillar(w["x"], w["y"], w["yaw"], (w["length"]/2, w["width"]/2))
                           for w in scene["walls"])
        super().__init__(world, directory, stage="C2", rtf=30, record=record, target_gap_m=2.0,
                         cube_target=True, camera_resolution=(256, 144), tracking_reward=SAFE_EFFECTIVE_TRACKING)

    def build_world(self, template, output, rtf):
        IdealQuadTracking.build_world(self, template, output, rtf)
        tree = ET.parse(output)
        world = tree.getroot().find("world")
        x_half, y_half = self.scene["workspace_half_m"]
        floor = world.find("model[@name='floor']")
        for size in floor.findall(".//box/size"):
            size.text = f"{2*x_half} {2*y_half} {size.text.split()[-1]}"
        prototype = world.find("model[@name='pillar']")
        world.remove(prototype)
        for index, wall in enumerate(self.walls):
            item = deepcopy(prototype)
            item.set("name", f"v4_wall_{index:02d}")
            item.find("pose").text = f"{wall.x} {wall.y} {self.spawn_z} 0 0 {wall.yaw}"
            for size in item.findall(".//box/size"):
                size.text = f"{2*wall.half_size[0]} {2*wall.half_size[1]} {PILLAR_HEIGHT_M}"
            world.append(item)
        if self.record:
            overview = world.find("model[@name='overview']")
            overview.find("pose").text = f"0 0 {self.overview_height} 0 {math.pi/2} {math.pi/2}"
            camera = overview.find("link/sensor/camera")
            camera.find("image/width").text = "960"
            camera.find("image/height").text = "720"
        tree.write(output, encoding="utf-8", xml_declaration=True)

    def c2_layout(self, rng):
        route = np.asarray(self.scene["target_x_y_yaw"], dtype=float)
        self.duration = len(route) - 1
        self.target_path = tuple(Pose(float(x), float(y), float(yaw), self.spawn_z) for x, y, yaw in route)
        self.target_motion = tuple(zip(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1) / DT,
                                       np.diff(route[:, 2]) / DT))
        # The existing rotated-box clearance and occlusion methods use this property.
        from types import SimpleNamespace
        self.layout = SimpleNamespace(pillars=self.walls)
        start = Pose(*self.scene["pursuer_start"], self.spawn_z)
        return start, self.target_path[0], ()

    def measure(self, pursuer, target, collided):
        info = IdealQuadTracking.measure(self, pursuer, target, collided)
        info["target_clearance_m"] = self.nearest_clearance(target, (TARGET_CUBE_SIDE_M/2,)*2)
        if info["target_clearance_m"] <= 0:
            raise RuntimeError("The scripted V4 target overlapped a Gazebo wall")
        return info

    def outside(self, pursuer):
        wx, wy = self.scene["workspace_half_m"]
        return abs(pursuer.x) > wx-0.5 or abs(pursuer.y) > wy-0.5 or abs(pursuer.z-self.spawn_z)>0.5


class EvaluationVideo:
    """A 4x video combining actual onboard RGB, overhead flight and range."""

    def __init__(self, path, env):
        from PIL import Image, ImageDraw, ImageFont

        self.image, self.draw = Image, ImageDraw
        self.filter = getattr(Image, "Resampling", Image).BILINEAR
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.title = ImageFont.truetype(font, 24)
        self.label = ImageFont.truetype(font, 17)
        self.small = ImageFont.truetype(font, 15)
        self.pixels_m = 688 / (2*math.tan(.86/2)*(env.overview_height-env.spawn_z))
        self.planned = [self.project(p.x,p.y) for p in env.target_path[::4]]
        self.seconds = env.duration / CONTROL_HZ
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
             "-r", str(CONTROL_HZ), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE,
        )

    def project(self, x, y):
        return 912 + self.pixels_m*x, 344 - self.pixels_m*y

    def add(self, env, info, rows, outcome=None):
        if round(info["time"] * CONTROL_HZ) % 4 and outcome is None:
            return
        frame = self.image.new("RGB", (1280, 720), "#11212a")
        draw = self.draw.Draw(frame)
        white, grey, orange, blue = "#f4f9fb", "#cfe0e9", "#ffb087", "#71dbff"
        draw.text((24,12), f"Frozen C2 policy  |  V4 wall route  |  Seed {env.scene['seed']:02d}", font=self.title, fill=white)
        draw.text((1256,16), f"{info['time']:.1f} / {self.seconds:.0f} s  |  4x", font=self.label, fill=grey, anchor="ra")
        draw.text((24,60), "Pursuer onboard RGB", font=self.label, fill=white)
        draw.text((568,60), "Gazebo overview", font=self.label, fill=white)
        frame.paste(self.image.fromarray(env.front_frame).resize((520,292),self.filter),(24,86))
        frame.paste(self.image.fromarray(env.overhead_frame).resize((688,516),self.filter),(568,86))
        draw.line(self.planned,fill="#6aa7a6",width=2)
        for (xkey,ykey),color in ((('x','y'),blue),(('target_x','target_y'),orange)):
            x,y=self.project(info[xkey],info[ykey])
            trail=[self.project(r[xkey],r[ykey]) for r in rows[-240::4]]+[(x,y)]
            if len(trail)>1:
                draw.line(trail,fill="#263542",width=5)
                draw.line(trail,fill=color,width=2)
            draw.ellipse((x-6,y-6,x+6,y+6),outline=color,width=2)
        draw.text((568,618), "Pursuer", font=self.small, fill=blue)
        draw.text((674,618), "Target", font=self.small, fill=orange)
        draw.text((770,618), "Planned target route", font=self.small, fill="#a8d9d7")
        status = "Visible" if info["visible"] else "Occluded by wall" if info["occluded"] else "Out of view"
        draw.text((24,397), f"Gap {info['distance']:.2f} m  |  {status}", font=self.label, fill=white)
        if "pillar_clearance_m" in info:
            draw.text((24,425), f"Nearest wall clearance {info['pillar_clearance_m']:.2f} m", font=self.label, fill=grey)
        if outcome:
            draw.text((24,452), outcome.replace("_"," ").title(), font=self.label, fill="#ffcc8f")
        draw.text((24,486),"Following distance (m)",font=self.small,fill=white)
        x0, x1, y0, y1 = 59, 544, 514, 664

        def chart(t, d):
            return x0+(x1-x0)*t/self.seconds, y1-(y1-y0)*min(6,max(0,d))/6

        draw.rectangle((x0,y0,x1,y1),fill="#243742")
        draw.rectangle((x0,chart(0,2.5)[1],x1,chart(0,1.5)[1]),fill="#285148")
        for d in (1.5, 2.5, 6):
            draw.text((24,chart(0,d)[1]-8),str(d),font=self.small,fill=grey)
        for t in (0, self.seconds):
            draw.text((chart(t,0)[0],671),f"{t:.0f}s",font=self.small,fill=grey,anchor="ra" if t else "la")
        points=[chart(r["time"],r["distance"]) for r in rows[::4]]
        if len(points)>1:
            draw.line(points,fill="#ffd691",width=2)
        encoded=frame.tobytes()
        for _ in range(13 if outcome else 1):
            self.ffmpeg.stdin.write(encoded)

    def close(self):
        self.ffmpeg.stdin.close()
        if self.ffmpeg.wait() != 0:
            raise RuntimeError("Unable to finish the V4 evaluation video")


def check_manifest(scene, env):
    if env.duration != round(scene["duration_s"] / DT):
        raise ValueError("V4 route does not cover the configured episode")
    samples = np.asarray(scene["target_x_y_yaw"])
    if np.max(np.linalg.norm(np.diff(samples[:, :2], axis=0), axis=1) / DT) > scene["target_speed_mps"] + 1e-6:
        raise ValueError("A V4 route sample moved faster than the source setting")
    generated = ET.parse(env.directory / "world.sdf").getroot().find("world")
    walls = [m for m in generated.findall("model") if m.get("name", "").startswith("v4_wall_")]
    if len(walls) != len(scene["walls"]):
        raise RuntimeError("Scene wall count differs from the archived manifest")
    for model, wall in zip(walls, env.walls):
        p = list(map(float, model.findtext("pose").split()))
        if (p[0], p[1], p[5]) != (wall.x, wall.y, wall.yaw):
            raise RuntimeError("A Gazebo wall pose differs from its manifest")


def evaluate(args, scene):
    policy = None
    if args.model:
        import torch
        from sb3_contrib import RecurrentPPO

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        policy = RecurrentPPO.load(args.model, device="cpu")
        if (policy.num_timesteps != 262144 or policy.observation_space.shape != (3,144,256)
                or policy.action_space.shape != (3,)):
            raise ValueError("Expected the trained C2 RGB final checkpoint at cumulative step 262144")
    args.output.mkdir(parents=True, exist_ok=False)
    env = V4WallTracking(args.world, args.output / "sim", scene, record=args.video)
    video = None
    try:
        obs, initial = env.reset(seed=scene["seed"])
        check_manifest(scene, env)
        if not initial["valid"] or abs(initial["distance"]-2.0)>1e-6 or obs.shape!=(3,144,256):
            raise RuntimeError("V4 scene failed the 2 m RGB reset check")
        if args.video:
            video = EvaluationVideo(args.output / "evaluation.mp4", env)
            video.add(env, initial, [])
        rows, state, episode_start = [], None, np.ones(1, dtype=bool)
        visible = bool(initial["visible"])
        last_loss = None
        hidden_episodes, recovered, occlusions = 0, [], 0
        stop = "smoke"
        for index in range(args.smoke_steps if policy is None else env.duration):
            if policy is None:
                action = env.neutral_action
            else:
                action, state = policy.predict(obs, state=state, episode_start=episode_start, deterministic=True)
                episode_start[:] = False
            obs, _, terminated, truncated, info = env.step(action)
            target = env.target_path[index+1]
            if math.hypot(info["target_x"]-target.x,info["target_y"]-target.y)>1e-6:
                raise RuntimeError("Gazebo playback diverges from the archived target route")
            if visible and not info["visible"]:
                last_loss = info["time"]
                hidden_episodes += 1
                occlusions += int(info["occluded"])
            if not visible and info["visible"] and last_loss is not None:
                recovered.append(info["time"]-last_loss)
                last_loss = None
            visible = bool(info["visible"])
            done = terminated or truncated
            if done:
                stop = "collision" if info["collision"] else "out_of_bounds" if terminated else "completed"
            rows.append(info)
            if video:
                video.add(env, info, rows, stop if done else None)
            if done:
                break
        with (args.output / "trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = dict(seed=scene["seed"], fingerprint=scene["fingerprint"], mode="deterministic" if policy else "smoke",
                      model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest() if policy else None,
                      steps_executed=len(rows), horizon_steps=env.duration, survival_s=rows[-1]["time"], stop=stop,
                      valid_rate=sum(r["valid"] for r in rows)/env.duration,
                      visible_rate=sum(r["visible"] for r in rows)/env.duration,
                      gap_mae_m=mean(abs(r["distance"]-2.0) for r in rows),
                      min_wall_clearance_m=min(r["pillar_clearance_m"] for r in rows),
                      min_target_clearance_m=min(r["target_clearance_m"] for r in rows),
                      wall_count=len(scene["walls"]), visibility_losses=hidden_episodes,
                      initial_wall_occlusions=occlusions, recovered=len(recovered), recovery_times_s=recovered)
        report["passed"] = stop == "completed" and report["valid_rate"]>=0.8
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    finally:
        try:
            if video:
                video.close()
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", type=Path)
    mode.add_argument("--smoke-steps", type=int)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error("--smoke-steps must be positive")
    if args.video and not args.model:
        parser.error("--video requires a policy")
    document = json.loads(args.scenes.read_text())
    if document["dt_s"] != DT or document["following_gap_m"] != 2.0:
        raise ValueError("Manifest step size or following gap differs from the C2 policy contract")
    case = next(s for s in document["cases"] if s["seed"] == args.seed)
    evaluate(args, case)
