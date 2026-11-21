import argparse
import csv
from functools import partial
import json
import math
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
import torch

from tracking_env import (
    APPROACH_WEIGHT, CAMERA_HZ, CONTROL_HZ, GOAL, LUO_POSITION_SCALE_M, PILLAR_STAGES, STAGES, TOLERANCE,
    HEIGHT, IDEAL_VELOCITY_STAGES, TRACKING_REWARDS, WIDTH, GazeboTracking, clearance_settings,
)


def evaluate(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = GazeboTracking
    if args.vehicle == "quad":
        from quad_tracking_env import IdealQuadTracking, QuadTracking, TruthQuadTracking
        quad_environment = IdealQuadTracking if getattr(args, "ideal_velocity", False) else QuadTracking
        if args.observation == "truth":
            quad_environment = TruthQuadTracking
        environment = partial(quad_environment, cube_target=getattr(args, "cube_target", False),
                              target_visual_scale=getattr(args, "target_visual_scale", 1.0))
    if args.stage == "B1":
        from b1_tracking_env import B1Tracking
        environment = partial(B1Tracking, routes=(args.b1_route,), cube_target=args.cube_target,
                              reverse_speed_mps=args.b1_reverse_speed,
                              target_visual_scale=args.target_visual_scale)
    policy = None if args.constant_speed is not None else RecurrentPPO.load(args.model, device="cpu")
    camera_resolution = tuple(args.camera_resolution)
    if policy is not None and args.observation == "rgb":
        _, height, width = policy.observation_space.shape
        camera_resolution = (width, height)
    side = args.fixed_route_side
    if args.stage in ("B", "B1"):
        side = {"left": 1, "right": -1}.get(args.b_first_turn)
    env = environment(args.world, output / "sim", args.stage, record=args.video, rtf=args.rtf,
                         penalize_clearance=args.clearance_penalty, remove_pillar=args.remove_pillar,
                         clearance_threshold=args.clearance_threshold,
                         max_clearance_penalty=args.max_clearance_penalty,
                         pillar_center=tuple(args.pillar_center), tracking_reward=args.tracking_reward,
                         luo_position_scale_m=args.luo_position_scale,
                         penalize_horizontal_fov=args.horizontal_fov_penalty,
                         camera_resolution=camera_resolution, fixed_route_side=side,
                         c1_target_detour=args.c1_target_detour, b_weave=args.b_weave)
    video = None
    try:
        if policy is not None and policy.observation_space != env.observation_space:
            raise ValueError("Checkpoint observation space does not match the evaluation environment")
        if args.video:
            from replay import Video
            video = Video(output / "tracking.mp4", args.stage)
        trace, episodes = [], []
        for number in range(args.episodes):
            seed = args.first_seed + number * args.seed_stride
            obs, info = env.reset(seed=seed)
            if args.stage == "C1":
                initial_state = {key: info[key] for key in ("x", "y", "yaw", "target_x", "target_y")}
                initial_state["target_speeds"] = env.target_speeds.tolist()
            time_to_visible, time_to_gap = [], []
            lost = None
            recovered = None
            count_lost = 0
            scores = []
            state, episode_start = None, np.ones(1, dtype=bool)
            was_visible = info["visible"]
            if video:
                video.add(env, info, seed, [])
            done = False
            while not done:
                if policy is None:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                    action[0] = args.constant_speed / 0.75 - 1
                else:
                    policy_input = np.zeros_like(obs) if args.blank_rgb else obs
                    action, state = policy.predict(policy_input, state=state, episode_start=episode_start,
                                                   deterministic=True)
                episode_start[:] = False
                obs, reward, terminated, truncated, info = env.step(action)
                step = dict(info, reward=reward, seed=seed)
                trace.append(step)
                scores.append(step)
                if was_visible and not info["visible"] and info["occluded"]:
                    lost, recovered = info["time"], None
                    count_lost += 1
                if lost is not None and recovered is None and info["visible"]:
                    recovered = info["time"]
                    time_to_visible.append(recovered - lost)
                if recovered is not None and info["valid"]:
                    time_to_gap.append(info["time"] - lost)
                    lost, recovered = None, None
                was_visible = info["visible"]
                done = terminated or truncated
                if video:
                    video.add(env, info, seed, scores)
            episodes.append(dict(
                seed=seed, return_=sum(row["reward"] for row in scores),
                valid=sum(row["valid"] for row in scores) / env.duration,
                visible=sum(row["visible"] for row in scores) / env.duration,
                gap_mae=float(np.mean([abs(row["distance"] - GOAL) for row in scores])),
                collision=bool(info["collision"]), ended_early=bool(terminated and not info.get("time_limit", False)),
                occlusion_events=count_lost, reacquired=len(time_to_visible),
                recovery_times_s=time_to_visible, gap_recovery_times_s=time_to_gap,
            ))
            if args.stage in ("B", "B1"):
                episodes[-1]["b_first_turn"] = info["b_first_turn"]
            if args.stage == "B1":
                episodes[-1].update(b1_route=info["b1_route"], b1_route_complete=info["b1_route_complete"],
                                    planned_steps=env.duration)
            if args.stage == "C1":
                after_pillar = [row for row in scores if row["past_pillar"]]
                episodes[-1].update(
                    side="right" if info["route_side"] > 0 else "left", initial_state=initial_state,
                    passed_pillar=bool(after_pillar),
                    post_pillar_valid_rate=(sum(row["valid"] for row in after_pillar) / len(after_pillar)
                                            if after_pillar else None),
                )
        with (output / "trace.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=trace[0])
            writer.writeheader()
            writer.writerows(trace)
        report = dict(
            vehicle=args.vehicle, ideal_velocity=getattr(args, "ideal_velocity", False),
            stage=args.stage, model=str(args.model), episodes=len(episodes), target_gap_m=GOAL,
            observation=args.observation, tracking_reward=args.tracking_reward, cube_target=getattr(args, "cube_target", False),
            target_visual_scale=getattr(args, "target_visual_scale", 1.0),
            horizontal_fov_penalty=args.horizontal_fov_penalty,
            blank_rgb=args.blank_rgb,
            constant_speed_mps=args.constant_speed,
            camera_hz=CAMERA_HZ, control_hz=CONTROL_HZ, camera_resolution=camera_resolution,
            gap_band_m=[GOAL - TOLERANCE, GOAL + TOLERANCE],
            valid_rate=float(np.mean([row["valid"] for row in episodes])),
            visible_rate=float(np.mean([row["visible"] for row in episodes])),
            mean_gap_error_m=float(np.mean([row["gap_mae"] for row in episodes])),
            collisions=sum(row["collision"] for row in episodes),
            early_ends=sum(row["ended_early"] for row in episodes),
            results=episodes,
        )
        if args.stage == "B1":
            report.update(b1_route=args.b1_route, b_first_turn=args.b_first_turn,
                          b1_reverse_speed=args.b1_reverse_speed)
        if args.stage == "B":
            report.update(b_weave=args.b_weave, b_first_turn=args.b_first_turn,
                          episode_steps=env.duration)
        if args.stage in PILLAR_STAGES:
            count = sum(row["occlusion_events"] for row in episodes)
            recovered = [value for row in episodes for value in row["recovery_times_s"]]
            restored_gap = [value for row in episodes for value in row["gap_recovery_times_s"]]
            report.update(occlusion_events=count, reacquired=len(recovered),
                          reacquisition_rate=len(recovered) / count if count else None,
                          mean_reacquisition_s=float(np.mean(recovered)) if recovered else None,
                          gap_restored=len(restored_gap),
                          mean_gap_recovery_s=float(np.mean(restored_gap)) if restored_gap else None)
            report["clearance_penalty_enabled"] = env.penalize_clearance
            if env.penalize_clearance:
                report["clearance_threshold_m"] = env.clearance_threshold
                report["max_clearance_penalty"] = env.max_clearance_penalty
            report["mean_episode_minimum_clearance_m"] = float(np.mean([
                min(row["pillar_clearance_m"] for row in trace if row["seed"] == episode["seed"])
                for episode in episodes
            ]))
        if args.tracking_reward in ("luo", "luo-progress"):
            report["luo_c_m"] = args.luo_position_scale
        if args.tracking_reward == "luo-progress":
            report["approach_weight"] = APPROACH_WEIGHT
        report["passed"] = report["valid_rate"] >= 0.8 and report["collisions"] == 0
        if args.stage == "C1":
            report["pillar_removed"] = args.remove_pillar
            report["pillar_center_m"] = args.pillar_center
            report["c1_target_detour"] = args.c1_target_detour
            if args.c1_target_detour:
                report["clearance_curve"] = "quadratic-two-zone"
            report["routes"] = {}
            for side in ("left", "right"):
                routes = [row for row in episodes if row["side"] == side]
                if routes:
                    report["routes"][side] = dict(
                        episodes=len(routes),
                        passed_pillar=sum(row["passed_pillar"] for row in routes),
                        valid_rate=float(np.mean([row["valid"] for row in routes])),
                        collisions=sum(row["collision"] for row in routes),
                    )
        (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)
    finally:
        if video:
            video.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=Path, default=Path("../one_pillar.sdf"))
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--model", type=Path)
    policy.add_argument("--constant-speed", type=float, help="Uncontrolled straight-line comparison, in m/s.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "B1"), default="A")
    parser.add_argument("--vehicle", choices=("car", "quad"), default="car")
    parser.add_argument("--cube-target", action="store_true",
                        help="Render the quadrotor target as a red cube.")
    parser.add_argument("--target-visual-scale", type=float, default=1.0,
                        help="Uniformly scale the red X3 target visuals around the model origin.")
    parser.add_argument("--ideal-velocity", action="store_true",
                        help="Integrate speed commands directly for quadrotor stages A, B, and C1.")
    parser.add_argument("--tracking-reward", choices=TRACKING_REWARDS,
                        help="Defaults to legacy-car for quad C1 and original otherwise.")
    parser.add_argument("--luo-position-scale", type=float, default=LUO_POSITION_SCALE_M,
                        metavar="METRES", help="Position error scale in Luo et al.'s tracking reward.")
    parser.add_argument("--horizontal-fov-penalty", action="store_true",
                        help="Include the horizontal edge penalty used during training in the recorded rewards.")
    parser.add_argument("--observation", choices=("rgb", "truth"), default="rgb")
    parser.add_argument("--camera-resolution", nargs=2, type=int, default=(WIDTH, HEIGHT),
                        metavar=("WIDTH", "HEIGHT"), help="Camera size for truth-input runs or constant-speed tests.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=20000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--rtf", type=float, default=12)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--blank-rgb", action="store_true", help="Check policy behavior with camera pixels masked.")
    parser.add_argument("--clearance-penalty", action="store_true")
    parser.add_argument("--clearance-threshold", type=float)
    parser.add_argument("--max-clearance-penalty", type=float)
    parser.add_argument("--remove-pillar", action="store_true",
                        help="Keep C1 routes and measure clearance against the original pillar footprint.")
    parser.add_argument("--pillar-center", nargs=2, type=float, metavar=("X", "Y"), default=(0.0, 0.0),
                        help="Set the C1 pillar's center or reference footprint in world coordinates.")
    parser.add_argument("--fixed-route-side", type=int, choices=(-1, 1))
    parser.add_argument("--c1-target-detour", action="store_true",
                        help="Use the C1 target detour and its two-zone clearance penalty.")
    parser.add_argument("--b-weave", action="store_true")
    parser.add_argument("--b-first-turn", choices=("left", "right"))
    from b1_tracking_env import B1_ROUTES
    parser.add_argument("--b1-route", choices=B1_ROUTES)
    parser.add_argument("--b1-reverse-speed", type=float, default=0.0, metavar="MPS")
    args = parser.parse_args()
    if not math.isfinite(args.b1_reverse_speed) or args.b1_reverse_speed < 0:
        parser.error("--b1-reverse-speed must be finite and nonnegative")
    if args.b1_reverse_speed and args.stage != "B1":
        parser.error("--b1-reverse-speed requires Stage B1")
    if args.stage == "B1" and (args.vehicle != "quad" or not args.ideal_velocity or args.observation != "rgb"
                               or args.b1_route is None):
        parser.error("Stage B1 requires an ideal-velocity quadrotor, RGB and a specific --b1-route")
    if args.b1_route and args.stage != "B1":
        parser.error("--b1-route requires Stage B1")
    if (args.b_weave and args.stage != "B") or (args.b_first_turn and args.stage not in ("B", "B1")):
        parser.error("Target trajectory options require Stage B")
    if args.fixed_route_side is not None and args.stage != "C1":
        parser.error("The fixed route side requires C1")
    if args.c1_target_detour and (args.stage != "C1" or not args.ideal_velocity):
        parser.error("The target detour requires ideal C1")
    if args.observation == "truth" and (not args.ideal_velocity or args.blank_rgb):
        parser.error("--observation truth requires --ideal-velocity and is incompatible with --blank-rgb")
    if min(args.camera_resolution) < 1:
        parser.error("--camera-resolution dimensions must be positive")
    if args.cube_target and args.vehicle != "quad":
        parser.error("--cube-target requires --vehicle quad")
    if not math.isfinite(args.target_visual_scale) or args.target_visual_scale <= 0:
        parser.error("--target-visual-scale must be finite and positive")
    if args.target_visual_scale != 1.0 and (args.vehicle != "quad" or args.cube_target):
        parser.error("--target-visual-scale requires an X3 quadrotor target")
    if args.ideal_velocity and (args.vehicle != "quad" or args.stage not in (*IDEAL_VELOCITY_STAGES, "B1")):
        parser.error("--ideal-velocity requires --vehicle quad and --stage A, B, or C1")
    if args.tracking_reward is None:
        args.tracking_reward = "legacy-car" if args.stage in ("C1", "B1") and args.vehicle == "quad" else "original"
    args.clearance_penalty, args.clearance_threshold, args.max_clearance_penalty = clearance_settings(
        args.stage, args.tracking_reward, args.clearance_penalty,
        args.clearance_threshold, args.max_clearance_penalty,
        remove_pillar=args.remove_pillar, c1_target_detour=args.c1_target_detour,
    )
    if args.tracking_reward == "visible-gap" and args.horizontal_fov_penalty:
        parser.error("visible-gap already includes a horizontal bearing penalty")
    if not math.isfinite(args.luo_position_scale) or args.luo_position_scale <= 0:
        parser.error("--luo-position-scale must be finite and positive")
    if min(args.episodes, args.seed_stride) < 1:
        parser.error("--episodes and --seed-stride must be positive")
    if args.constant_speed is not None and not 0 <= args.constant_speed <= 1.5:
        parser.error("--constant-speed must be within [0, 1.5]")
    if args.blank_rgb and not args.model:
        parser.error("--blank-rgb requires --model")
    if args.clearance_penalty and args.stage not in PILLAR_STAGES:
        parser.error("--clearance-penalty requires a stage with a pillar")
    if not math.isfinite(args.clearance_threshold) or args.clearance_threshold <= 0:
        parser.error("--clearance-threshold must be finite and positive")
    if not math.isfinite(args.max_clearance_penalty) or args.max_clearance_penalty < 0:
        parser.error("--max-clearance-penalty must be finite and nonnegative")
    if args.remove_pillar and (args.stage != "C1" or args.clearance_penalty):
        parser.error("--remove-pillar requires stage C1 with the clearance penalty disabled")
    if not all(map(math.isfinite, args.pillar_center)):
        parser.error("--pillar-center requires finite coordinates")
    if tuple(args.pillar_center) != (0.0, 0.0) and args.stage != "C1":
        parser.error("--pillar-center requires stage C1")
    evaluate(args)
