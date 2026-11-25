import argparse
import csv
from functools import partial
import json
import math
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import torch

from tracking_env import (
    APPROACH_WEIGHT, CAMERA_HZ, CONTROL_HZ, GOAL, LUO_POSITION_SCALE_M, PILLAR_STAGES, STAGES, TOLERANCE,
    HEIGHT, IDEAL_VELOCITY_STAGES, TRACKING_REWARDS, WIDTH, GazeboTracking, clearance_settings,
    SAFE_EFFECTIVE_TRACKING, safe_tracking_reward_config,
)


def evaluate(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = GazeboTracking
    if args.vehicle == "quad":
        from quad_tracking_env import IdealQuadTracking, QuadTracking, RandomizedC1QuadTracking, TruthQuadTracking
        quad_environment = IdealQuadTracking if getattr(args, "ideal_velocity", False) else QuadTracking
        if getattr(args, "free_flight_3d", False):
            from quad_tracking_env import FreeFlightQuadTracking

            quad_environment = FreeFlightQuadTracking
        elif args.stage == "C2":
            from multi_pillar_env import MultiPillarQuadTracking
            quad_environment = partial(
                MultiPillarQuadTracking, c2_route=getattr(args, "c2_route", "mixed"),
                transverse_clearance_shaping=getattr(args, "transverse_clearance_shaping", False),
            )
        elif args.randomized_c1:
            quad_environment = partial(
                RandomizedC1QuadTracking, c1_pillar_count=args.c1_pillar_count,
                transverse_clearance_shaping=getattr(args, "transverse_clearance_shaping", False),
                smooth_c1_route=getattr(args, "smooth_c1_route", False),
            )
        elif args.observation == "truth":
            quad_environment = partial(TruthQuadTracking, observe_target_motion=args.observe_target_motion)
        environment = partial(quad_environment, cube_target=getattr(args, "cube_target", False),
                              target_visual_scale=getattr(args, "target_visual_scale", 1.0))
    algorithm = PPO if args.feedforward else RecurrentPPO
    policy = None if args.constant_speed is not None else algorithm.load(args.model, device="cpu")
    camera_resolution = tuple(args.camera_resolution)
    if policy is not None and args.observation == "rgb":
        _, height, width = policy.observation_space.shape
        camera_resolution = (width, height)
    env = environment(args.world, output / "sim", args.stage, record=args.video, rtf=args.rtf,
                         penalize_clearance=args.clearance_penalty, remove_pillar=args.remove_pillar,
                         clearance_threshold=args.clearance_threshold,
                         max_clearance_penalty=args.max_clearance_penalty,
                         pillar_center=tuple(args.pillar_center), tracking_reward=args.tracking_reward,
                         luo_position_scale_m=args.luo_position_scale,
                         penalize_horizontal_fov=args.horizontal_fov_penalty,
                         camera_resolution=camera_resolution, target_gap_m=getattr(args, "target_gap_m", GOAL))
    video = None
    try:
        if policy is not None and policy.observation_space != env.observation_space:
            raise ValueError("Checkpoint observation space does not match the evaluation environment")
        if policy is not None and policy.action_space != env.action_space:
            raise ValueError("Checkpoint action space does not match the evaluation environment")
        if args.video:
            from replay import Video
            video = Video(output / "tracking.mp4", args.stage)
        trace, episodes = [], []
        for number in range(args.episodes):
            seed = args.first_seed + number * args.seed_stride
            obs, info = env.reset(seed=seed)
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
                gap_mae=float(np.mean([abs(row["distance"] - env.target_gap_m) for row in scores])),
                collision=bool(info["collision"]), ended_early=bool(terminated and not info.get("time_limit", False)),
                occlusion_events=count_lost, reacquired=len(time_to_visible),
                recovery_times_s=time_to_visible, gap_recovery_times_s=time_to_gap,
            ))
            if getattr(args, "free_flight_3d", False):
                episodes[-1]["altitude_mae_m"] = float(np.mean([abs(row["z"] - row["target_z"]) for row in scores]))
            if args.stage == "C1":
                after_pillar = [row for row in scores if row["past_pillar"]]
                episodes[-1].update(
                    side="right" if info["route_side"] > 0 else "left",
                    passed_pillar=bool(after_pillar),
                    post_pillar_valid_rate=(sum(row["valid"] for row in after_pillar) / len(after_pillar)
                                            if after_pillar else None),
                )
                if args.randomized_c1:
                    episodes[-1]["pillar_center_m"] = list(env.pillar_center)
                    if args.c1_pillar_count == 2:
                        episodes[-1].update(
                            pillar_centers_m=[list(center) for center in env.pillar_centers],
                            pillars_passed=max(row["pillars_passed"] for row in scores),
                            min_target_clearance_m=min(row["target_clearance_m"] for row in scores),
                        )
            if args.stage == "C2":
                episodes[-1].update(
                    route_checkpoints_reached=int(info["route_checkpoints_reached"]),
                    route_mode=info["route_mode"], episode_duration_steps=env.duration,
                    min_target_clearance_m=min(row["target_clearance_m"] for row in scores),
                )
        with (output / "trace.csv").open("w", newline="") as file:
            fields = dict.fromkeys(key for row in trace for key in row)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trace)
        report = dict(
            algorithm=type(policy).__name__ if policy is not None else None,
            observe_target_motion=args.observe_target_motion,
            vehicle=args.vehicle, ideal_velocity=getattr(args, "ideal_velocity", False),
            stage=args.stage, model=str(args.model), episodes=len(episodes), target_gap_m=env.target_gap_m,
            observation=args.observation, tracking_reward=args.tracking_reward, cube_target=getattr(args, "cube_target", False),
            target_visual_scale=getattr(args, "target_visual_scale", 1.0),
            horizontal_fov_penalty=args.horizontal_fov_penalty,
            blank_rgb=args.blank_rgb,
            constant_speed_mps=args.constant_speed,
            camera_hz=CAMERA_HZ, control_hz=CONTROL_HZ, camera_resolution=camera_resolution,
            gap_band_m=[env.target_gap_m - TOLERANCE, env.target_gap_m + TOLERANCE],
            valid_rate=float(np.mean([row["valid"] for row in episodes])),
            visible_rate=float(np.mean([row["visible"] for row in episodes])),
            mean_gap_error_m=float(np.mean([row["gap_mae"] for row in episodes])),
            collisions=sum(row["collision"] for row in episodes),
            early_ends=sum(row["ended_early"] for row in episodes),
            results=episodes,
        )
        if getattr(args, "free_flight_3d", False):
            report["flight"] = env.free_flight_config()
            report["mean_altitude_error_m"] = float(np.mean([row["altitude_mae_m"] for row in episodes]))
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
        if args.tracking_reward == SAFE_EFFECTIVE_TRACKING:
            report["reward"] = safe_tracking_reward_config(env.target_gap_m)
        if args.stage == "C2":
            report["target_route"] = env.layout_config(env.c2_route)
        if getattr(args, "smooth_c1_route", False):
            report["target_route"] = env.smooth_route_config()
        if getattr(args, "transverse_clearance_shaping", False):
            from quad_tracking_env import TransverseClearanceShaping
            if args.stage == "C2":
                from multi_pillar_env import RouteClearanceShaping as TransverseClearanceShaping
            report["reward"]["transverse_clearance_shaping"] = TransverseClearanceShaping.config()
        if args.tracking_reward in ("luo", "luo-progress"):
            report["luo_c_m"] = args.luo_position_scale
        if args.tracking_reward == "luo-progress":
            report["approach_weight"] = APPROACH_WEIGHT
        report["passed"] = report["valid_rate"] >= 0.8 and report["collisions"] == 0
        if args.stage == "C1":
            report["pillar_removed"] = args.remove_pillar
            if args.randomized_c1:
                report["randomized_c1"] = True
                report["c1_pillar_count"] = args.c1_pillar_count
            else:
                report["pillar_center_m"] = args.pillar_center
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
    parser.add_argument("--stage", choices=STAGES, default="A")
    parser.add_argument("--vehicle", choices=("car", "quad"), default="car")
    parser.add_argument("--cube-target", action="store_true",
                        help="Render the quadrotor target as a red cube.")
    parser.add_argument("--target-visual-scale", type=float, default=1.0,
                        help="Uniformly scale the red X3 target visuals around the model origin.")
    parser.add_argument("--ideal-velocity", action="store_true",
                        help="Integrate speed commands directly for quadrotor stages A, B, and C1.")
    parser.add_argument("--free-flight-3d", action="store_true",
                        help="Add vertical target motion and a vertical velocity action in obstacle-free quad A/B.")
    parser.add_argument("--tracking-reward", choices=TRACKING_REWARDS,
                        help="Defaults to legacy-car for quad C1 and original otherwise.")
    parser.add_argument("--luo-position-scale", type=float, default=LUO_POSITION_SCALE_M,
                        metavar="METRES", help="Position error scale in Luo et al.'s tracking reward.")
    parser.add_argument("--horizontal-fov-penalty", action="store_true",
                        help="Include the horizontal edge penalty used during training in the recorded rewards.")
    parser.add_argument("--observation", choices=("rgb", "truth"), default="rgb")
    parser.add_argument("--observe-target-motion", action="store_true",
                        help="Include target speed and yaw rate in truth observations.")
    parser.add_argument("--feedforward", action="store_true",
                        help="Use standard PPO with an MLP policy for truth observations.")
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
    parser.add_argument("--randomized-c1", action="store_true",
                        help="Resample the pillar location and bypass side at each C1 reset.")
    parser.add_argument("--target-gap-m", type=float, choices=(2.0, GOAL), default=GOAL,
                        help="Use a 2 m following gap in smooth C1 or C2, or the usual 3 m gap.")
    parser.add_argument("--c2-route", choices=("mixed", "weave", "figure-eight"), default="mixed",
                        help="Select a C2 route type, or sample it independently each episode.")
    parser.add_argument("--smooth-c1-route", action="store_true",
                        help="Use a smooth approach, bypass and return to the main line in randomized single-pillar C1.")
    parser.add_argument("--transverse-clearance-shaping", action="store_true",
                        help="Shape progress to the observed target-side corridor in single-pillar C1.")
    parser.add_argument("--c1-pillar-count", type=int, choices=(1, 2), default=1,
                        help="Number of pillars in randomized C1.")
    args = parser.parse_args()
    if args.c1_pillar_count != 1 and not args.randomized_c1:
        parser.error("--c1-pillar-count 2 requires --randomized-c1")
    if args.randomized_c1 and (args.stage != "C1" or args.vehicle != "quad"
                               or not args.ideal_velocity or args.observation != "rgb"):
        parser.error("--randomized-c1 requires RGB ideal-velocity quadrotor C1")
    if args.randomized_c1 and tuple(args.pillar_center) != (0.0, 0.0):
        parser.error("--randomized-c1 resamples the pillar position and cannot use --pillar-center")
    if args.observe_target_motion and args.observation != "truth":
        parser.error("--observe-target-motion requires --observation truth")
    if args.feedforward and args.observation != "truth":
        parser.error("--feedforward requires --observation truth")
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
    if args.ideal_velocity and (args.vehicle != "quad" or args.stage not in IDEAL_VELOCITY_STAGES):
        parser.error("--ideal-velocity requires --vehicle quad and stage A, B, C1, or C2")
    if args.tracking_reward is None:
        use_legacy = args.stage == "C1" and args.vehicle == "quad"
        args.tracking_reward = SAFE_EFFECTIVE_TRACKING if args.free_flight_3d else (
            "legacy-car" if use_legacy else "original"
        )
    if args.free_flight_3d and (
        args.stage not in ("A", "B") or args.vehicle != "quad" or not args.ideal_velocity
        or args.observation != "rgb" or args.tracking_reward != SAFE_EFFECTIVE_TRACKING
    ):
        parser.error("--free-flight-3d requires obstacle-free A/B, RGB quad ideal velocity and safe-effective-tracking")
    if args.stage == "C2" and (
        args.vehicle != "quad" or not args.ideal_velocity
        or args.observation != "rgb" or args.tracking_reward != SAFE_EFFECTIVE_TRACKING
    ):
        parser.error("C2 requires RGB ideal-velocity quadrotors and --tracking-reward safe-effective-tracking")
    if args.target_gap_m != GOAL and (
        not (args.smooth_c1_route or args.stage == "C2" or args.free_flight_3d)
        or args.tracking_reward != SAFE_EFFECTIVE_TRACKING
    ):
        parser.error("--target-gap-m 2 requires smooth C1, C2 or --free-flight-3d and safe-effective-tracking")
    if args.smooth_c1_route and (not args.randomized_c1 or args.c1_pillar_count != 1):
        parser.error("--smooth-c1-route requires randomized single-pillar C1")
    if args.c2_route != "mixed" and args.stage != "C2":
        parser.error("--c2-route applies only to C2")
    if args.transverse_clearance_shaping and (
        not (args.stage == "C2" or (args.randomized_c1 and args.c1_pillar_count == 1))
        or args.tracking_reward != SAFE_EFFECTIVE_TRACKING
    ):
        parser.error("--transverse-clearance-shaping requires randomized single-pillar C1 or C2 and safe-effective-tracking")
    if args.tracking_reward == SAFE_EFFECTIVE_TRACKING and (
        args.clearance_penalty or args.clearance_threshold is not None
        or args.max_clearance_penalty is not None or args.horizontal_fov_penalty
    ):
        parser.error("safe-effective-tracking defines its own clearance and viewing terms")
    args.clearance_penalty, args.clearance_threshold, args.max_clearance_penalty = clearance_settings(
        args.stage, args.tracking_reward, args.clearance_penalty,
        args.clearance_threshold, args.max_clearance_penalty,
        remove_pillar=args.remove_pillar,
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
