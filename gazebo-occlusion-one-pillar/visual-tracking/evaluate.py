import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
import torch

from tracking_env import GOAL, TOLERANCE, GazeboTracking


def evaluate(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = GazeboTracking(args.world, output / "sim", args.stage, record=args.video, rtf=args.rtf)
    video = None
    try:
        policy = None if args.constant_speed is not None else RecurrentPPO.load(args.model, device="cpu")
        if args.video:
            from replay import Video
            video = Video(output / "tracking.mp4", args.stage)
        trace, episodes = [], []
        for number in range(args.episodes):
            seed = args.first_seed + number
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
                    action = np.array([args.constant_speed / 0.75 - 1, 0], dtype=np.float32)
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
                collision=bool(info["collision"]), ended_early=bool(terminated),
                occlusion_events=count_lost, reacquired=len(time_to_visible),
                recovery_times_s=time_to_visible, gap_recovery_times_s=time_to_gap,
            ))
        with (output / "trace.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=trace[0])
            writer.writeheader()
            writer.writerows(trace)
        report = dict(
            stage=args.stage, model=str(args.model), episodes=len(episodes), target_gap_m=GOAL,
            blank_rgb=args.blank_rgb,
            constant_speed_mps=args.constant_speed,
            gap_band_m=[GOAL - TOLERANCE, GOAL + TOLERANCE],
            valid_rate=float(np.mean([row["valid"] for row in episodes])),
            visible_rate=float(np.mean([row["visible"] for row in episodes])),
            mean_gap_error_m=float(np.mean([row["gap_mae"] for row in episodes])),
            collisions=sum(row["collision"] for row in episodes),
            early_ends=sum(row["ended_early"] for row in episodes),
            results=episodes,
        )
        report["passed"] = report["valid_rate"] >= 0.8 and report["collisions"] == 0
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
    parser.add_argument("--stage", choices=("A", "B", "C"), default="A")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=20000)
    parser.add_argument("--rtf", type=float, default=12)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--blank-rgb", action="store_true", help="Check policy behavior with camera pixels masked.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.constant_speed is not None and not 0 <= args.constant_speed <= 1.5:
        parser.error("--constant-speed must be within [0, 1.5]")
    if args.blank_rgb and not args.model:
        parser.error("--blank-rgb requires --model")
    evaluate(args)
