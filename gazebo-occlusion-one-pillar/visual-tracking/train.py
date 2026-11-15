import argparse
from collections import deque
from functools import partial
import json
import math
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import nn

from tracking_env import (
    APPROACH_WEIGHT, CAMERA_HZ, CENTER_WEIGHT, CLEARANCE_THRESHOLD_M,
    COLLISION_PENALTY, CONTROL_HZ, DISTANCE_WEIGHT, FOV_MAX_PENALTY, FOV_PENALTY_START_DEG,
    GAMMA, GAP_SCALE, GOAL, HORIZONTAL_FOV_HALF_DEG,
    IN_BAND_BONUS, LOST_REWARD, LUO_HEADING_WEIGHT, LUO_POSITION_SCALE_M, LUO_REWARD_BASE,
    MAX_CLEARANCE_PENALTY, PILLAR_STAGES, STAGES, TOLERANCE, TRACKING_REWARDS,
    VALID_PROGRESS_BEARING_WEIGHT, VALID_PROGRESS_CENTER_PENALTY_WEIGHT,
    VALID_PROGRESS_DISTANCE_REWARD_SCALE_M, VALID_PROGRESS_DISTANCE_REWARD_WEIGHT,
    VALID_PROGRESS_GAP_SCALE_M, VALID_PROGRESS_GAP_WEIGHT,
    VALID_PROGRESS_IN_BAND_REWARD, VALID_PROGRESS_LOST_REWARD, VALID_PROGRESS_VISIBLE_REWARD,
    VISIBLE_GAP_BEARING_WEIGHT, VISIBLE_GAP_DISTANCE_WEIGHT, VISIBLE_GAP_LOST_REWARD, GazeboTracking,
)


GAE_LAMBDA = 0.95 ** (6 / CONTROL_HZ)


class FollowingCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=96):
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(), nn.Linear(32 * 4 * 8, features_dim), nn.ReLU(),
        )

    def forward(self, observation):
        return self.network(observation)


class Progress(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.start_time = monotonic()
        self.start_step = None
        self.last_report = 0
        self.episodes = deque(maxlen=20)

    def _on_training_start(self):
        self.start_step = self.num_timesteps
        self.planned_steps = np.asarray(self.training_env.get_attr("duration"))
        self.valid_steps = np.zeros(self.training_env.num_envs, dtype=int)
        self.visible_steps = np.zeros_like(self.valid_steps)

    def _on_step(self):
        for index, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            self.valid_steps[index] += info["valid"]
            self.visible_steps[index] += info["visible"]
            if done:
                duration = self.planned_steps[index]
                valid_rate = self.valid_steps[index] / duration
                self.episodes.append({
                    "return": info["episode"]["r"],
                    "valid_rate": valid_rate,
                    "visible_rate": self.visible_steps[index] / duration,
                    "collision": info["collision"],
                    "passed": info["episode"]["l"] == duration and valid_rate >= 0.8
                              and not info["collision"],
                })
                self.valid_steps[index] = 0
                self.visible_steps[index] = 0
        relative_steps = self.num_timesteps - self.start_step
        if relative_steps - self.last_report >= 512 and self.episodes:
            self.last_report = relative_steps
            step = self.num_timesteps
            summary = self.episode_summary()
            print(f"steps={step} mean_return_20={summary['mean_return']:.2f} "
                  f"training_valid_rate={summary['valid_rate']:.1%} "
                  f"training_visible_rate={summary['visible_rate']:.1%} "
                  f"training_collisions={summary['collisions']}/{summary['episodes']} "
                  f"training_passed_episodes={summary['passed_episodes']}/{summary['episodes']} "
                  f"steps_per_second={(step - self.start_step)/(monotonic()-self.start_time):.1f}", flush=True)
        return True

    def episode_summary(self):
        return {
            "episodes": len(self.episodes),
            "mean_return": float(np.mean([episode["return"] for episode in self.episodes])),
            "valid_rate": float(np.mean([episode["valid_rate"] for episode in self.episodes])),
            "visible_rate": float(np.mean([episode["visible_rate"] for episode in self.episodes])),
            "collisions": sum(bool(episode["collision"]) for episode in self.episodes),
            "passed_episodes": sum(bool(episode["passed"]) for episode in self.episodes),
        }

    def _on_training_end(self):
        if self.episodes:
            path = self.directory / "training-summary.json"
            report = {"steps": self.num_timesteps, **self.episode_summary()}
            path.write_text(json.dumps(report, indent=2) + "\n")

    def _on_rollout_end(self):
        if (self.num_timesteps - self.start_step) % 2048 == 0:
            path = self.directory / f"checkpoint-{self.num_timesteps:07}.zip"
            self.model.save(path)


def make_env(args, directory, index):
    instance = directory / f"sim-{index}"
    side = 1 if index % 2 == 0 else -1
    pillar_x, pillar_y = args.mirrored_pillar or (0.0, 0.0)
    environment = GazeboTracking
    if args.vehicle == "quad":
        from quad_tracking_env import IdealQuadTracking, QuadTracking
        environment = IdealQuadTracking if getattr(args, "ideal_velocity", False) else QuadTracking
    return Monitor(environment(args.world, instance, args.stage, rtf=args.rtf,
                                  penalize_clearance=args.clearance_penalty,
                                  clearance_threshold=args.clearance_threshold,
                                  max_clearance_penalty=args.max_clearance_penalty,
                                  tracking_reward=args.tracking_reward,
                                  luo_position_scale_m=args.luo_position_scale,
                                  penalize_horizontal_fov=args.horizontal_fov_penalty,
                                  pillar_center=(side * pillar_x, pillar_y),
                                  fixed_route_side=side if args.mirrored_pillar else None),
                   str(instance / "episodes"),
                   info_keywords=("collision", "past_pillar") if args.stage == "C1" else ())


def train(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Activate the CUDA environment or explicitly choose --device cpu.")
    if args.device == "cuda":
        print(f"Training on {torch.cuda.get_device_name()}", flush=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(n_steps=args.rollout_steps, batch_size=64, n_epochs=5, learning_rate=args.learning_rate, gamma=GAMMA,
                  gae_lambda=GAE_LAMBDA, clip_range=0.2, ent_coef=0.005, target_kl=0.035,
                  lstm_hidden_size=96, control_hz=CONTROL_HZ, camera_hz=CAMERA_HZ, gap_metres=GOAL,
                  reward=dict(distance_weight=DISTANCE_WEIGHT, center_weight=CENTER_WEIGHT,
                              gap_scale=GAP_SCALE, in_band_bonus=IN_BAND_BONUS,
                              approach_weight=APPROACH_WEIGHT, lost=LOST_REWARD,
                              collision=-COLLISION_PENALTY, clearance_threshold_m=args.clearance_threshold,
                              max_clearance_penalty=args.max_clearance_penalty if args.clearance_penalty else 0.0))
    if args.tracking_reward == "visible-gap":
        for term in ("distance_weight", "center_weight", "gap_scale", "in_band_bonus", "approach_weight", "lost"):
            del config["reward"][term]
        config["reward"].update(distance_weight=VISIBLE_GAP_DISTANCE_WEIGHT, gap_scale_m=TOLERANCE,
                                bearing_weight=VISIBLE_GAP_BEARING_WEIGHT, bearing_scale_deg=HORIZONTAL_FOV_HALF_DEG,
                                lost=VISIBLE_GAP_LOST_REWARD)
    elif args.tracking_reward == "valid-progress":
        for term in ("distance_weight", "center_weight", "gap_scale", "in_band_bonus", "approach_weight", "lost"):
            del config["reward"][term]
        config["reward"].update(in_band=VALID_PROGRESS_IN_BAND_REWARD,
                                visible_out_of_band=VALID_PROGRESS_VISIBLE_REWARD,
                                lost=VALID_PROGRESS_LOST_REWARD,
                                tolerance_m=TOLERANCE,
                                center_penalty_weight=VALID_PROGRESS_CENTER_PENALTY_WEIGHT,
                                center_penalty_full_deg=HORIZONTAL_FOV_HALF_DEG,
                                distance_reward_weight=VALID_PROGRESS_DISTANCE_REWARD_WEIGHT,
                                distance_reward_scale_m=VALID_PROGRESS_DISTANCE_REWARD_SCALE_M,
                                potential_gap_weight=VALID_PROGRESS_GAP_WEIGHT,
                                potential_gap_scale_m=VALID_PROGRESS_GAP_SCALE_M,
                                potential_bearing_weight=VALID_PROGRESS_BEARING_WEIGHT,
                                potential_bearing_scale_deg=180.0)
    elif args.tracking_reward in ("luo", "luo-progress"):
        for term in ("distance_weight", "center_weight", "gap_scale", "in_band_bonus", "lost"):
            del config["reward"][term]
        if args.tracking_reward == "luo":
            del config["reward"]["approach_weight"]
        config["reward"].update(luo_a=LUO_REWARD_BASE, luo_c_m=args.luo_position_scale,
                                luo_lambda=LUO_HEADING_WEIGHT)
    if args.horizontal_fov_penalty:
        config["reward"].update(fov_penalty_start_deg=FOV_PENALTY_START_DEG,
                                fov_penalty_full_deg=HORIZONTAL_FOV_HALF_DEG,
                                fov_max_penalty=FOV_MAX_PENALTY)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    factories = [partial(make_env, args, directory, index) for index in range(args.environments)]
    env = DummyVecEnv(factories) if args.environments == 1 else SubprocVecEnv(factories, start_method="spawn")
    env.seed(args.seed)
    try:
        if args.resume:
            model = RecurrentPPO.load(args.resume, env=env, device=args.device,
                                     n_steps=args.rollout_steps, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                                     seed=args.seed, learning_rate=args.learning_rate,
                                     tensorboard_log=str(directory / "tensorboard"))
        else:
            model = RecurrentPPO(
                "CnnLstmPolicy", env, n_steps=args.rollout_steps, batch_size=64, n_epochs=5,
                learning_rate=args.learning_rate,
                gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=0.2, ent_coef=0.005, target_kl=0.035,
                policy_kwargs=dict(features_extractor_class=FollowingCNN, lstm_hidden_size=96,
                                   net_arch=dict(pi=[64], vf=[64]), log_std_init=-1.15),
                verbose=1, device=args.device, seed=args.seed,
                tensorboard_log=str(directory / "tensorboard"),
            )
        model.learn(args.steps, callback=Progress(directory), reset_num_timesteps=not args.resume)
        model.save(directory / "final.zip")
        (directory / "completed").touch()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=Path, default=Path("../one_pillar.sdf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="A")
    parser.add_argument("--vehicle", choices=("car", "quad"), default="car")
    parser.add_argument("--ideal-velocity", action="store_true",
                        help="Integrate speed commands directly for quadrotor stage A.")
    parser.add_argument("--tracking-reward", choices=TRACKING_REWARDS, default="original")
    parser.add_argument("--luo-position-scale", type=float, default=LUO_POSITION_SCALE_M,
                        metavar="METRES", help="Position error scale in Luo et al.'s tracking reward.")
    parser.add_argument("--horizontal-fov-penalty", action="store_true",
                        help="Penalize the target approaching or crossing the horizontal edge of the camera.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=12288)
    parser.add_argument("--rtf", type=float, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--environments", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--clearance-penalty", action="store_true")
    parser.add_argument("--clearance-threshold", type=float, default=CLEARANCE_THRESHOLD_M)
    parser.add_argument("--max-clearance-penalty", type=float, default=MAX_CLEARANCE_PENALTY)
    parser.add_argument("--mirrored-pillar", nargs=2, type=float, metavar=("ABS_X", "Y"),
                        help="Pair each C1 route side with a mirrored pillar center.")
    args = parser.parse_args()
    if args.ideal_velocity and (args.vehicle != "quad" or args.stage != "A"):
        parser.error("--ideal-velocity requires --vehicle quad and --stage A")
    if args.tracking_reward == "visible-gap" and args.horizontal_fov_penalty:
        parser.error("visible-gap already includes a horizontal bearing penalty")
    if not math.isfinite(args.luo_position_scale) or args.luo_position_scale <= 0:
        parser.error("--luo-position-scale must be finite and positive")
    if args.environments < 1:
        parser.error("--environments must be positive")
    if args.clearance_penalty and args.stage not in PILLAR_STAGES:
        parser.error("--clearance-penalty requires a stage with a pillar")
    if not math.isfinite(args.clearance_threshold) or args.clearance_threshold <= 0:
        parser.error("--clearance-threshold must be finite and positive")
    if not math.isfinite(args.max_clearance_penalty) or args.max_clearance_penalty < 0:
        parser.error("--max-clearance-penalty must be finite and nonnegative")
    if args.mirrored_pillar and (args.stage != "C1" or args.environments % 2
                                 or not all(map(math.isfinite, args.mirrored_pillar))
                                 or args.mirrored_pillar[0] < 0):
        parser.error("--mirrored-pillar requires finite coordinates, nonnegative ABS_X and an even number of C1 workers")
    if args.rollout_steps < 64 or args.rollout_steps % 64:
        parser.error("--rollout-steps must be a multiple of 64")
    train(args)
