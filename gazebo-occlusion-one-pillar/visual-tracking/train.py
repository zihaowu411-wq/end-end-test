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
    APPROACH_WEIGHT, CAMERA_HZ, CENTER_WEIGHT,
    COLLISION_PENALTY, CONTROL_HZ, DISTANCE_WEIGHT, FOV_MAX_PENALTY, FOV_PENALTY_START_DEG,
    GAMMA, GAP_SCALE, GOAL, HEIGHT, HORIZONTAL_FOV_HALF_DEG, IDEAL_VELOCITY_STAGES,
    IN_BAND_BONUS, LEGACY_CAR_ACTION_CHANGE_WEIGHT, LEGACY_CAR_DISTANCE_SCALE_M,
    LEGACY_CAR_LOST_REWARD, LEGACY_CAR_WEIGHTS,
    LOST_REWARD, LUO_HEADING_WEIGHT, LUO_POSITION_SCALE_M, LUO_REWARD_BASE,
    PILLAR_STAGES, STAGES, TOLERANCE, TRACKING_REWARDS,
    VALID_PROGRESS_BEARING_WEIGHT, VALID_PROGRESS_CENTER_PENALTY_WEIGHT,
    VALID_PROGRESS_DISTANCE_REWARD_SCALE_M, VALID_PROGRESS_DISTANCE_REWARD_WEIGHT,
    VALID_PROGRESS_GAP_SCALE_M, VALID_PROGRESS_GAP_WEIGHT,
    VALID_PROGRESS_IN_BAND_REWARD, VALID_PROGRESS_LOST_REWARD, VALID_PROGRESS_VISIBLE_REWARD,
    VISIBLE_GAP_BEARING_WEIGHT, VISIBLE_GAP_DISTANCE_WEIGHT, VISIBLE_GAP_LOST_REWARD, WIDTH, GazeboTracking,
    ARENA_HALF_SIZE, arena_walls_enabled, clearance_settings,
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


class FollowingState(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=96):
        super().__init__(observation_space, features_dim)
        self.network = nn.Sequential(nn.Linear(observation_space.shape[0], features_dim), nn.ReLU())

    def forward(self, observation):
        return self.network(observation)


class Progress(BaseCallback):
    def __init__(self, directory, *, checkpoint_after_update=False):
        super().__init__()
        self.directory = directory
        self.checkpoint_after_update = checkpoint_after_update
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
                duration = info.get("b1_planned_steps", self.planned_steps[index])
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

    def _save_checkpoint(self):
        relative_steps = self.num_timesteps - self.start_step
        if relative_steps > 0 and relative_steps % 2048 == 0:
            path = self.directory / f"checkpoint-{self.num_timesteps:07}.zip"
            self.model.save(path)

    def _on_rollout_start(self):
        if self.checkpoint_after_update:
            self._save_checkpoint()

    def _on_rollout_end(self):
        if not self.checkpoint_after_update:
            self._save_checkpoint()


def worker_stage(args, index):
    b1_workers = getattr(args, "c1_b1_environments", 0)
    if b1_workers and index >= args.environments - b1_workers:
        return "B1"
    return args.stage


def make_env(args, directory, index):
    stage = worker_stage(args, index)
    instance = directory / f"sim-{index}"
    side = 1 if index % 2 == 0 else -1
    pillar_x, pillar_y = (args.mirrored_pillar or (0.0, 0.0)) if stage == "C1" else (0.0, 0.0)
    info_keywords = ()
    if stage == "C1":
        info_keywords = ("collision", "past_pillar")
    elif stage == "B1":
        info_keywords = ("b1_route", "b_first_turn", "collision", "b1_route_complete", "b1_planned_steps")
    elif args.b_weave:
        info_keywords = ("b_first_turn",)
    environment = GazeboTracking
    if args.vehicle == "quad":
        from quad_tracking_env import IdealQuadTracking, QuadTracking, TruthQuadTracking
        quad_environment = IdealQuadTracking if getattr(args, "ideal_velocity", False) else QuadTracking
        if args.observation == "truth":
            quad_environment = TruthQuadTracking
        environment = partial(quad_environment, cube_target=getattr(args, "cube_target", False),
                              target_visual_scale=getattr(args, "target_visual_scale", 1.0))
    if stage == "B1":
        from b1_tracking_env import B1_ROUTES, B1Tracking
        count = args.environments if args.stage == "B1" else args.c1_b1_environments
        b1_index = index - (args.environments - count)
        focus = args.b1_focus_route
        if b1_index < len(focus):
            routes = (focus[b1_index],)
        else:
            background = tuple(route for route in B1_ROUTES if route not in focus) or B1_ROUTES
            worker = b1_index - len(focus)
            background_workers = count - len(focus)
            routes = background[worker::background_workers] or (background[worker % len(background)],)
        environment = partial(B1Tracking, routes=routes, mirror_episodes=True, cube_target=args.cube_target,
                              reverse_speed_mps=args.b1_reverse_speed,
                              target_visual_scale=args.target_visual_scale)
    return Monitor(environment(args.world, instance, stage, rtf=args.rtf,
                                  penalize_clearance=args.clearance_penalty,
                                  clearance_threshold=args.clearance_threshold,
                                  max_clearance_penalty=args.max_clearance_penalty,
                                  tracking_reward=args.tracking_reward,
                                  luo_position_scale_m=args.luo_position_scale,
                                  penalize_horizontal_fov=args.horizontal_fov_penalty,
                                  camera_resolution=args.camera_resolution,
                                  pillar_center=(side * pillar_x, pillar_y),
                                  fixed_route_side=side if args.mirrored_pillar or args.b_weave or stage == "B1" else None,
                                  c1_target_detour=args.c1_target_detour and stage == "C1",
                                  b_weave=args.b_weave),
                   str(instance / "episodes"),
                   info_keywords=info_keywords)


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
    arena_walls = arena_walls_enabled(args.stage, args.ideal_velocity)
    if args.c1_target_detour or arena_walls:
        config["reward"]["clearance_curve"] = "quadratic-two-zone"
    if arena_walls:
        config["arena_inner_half_size_m"] = ARENA_HALF_SIZE
    legacy_car_weights = LEGACY_CAR_WEIGHTS.get(args.tracking_reward)
    if legacy_car_weights is not None:
        distance_weight, center_weight = legacy_car_weights
        del config["reward"]["in_band_bonus"]
        del config["reward"]["approach_weight"]
        config["reward"].update(distance_weight=distance_weight,
                                center_weight=center_weight,
                                gap_scale=LEGACY_CAR_DISTANCE_SCALE_M,
                                bearing_scale_deg=HORIZONTAL_FOV_HALF_DEG,
                                lost=LEGACY_CAR_LOST_REWARD,
                                action_change_weight=LEGACY_CAR_ACTION_CHANGE_WEIGHT,
                                step_scale=6 / CONTROL_HZ)
    elif args.tracking_reward == "visible-gap":
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
    if args.c1_b1_environments:
        config["parallel_stages"] = [worker_stage(args, i) for i in range(args.environments)]
    if args.stage == "B1" or args.c1_b1_environments:
        from b1_tracking_env import B1_LOOP_RADIUS_M, B1_ROUTES
        config.update(b1_routes=B1_ROUTES, b1_loop_radius_m=B1_LOOP_RADIUS_M)
    if args.observation == "truth":
        from quad_tracking_env import TruthQuadTracking
        config["observation_fields"] = TruthQuadTracking.fields
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
                ("MlpLstmPolicy" if args.observation == "truth" else "CnnLstmPolicy"), env, n_steps=args.rollout_steps, batch_size=64, n_epochs=5,
                learning_rate=args.learning_rate,
                gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=0.2, ent_coef=0.005, target_kl=0.035,
                policy_kwargs=dict(features_extractor_class=(FollowingState if args.observation == "truth" else FollowingCNN),
                                   lstm_hidden_size=96,
                                   net_arch=dict(pi=[64], vf=[64]), log_std_init=-1.15),
                verbose=1, device=args.device, seed=args.seed,
                tensorboard_log=str(directory / "tensorboard"),
            )
        progress = Progress(directory, checkpoint_after_update=bool(args.c1_b1_environments))
        model.learn(args.steps, callback=progress, reset_num_timesteps=not args.resume)
        model.save(directory / "final.zip")
        (directory / "completed").touch()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=Path, default=Path("../one_pillar.sdf"))
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
                        help="Defaults to legacy-car for quad C1 or ideal-velocity mode.")
    parser.add_argument("--luo-position-scale", type=float, default=LUO_POSITION_SCALE_M,
                        metavar="METRES", help="Position error scale in Luo et al.'s tracking reward.")
    parser.add_argument("--horizontal-fov-penalty", action="store_true",
                        help="Penalize the target approaching or crossing the horizontal edge of the camera.")
    parser.add_argument("--camera-resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        default=(WIDTH, HEIGHT), help="Onboard RGB image size in pixels.")
    parser.add_argument("--observation", choices=("rgb", "truth"), default="rgb")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=12288)
    parser.add_argument("--rtf", type=float, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--environments", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--clearance-penalty", action="store_true")
    parser.add_argument("--clearance-threshold", type=float)
    parser.add_argument("--max-clearance-penalty", type=float)
    parser.add_argument("--mirrored-pillar", nargs=2, type=float, metavar=("ABS_X", "Y"),
                        help="Pair each C1 route side with a mirrored pillar center.")
    parser.add_argument("--c1-b1-environments", type=int, default=0,
                        help="Use this many B1 workers alongside the mirrored C1 workers.")
    parser.add_argument("--c1-target-detour", action="store_true",
                        help="Use the C1 target detour and its two-zone clearance penalty.")
    parser.add_argument("--b-weave", action="store_true",
                        help="Use the two-direction target trajectory in Stage B.")
    from b1_tracking_env import B1_ROUTES
    parser.add_argument("--b1-focus-route", action="append", choices=B1_ROUTES, default=[],
                        help="Reserve one B1 worker per occurrence for the selected route.")
    parser.add_argument("--b1-reverse-speed", type=float, default=0.0, metavar="MPS",
                        help="Maximum backward speed for the low end of the Stage B1 forward action.")
    args = parser.parse_args()
    c1_workers = args.environments - args.c1_b1_environments
    if args.c1_b1_environments < 0 or (args.c1_b1_environments and c1_workers < 2):
        parser.error("--c1-b1-environments must leave at least two C1 workers")
    if args.c1_b1_environments and (args.stage != "C1" or args.vehicle != "quad" or not args.ideal_velocity
                                    or args.observation != "rgb" or not args.mirrored_pillar or c1_workers % 2):
        parser.error("Mixed B1 workers require ideal RGB quadrotors and an even number of mirrored C1 workers")
    if not math.isfinite(args.b1_reverse_speed) or args.b1_reverse_speed < 0:
        parser.error("--b1-reverse-speed must be finite and nonnegative")
    if args.b1_reverse_speed and args.stage != "B1":
        parser.error("--b1-reverse-speed requires Stage B1")
    if args.b1_focus_route and (args.stage != "B1" or len(args.b1_focus_route) >= args.environments):
        parser.error("--b1-focus-route requires Stage B1 and a worker for the other routes")
    if args.stage == "B1" and (args.vehicle != "quad" or not args.ideal_velocity or args.observation != "rgb"):
        parser.error("Stage B1 requires an ideal-velocity quadrotor with RGB observations")
    if args.b_weave and (args.stage != "B" or args.environments % 2):
        parser.error("--b-weave requires Stage B and an even environment count")
    if args.c1_target_detour and (args.stage != "C1" or not args.ideal_velocity
                                  or not args.mirrored_pillar or c1_workers % 2):
        parser.error("The target detour requires ideal C1 and even mirror-paired environments")
    if args.cube_target and args.vehicle != "quad":
        parser.error("--cube-target requires --vehicle quad")
    if not math.isfinite(args.target_visual_scale) or args.target_visual_scale <= 0:
        parser.error("--target-visual-scale must be finite and positive")
    if args.target_visual_scale != 1.0 and (args.vehicle != "quad" or args.cube_target):
        parser.error("--target-visual-scale requires an X3 quadrotor target")
    if args.ideal_velocity and (args.vehicle != "quad" or args.stage not in (*IDEAL_VELOCITY_STAGES, "B1")):
        parser.error("--ideal-velocity requires --vehicle quad and --stage A, B, or C1")
    if args.observation == "truth" and not args.ideal_velocity:
        parser.error("--observation truth requires --ideal-velocity")
    if args.tracking_reward is None:
        use_legacy = args.ideal_velocity or (args.stage == "C1" and args.vehicle == "quad")
        args.tracking_reward = "legacy-car" if use_legacy else "original"
    args.clearance_penalty, args.clearance_threshold, args.max_clearance_penalty = clearance_settings(
        args.stage, args.tracking_reward, args.clearance_penalty,
        args.clearance_threshold, args.max_clearance_penalty,
        c1_target_detour=args.c1_target_detour, arena_walls=arena_walls_enabled(args.stage, args.ideal_velocity),
    )
    if args.tracking_reward == "visible-gap" and args.horizontal_fov_penalty:
        parser.error("visible-gap already includes a horizontal bearing penalty")
    if not math.isfinite(args.luo_position_scale) or args.luo_position_scale <= 0:
        parser.error("--luo-position-scale must be finite and positive")
    if min(args.camera_resolution) < 1:
        parser.error("--camera-resolution dimensions must be positive")
    if args.environments < 1:
        parser.error("--environments must be positive")
    if args.clearance_penalty and args.stage not in (*PILLAR_STAGES, "B1"):
        parser.error("--clearance-penalty requires a stage with a pillar or walls")
    if not math.isfinite(args.clearance_threshold) or args.clearance_threshold <= 0:
        parser.error("--clearance-threshold must be finite and positive")
    if not math.isfinite(args.max_clearance_penalty) or args.max_clearance_penalty < 0:
        parser.error("--max-clearance-penalty must be finite and nonnegative")
    if args.mirrored_pillar and (args.stage != "C1" or c1_workers % 2
                                 or not all(map(math.isfinite, args.mirrored_pillar))
                                 or args.mirrored_pillar[0] < 0):
        parser.error("--mirrored-pillar requires finite coordinates, nonnegative ABS_X and an even number of C1 workers")
    if args.rollout_steps < 64 or args.rollout_steps % 64:
        parser.error("--rollout-steps must be a multiple of 64")
    train(args)
