import argparse
from functools import partial
import json
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

from tracking_env import GazeboTracking


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
        self.episodes = []

    def _on_training_start(self):
        self.start_step = self.num_timesteps

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes.append(info["episode"]["r"])
        relative_steps = self.num_timesteps - self.start_step
        if relative_steps - self.last_report >= 512 and self.episodes:
            self.last_report = relative_steps
            step = self.num_timesteps
            print(f"steps={step} mean_return_20={np.mean(self.episodes[-20:]):.2f} "
                  f"steps_per_second={(step - self.start_step)/(monotonic()-self.start_time):.1f}", flush=True)
        return True

    def _on_rollout_end(self):
        if (self.num_timesteps - self.start_step) % 2048 == 0:
            path = self.directory / f"checkpoint-{self.num_timesteps:07}.zip"
            self.model.save(path)


def make_env(args, directory, index):
    instance = directory / f"sim-{index}"
    return Monitor(GazeboTracking(args.world, instance, args.stage, rtf=args.rtf),
                   str(instance / "episodes"))


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
    config.update(n_steps=args.rollout_steps, batch_size=64, n_epochs=5, learning_rate=0.0003, gamma=0.99,
                  gae_lambda=0.95, clip_range=0.2, ent_coef=0.005, target_kl=0.035,
                  lstm_hidden_size=96, control_hz=6, gap_metres=3.0)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    factories = [partial(make_env, args, directory, index) for index in range(args.environments)]
    env = DummyVecEnv(factories) if args.environments == 1 else SubprocVecEnv(factories, start_method="spawn")
    env.seed(args.seed)
    try:
        if args.resume:
            model = RecurrentPPO.load(args.resume, env=env, device=args.device,
                                     n_steps=args.rollout_steps,
                                     tensorboard_log=str(directory / "tensorboard"))
        else:
            model = RecurrentPPO(
                "CnnLstmPolicy", env, n_steps=args.rollout_steps, batch_size=64, n_epochs=5, learning_rate=0.0003,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005, target_kl=0.035,
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
    parser.add_argument("--stage", choices=("A", "B", "C"), default="A")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=12288)
    parser.add_argument("--rtf", type=float, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--environments", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.environments < 1:
        parser.error("--environments must be positive")
    if args.rollout_steps < 64 or args.rollout_steps % 64:
        parser.error("--rollout-steps must be a multiple of 64")
    train(args)
