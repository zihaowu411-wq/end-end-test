import argparse
import math
from pathlib import Path
from time import monotonic

import numpy as np
from gymnasium.envs.registration import EnvSpec
from stable_baselines3.common.env_checker import check_env

from tracking_env import DT, GazeboTracking, Pose


def smoke(args):
    env = GazeboTracking(args.world, args.output, rtf=args.rtf)
    try:
        # The contract checker can exercise the API. Absolute simulator
        # timestamps continue across episode resets.
        if not args.fast:
            env.spec = EnvSpec("GazeboTracking-v0", nondeterministic=True)
            check_env(env, warn=True, skip_render_check=True)
        frame, initial = env.reset(seed=2)
        assert frame.shape == (3, 72, 128) and frame.dtype == np.uint8
        stamp = initial["sim_stamp_ns"]
        x0 = initial["x"]
        start = monotonic()
        for step in range(60):
            _, reward, episode_done, truncated, info = env.step(np.array([0.25, 0.0], dtype=np.float32))
            assert math.isfinite(reward) and not episode_done and not truncated
            assert math.isclose((info["sim_stamp_ns"] - stamp) / 1e9, DT, abs_tol=1e-6), info
            assert 0 <= info["sim_stamp_ns"] - info["image_stamp_ns"] <= 86_000_000, info
            stamp = info["sim_stamp_ns"]
        print(f"Fresh RGB, fixed step, vehicle moved {info['x'] - x0:.2f} m", flush=True)
        assert info["x"] - x0 > 5.0
        print(f"Throughput {60 / (monotonic() - start):.1f} environment steps/s", flush=True)
        _, reset = env.reset(seed=2)
        assert abs(reset["distance"] - initial["distance"]) < 0.02
        env._teleport(Pose(-3.0, 0), Pose(-1.4, 0))
        env.advance()
        env.target_speeds[:] = 0.0
        for _ in range(12):
            _, _, terminated, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            if terminated:
                break
        assert terminated and info["collision"], info
        print("Reset and chassis collision termination passed", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=Path, default=Path("../one_pillar.sdf"))
    parser.add_argument("--output", type=Path, default=Path("checks/stage-a"))
    parser.add_argument("--rtf", type=float, default=8)
    parser.add_argument("--fast", action="store_true")
    smoke(parser.parse_args())
