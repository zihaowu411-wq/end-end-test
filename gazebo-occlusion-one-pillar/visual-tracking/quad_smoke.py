import argparse
import json
import math
from pathlib import Path

from PIL import Image

from quad_tracking_env import ALTITUDE_M, PILLAR_HEIGHT_M, QuadTracking
from replay import Video
from tracking_env import CONTROL_HZ, angle_to


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    side = -1 if args.seed % 2 else 1
    env = QuadTracking(args.world, output / "sim", "C1", rtf=12, record=True,
                       pillar_center=(side * 0.18, -2.0))
    video = None
    try:
        video = Video(output / "flight-check.mp4", "Quadrotor control check")
        obs, initial = env.reset(seed=args.seed)
        assert obs.shape == (3, 72, 128)
        assert initial["visible"], initial
        assert abs(initial["z"] - ALTITUDE_M) < 0.12
        assert abs(initial["target_z"] - ALTITUDE_M) < 0.12
        Image.fromarray(env.front_frame).resize((768, 432)).save(output / "front.png")
        Image.fromarray(env.overhead_frame).save(output / "overhead.png")
        video.add(env, initial, args.seed, [])
        records = []
        for i in range(env.duration):
            with env.condition:
                _, pursuer, target = env.poses[-1]
            dx = side * 1.44 - pursuer.x
            lateral = -side * min(0.85, max(-0.5, 1.4 * side * dx))
            distance = math.hypot(target.x - pursuer.x, target.y - pursuer.y)
            speed = min(1.1, max(0.50, 0.76 + 0.7 * (distance - 3)))
            yaw_rate = 0.0 if i < CONTROL_HZ else max(
                -0.55, min(0.55, 1.2 * angle_to(pursuer, target.x, target.y))
            )
            _, _, terminated, truncated, info = env.step((speed / 0.75 - 1, lateral, yaw_rate / 1.5))
            records.append(info)
            video.add(env, info, args.seed, records)
            assert not terminated, info
            if truncated:
                break
        first_second = records[CONTROL_HZ - 1]
        altitude_error = max(abs(row[key] - ALTITUDE_M) for row in records for key in ("z", "target_z"))
        relative_altitude_error = max(abs(row["z"] - row["target_z"]) for row in records)
        lateral_displacement = side * (first_second["x"] - initial["x"])
        yaw_change = abs(first_second["yaw"] - initial["yaw"])
        report = {
            "seed": args.seed,
            "nominal_altitude_m": ALTITUDE_M,
            "pillar_height_m": PILLAR_HEIGHT_M,
            "max_altitude_error_m": altitude_error,
            "max_height_difference_m": relative_altitude_error,
            "first_second_sideways_displacement_m": lateral_displacement,
            "first_second_yaw_change_deg": math.degrees(yaw_change),
            "min_pillar_clearance_m": min(row["pillar_clearance_m"] for row in records),
            "passed_pillar": bool(records[-1]["past_pillar"]),
            "final_distance_m": records[-1]["distance"],
            "control": "scripted control check",
        }
        (output / "flight-check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        assert altitude_error < 0.15 and relative_altitude_error < 0.18, report
        assert lateral_displacement > 0.16 and yaw_change < math.radians(6), report
        assert report["passed_pillar"], report
    finally:
        if video:
            video.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=Path, default=Path("../one_pillar.sdf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=27600)
    run(parser.parse_args())
