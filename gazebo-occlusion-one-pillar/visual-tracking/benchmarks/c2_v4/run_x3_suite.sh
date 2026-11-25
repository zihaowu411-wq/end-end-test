#!/usr/bin/env bash
set -Eeuo pipefail
if (( $# != 3 )); then
    echo "Usage: bash run_x3_suite.sh SCENES_JSON C2_FINAL_ZIP OUTPUT_DIR" >&2
    exit 2
fi
bench=$(cd -- "$(dirname -- "$0")" && pwd)
code=$(cd -- "$bench/../.." && pwd)
scenes=$(realpath -e -- "$1")
model=$(realpath -e -- "$2")
mkdir -- "$3"
root=$(cd -- "$3" && pwd)
py="${GAZEBO_PYTHON:-python}"
cd "$code"
export PYTHONPATH="$code${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONDONTWRITEBYTECODE=1
common=(--scenes "$scenes" --world "$code/../one_pillar.sdf" --target x3)
mark() { printf '%s\n' "$*" > "$root/status"; }
trap 'r=$?; mark "error exit=$r line=$LINENO"; exit "$r"' ERR
run_group() {
    local group=$1
    shift
    local seed pid rc=0
    local -a pids=()
    for seed in 0 1 2 3; do
        timeout 480 "$py" -u "$bench/evaluate_scenes.py" "${common[@]}" --seed "$seed" --output "$root/$group-$seed" "$@" \
            > "$root/$group-$seed.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    test "$rc" -eq 0
}
mark smoke
run_group smoke --smoke-steps 20
mark evaluate
run_group eval --model "$model"
seed=$(python3 - "$root" "$scenes" <<'PY'
from pathlib import Path
from statistics import mean,median
from itertools import islice
import csv,json,sys
root=Path(sys.argv[1])
manifest=json.loads(Path(sys.argv[2]).read_text())
scenes={c["seed"]:c for c in manifest["cases"]}
assert all(s["target_speed_mps"] == scenes[0]["target_speed_mps"] and
           s["duration_s"] == scenes[0]["duration_s"] for s in scenes.values())
comparison_steps=round(100.0/manifest["dt_s"])
rows=[json.loads((root/f"eval-{seed}"/"evaluation.json").read_text()) for seed in range(4)]
assert len(set(r["model_sha256"] for r in rows))==1
center=median(r["valid_rate"] for r in rows)
chosen=min(rows,key=lambda r:(-r["survival_s"],abs(r["valid_rate"]-center),r["seed"]))
losses=sum(r["visibility_losses"] for r in rows)
times=[x for r in rows for x in r["recovery_times_s"]]
for r in rows:
    source=scenes[r["seed"]]
    assert source["fingerprint"]==r["fingerprint"] and r["mode"]=="deterministic"
    with (root/f"eval-{r['seed']}/trace.csv").open() as stream:
        first_100s=list(islice(csv.DictReader(stream),comparison_steps))
    r["valid_rate_first_100s"]=sum(s["valid"]=="True" for s in first_100s)/comparison_steps
    r["visible_rate_first_100s"]=sum(s["visible"]=="True" for s in first_100s)/comparison_steps
    stats=source["source_metrics"]
    r.update(turns_at_least_25deg=stats["turns_at_least_25deg_100s"],crossings=stats["crossing_count_100s"],
             target_cube_clearance_bound_m=source["target_cube_clearance_lower_bound_m"])
summary=dict(episodes=len(rows),horizon_s=scenes[0]["duration_s"],mode="deterministic",model_sha256=rows[0]["model_sha256"],
             target="native red X3",target_visual_scale=1.0,policy_timesteps=524288,
             target_speed_mps=scenes[0]["target_speed_mps"],initial_gap_m=2.0,effective_gap_band_m=[1.5,2.5],rgb_resolution=[256,144],
             mean_valid_rate=mean(r["valid_rate"] for r in rows),mean_visible_rate=mean(r["visible_rate"] for r in rows),
             comparison_horizon_s=100.0,mean_valid_rate_first_100s=mean(r["valid_rate_first_100s"] for r in rows),
             mean_visible_rate_first_100s=mean(r["visible_rate_first_100s"] for r in rows),
             mean_gap_mae_m=mean(r["gap_mae_m"] for r in rows),mean_survival_s=mean(r["survival_s"] for r in rows),
             collisions=sum(r["stop"]=="collision" for r in rows),out_of_bounds=sum(r["stop"]=="out_of_bounds" for r in rows),
             completed=sum(r["stop"]=="completed" for r in rows),passed=sum(r["passed"] for r in rows),
             visibility_losses=losses, wall_occlusion_losses=sum(r["initial_wall_occlusions"] for r in rows),
             reacquired=len(times),mean_recovery_s=mean(times) if times else None,
             video_seed=chosen["seed"],video_selection="longest survival, then closest to median valid rate",results=rows)
(root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(chosen["seed"])
PY
)
mark video
timeout 480 "$py" -u "$bench/evaluate_scenes.py" "${common[@]}" --seed "$seed" --output "$root/video-$seed" \
    --model "$model" --video > "$root/video.log" 2>&1
mkdir "$root/delivery"
python3 - "$root" "$seed" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]); seed=sys.argv[2]
def get(relative):
    return json.loads((root/relative/"evaluation.json").read_text())
expected, recorded=get(f"eval-{seed}"),get(f"video-{seed}")
for key in ("model_sha256","mode","fingerprint","steps_executed","horizon_steps","stop","valid_rate","visible_rate"):
    assert expected[key]==recorded[key],key
shutil.copyfile(root/f"video-{seed}"/"evaluation.mp4",root/"delivery"/f"c2-x3-v4-seed-{int(seed):02d}.mp4")
shutil.copyfile(root/"summary.json",root/"delivery"/"summary.json")
checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"delivery").iterdir()}
(root/"delivery"/"checksums.json").write_text(json.dumps(checks,indent=2)+"\n")
PY
ffprobe -v error -show_entries format=duration -of csv=p=0 "$root"/delivery/*.mp4
mark complete
