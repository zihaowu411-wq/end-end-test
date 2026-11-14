#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python=/opt/gazebo-visual-tracking/venv/bin/python
stage_b=${1:?Provide the Stage B checkpoint}
root=${2:?Provide a new run directory}
test -s "$stage_b"
mkdir "$root"

"$python" -u train.py --stage C --device cuda --environments 4 --rollout-steps 256 \
    --learning-rate 0.00008 --seed 1720 --steps 65536 --rtf 35 \
    --clearance-penalty --resume "$stage_b" --output "$root/train"
"$python" -u evaluate.py --stage C --clearance-penalty --rtf 30 --episodes 20 --first-seed 25000 \
    --model "$root/train/final.zip" --output "$root/eval"
video_seed=$("$python" -c 'import json,sys; rows=json.load(open(sys.argv[1]))["results"]; print(max(rows,key=lambda x:(not x["collision"],bool(x["gap_recovery_times_s"]),x["valid"]))["seed"])' "$root/eval/evaluation.json")
"$python" -u evaluate.py --stage C --clearance-penalty --rtf 12 --episodes 1 --first-seed "$video_seed" --video \
    --model "$root/train/final.zip" --output "$root/replay-selected"
"$python" -u evaluate.py --stage C --clearance-penalty --rtf 12 --episodes 1 --first-seed 25000 --video \
    --model "$root/train/final.zip" --output "$root/replay-route-25000"
for replay in "$root"/replay-*/tracking.mp4; do
    ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames -of compact "$replay"
done
touch "$root/completed"
