#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python=/opt/gazebo-visual-tracking/venv/bin/python
initial_model=${1:?Provide the Stage A checkpoint to continue from}
root=${2:?Provide a new run directory}
test -s "$initial_model"
mkdir -p "$(dirname -- "$root")"
mkdir "$root"

train_stage() {
    local stage=$1 seed=$2 steps=$3 checkpoint=$4 directory=$5
    "$python" -u train.py --stage "$stage" --device cuda --environments 4 \
        --rollout-steps 256 --learning-rate 0.00008 --seed "$seed" --steps "$steps" --rtf 35 \
        --resume "$checkpoint" --output "$directory"
}

evaluate_stage() {
    local stage=$1 seed=$2 directory=$3
    "$python" -u evaluate.py --stage "$stage" --rtf 30 --episodes 20 --first-seed "$seed" \
        --model "$directory/final.zip" --output "$directory-eval"
}

replay_stage() {
    local stage=$1 seed=$2 directory=$3
    "$python" -u evaluate.py --stage "$stage" --rtf 10 --episodes 2 --first-seed "$seed" --video \
        --model "$directory/final.zip" --output "$directory-replay"
    ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames \
        -of compact "$directory-replay/tracking.mp4"
}

passed() {
    "$python" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["passed"])' \
        "$1-eval/evaluation.json"
}

a="$root/stage-a"
train_stage A 1717 32768 "$initial_model" "$a"
evaluate_stage A 23000 "$a"
if ! passed "$a"; then
    a_more="$root/stage-a-continued"
    train_stage A 1718 32768 "$a/final.zip" "$a_more"
    a="$a_more"
    evaluate_stage A 23000 "$a"
fi
replay_stage A 23000 "$a"
if ! passed "$a"; then
    printf '%s\n' 'Stage A did not meet the distance criterion.' > "$root/outcome.txt"
    exit 0
fi

b="$root/stage-b"
train_stage B 1719 49152 "$a/final.zip" "$b"
evaluate_stage B 24000 "$b"
replay_stage B 24000 "$b"
if ! passed "$b"; then
    printf '%s\n' 'Stage B did not meet the distance criterion.' > "$root/outcome.txt"
    exit 0
fi

c="$root/stage-c"
train_stage C 1720 65536 "$b/final.zip" "$c"
evaluate_stage C 25000 "$c"
replay_stage C 25000 "$c"
printf '%s\n' 'Stages A–C trained and evaluated.' > "$root/outcome.txt"
