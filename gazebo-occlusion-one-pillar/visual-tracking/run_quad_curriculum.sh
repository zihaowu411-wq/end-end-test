#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python=/opt/gazebo-visual-tracking/venv/bin/python
root=${1:?Provide the curriculum run directory}
stage_a_pid=${2:?Provide the running Stage A trainer PID}

train_from() {
    local stage=$1 seed=$2 steps=$3 checkpoint=$4 directory=$5
    shift 5
    "$python" -u train.py --vehicle quad --stage "$stage" --device cuda --environments 4 \
        --rollout-steps 256 --learning-rate 0.0001 --seed "$seed" --steps "$steps" --rtf 30 \
        --resume "$checkpoint" --output "$directory" "$@" > "$directory.log" 2>&1
}

assess() {
    local stage=$1 seed=$2 directory=$3
    "$python" -u evaluate.py --vehicle quad --stage "$stage" --rtf 30 --episodes 20 \
        --first-seed "$seed" --model "$directory/final.zip" --output "$directory-eval"
}

passed() {
    "$python" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["passed"])' \
        "$1-eval/evaluation.json"
}

a="$root/stage-a"
until [ -f "$a/completed" ]; do
    if ! ps -p "$stage_a_pid" -o stat= | grep -q '^[^Z]'; then
        tail -40 "$a.log"
        exit 1
    fi
    sleep 10
done

if [ ! -s "$a-eval/evaluation.json" ]; then
    assess A 28100 "$a"
fi
if ! passed "$a"; then
    next="$root/stage-a-continued"
    if [ -e "$next" ]; then next="$root/stage-a-retry"; fi
    train_from A 1801 65536 "$a/final.zip" "$next"
    a=$next
    assess A 28100 "$a"
fi
if ! passed "$a"; then
    echo "Stage A threshold unmet" > "$root/outcome.txt"
    exit 0
fi

b="$root/stage-b"
train_from B 1810 65536 "$a/final.zip" "$b"
assess B 28200 "$b"
if ! passed "$b"; then
    next="$root/stage-b-continued"
    train_from B 1811 65536 "$b/final.zip" "$next"
    b=$next
    assess B 28200 "$b"
fi
if ! passed "$b"; then
    echo "Stage B threshold unmet" > "$root/outcome.txt"
    exit 0
fi

c="$root/stage-c1"
train_from C1 1820 65536 "$b/final.zip" "$c" --mirrored-pillar 0.18 -2.0 \
    --clearance-penalty --clearance-threshold 1.0 --max-clearance-penalty 0.18
for side in left right; do
    pillar_x=0.18
    seed=28300
    if [ "$side" = left ]; then pillar_x=-0.18; seed=28301; fi
    "$python" -u evaluate.py --vehicle quad --stage C1 --pillar-center "$pillar_x" -2.0 \
        --rtf 30 --episodes 10 --first-seed "$seed" --seed-stride 2 \
        --model "$c/final.zip" --output "$c-eval-$side"
done
echo "Stages A, B and C1 trained and evaluated" > "$root/outcome.txt"
