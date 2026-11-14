#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

project_dir=$(cd -- "$(dirname -- "$0")" && pwd)
gz sdf -k "$project_dir/one_pillar.sdf"
run_dir=$(mktemp -d "$project_dir/capture.XXXXXX")
printf 'CAPTURE_DIR=%s\n' "$run_dir"
cd "$run_dir"
export GZ_PARTITION="one-pillar-$(basename "$run_dir")"

# AutoDL already uses software EGL rendering for its headless Gazebo camera check.
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
timeout -k 5 95 env -u DISPLAY gz sim -r -s --headless-rendering -v 3 \
    "$project_dir/one_pillar.sdf" >gazebo.log 2>&1 </dev/null &
sim_pid=$!
trap 'kill -TERM "$sim_pid" 2>/dev/null || true; wait "$sim_pid" 2>/dev/null || true' EXIT

if ! /usr/bin/python3 "$project_dir/capture.py" | tee report.txt; then
    tail -n 45 gazebo.log >&2
    exit 1
fi
