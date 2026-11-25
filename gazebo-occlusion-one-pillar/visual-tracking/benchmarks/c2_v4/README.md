# RGB C2 多障碍轨迹评估

`scenes.json` 保存了 V4 的四个固定场景，包含矩形障碍物的位置、尺寸与朝向，以及目标每 0.05 s 的位姿。用现有的 Gazebo 四旋翼环境即可重放相同场景。

| 场景种子 | 障碍物 | ≥25° 弯道 | 路线自交点 |
| --- | ---: | ---: | ---: |
| 0 | 21 | 25 | 8 |
| 1 | 22 | 25 | 5 |
| 2 | 21 | 28 | 6 |
| 3 | 22 | 26 | 5 |

红色正方体以 1 m/s 沿路线运行 100 s。追捕者从目标后方 2 m 起步，使用 256×144 的前视 RGB 与 20 Hz 控制频率。

## 复现

从 `visual-tracking` 目录执行。使用项目的 Gazebo Python 环境，录像需要 Pillow、DejaVu Sans 字体及 FFmpeg。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
benchmark=benchmarks/c2_v4

python "$benchmark/evaluate_scenes.py" --scenes "$benchmark/scenes.json" \
  --world ../one_pillar.sdf --seed 0 --smoke-steps 20 --output /tmp/c2-v4-smoke-0
```

使用混合轨迹 C2 在累计 262,144 步保存的 RGB Recurrent PPO 最终模型。通过 `C2_MODEL` 指定模型文件的绝对路径。将 `--seed` 改为 0–3，给每个场景使用新的输出目录。

```bash
python "$benchmark/evaluate_scenes.py" --scenes "$benchmark/scenes.json" \
  --world ../one_pillar.sdf --seed 0 --model "$C2_MODEL" \
  --output /tmp/c2-v4-eval-0 --video
```

脚本以确定性策略评估，输出逐步记录 `trace.csv`、统计 `evaluation.json`，并可保存 `evaluation.mp4`。整局有效跟踪率以 2,000 步为分母，符合距离 1.5–2.5 m 且目标可见的步数计入分子。通过条件为全程安全完成且有效跟踪率至少 80%。

## 正常尺寸全红 X3

X3 模型使用 `--target x3` 和 [累计 524,288 步的 C2 最终权重](../../../../artifacts/2026-11-25/native-x3-c1-c2/c2-final.zip)。三个场景清单使用相同的 4 条路线、障碍物和 2 m 初始间距，按不同速度重放。

| 目标速度 | 场景清单 | 每局时长 |
| ---: | --- | ---: |
| 1.00 m/s | [scenes.json](scenes.json) | 100.00 s |
| 0.75 m/s | [scenes-075.json](scenes-075.json) | 133.35 s |
| 0.60 m/s | [scenes-060.json](scenes-060.json) | 166.70 s |

从仓库根目录运行四场景评估。输出目录应为一个新目录，父目录已存在。使用项目的 Gazebo Python 环境。录像依赖 FFmpeg、Pillow 和 DejaVu Sans 字体。

```bash
bench=gazebo-occlusion-one-pillar/visual-tracking/benchmarks/c2_v4
model=artifacts/2026-11-25/native-x3-c1-c2/c2-final.zip
GAZEBO_PYTHON=/opt/gazebo-visual-tracking/venv/bin/python \
  bash "$bench/run_x3_suite.sh" "$bench/scenes-060.json" "$model" /tmp/c2-v4-x3-060
```

脚本保存逐场景记录、完整路线及前 100 秒指标，并为持续时间最长的场景录制一段回放。多局持续时间相同时，选取整局有效跟踪率最接近四局中位数的场景。原始结果见 [三档速度的评估与回放](../../../../artifacts/2026-11-25/native-x3-c2-v4/)。
