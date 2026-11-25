# 无障碍三维 X3 跟踪

两次训练均从零开始，使用种子 17、8 个并行环境、每环境每轮采样 256 步、学习率 `1e-4`、批大小 64 和 Recurrent PPO。Stage B 沿用水平随机转向，并叠加平滑升降。策略使用 256×144 前视 RGB，控制频率为 20 Hz。动作依次为前向速度、侧向速度、偏航角速度和竖直速度。奖励使用 `safe-effective-tracking`，名义跟踪距离为 2 m，距离按三维直线计算。

| PPO 轮次 | 采样步数 | 最终模型 | 原始训练配置 | 训练汇总 | 独立评估 |
| ---: | ---: | --- | --- | --- | --- |
| 128 | 262,144 | [final.zip](round-128/final.zip) | [配置](round-128/train-config.json) | [汇总](round-128/training-summary.json) | [报告](round-128/evaluation.json) |
| 256 | 524,288 | [final.zip](round-256/final.zip) | [配置](round-256/train-config.json) | [汇总](round-256/training-summary.json) | [报告](round-256/evaluation.json) |

两次评估都使用最终模型的确定性动作，种子为 28400–28419，每局 12 s。有效跟踪要求目标可见且距离在 1.5–2.5 m。128 轮的有效跟踪率为 50.94%，20 局零碰撞。256 轮的有效跟踪率为 69.98%，20 局中 1 局碰撞。

## 复现

从仓库根目录进入训练代码目录。训练步数按上表选择，输出路径使用新目录。配置 JSON 保留原始服务器参数和路径。

```bash
repo="$PWD"
cd gazebo-occlusion-one-pillar/visual-tracking
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
py=/opt/gazebo-visual-tracking/venv/bin/python
common=(--stage B --vehicle quad --ideal-velocity --free-flight-3d
        --observation rgb --tracking-reward safe-effective-tracking
        --target-gap-m 2 --target-visual-scale 1 --camera-resolution 256 144 --rtf 30)

"$py" -u train.py "${common[@]}" --seed 17 --device cuda --environments 8 \
  --rollout-steps 256 --learning-rate 1e-4 --steps 524288 --output /tmp/flight-3d-256

"$py" -u evaluate.py "${common[@]}" --model "$repo/artifacts/2026-11-26/free-flight-3d/round-256/final.zip" \
  --episodes 20 --first-seed 28400 --output /tmp/flight-3d-256-eval
```

## 侧视回放

三段回放使用 128 轮最终模型，包含机载 RGB、固定侧视、实时高度和三维间距。

- [Seed 28409，跟踪达标](side-replays/3d-side-good-tracking-seed-28409.mp4)
- [Seed 28400，目标丢失](side-replays/3d-side-target-lost-seed-28400.mp4)
- [Seed 28413，间距偏大](side-replays/3d-side-visible-gap-too-far-seed-28413.mp4)

可在本目录运行 `sha256sum -c SHA256SUMS` 核对归档文件。
