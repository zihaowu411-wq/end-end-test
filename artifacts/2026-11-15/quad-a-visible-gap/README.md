# 四旋翼阶段 A · visible-gap

2026-11-15 在 AutoDL 的 RTX 3090 上完成 65,536 步。最后 20 个训练回合的有效跟踪率为 29.5%，目标可见率为 80.6%，碰撞回合为 0，达标回合为 4。阶段 A 尚未达标，本轮只有训练统计。

- [`final.zip`](final.zip) 为训练结束时保存的 Recurrent PPO 模型。
- [`config.json`](config.json) 和 [`training-summary.json`](training-summary.json) 来自该次运行。
- [对应的场景与训练代码](https://github.com/zihaowu411-wq/end-end-test/tree/0bf85cc/gazebo-occlusion-one-pillar)

从 AutoDL 克隆后的 `gazebo-occlusion-one-pillar/visual-tracking` 目录出发，需要续训时使用下列命令。输出目录位于新实例的数据盘。

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PY=/opt/gazebo-visual-tracking/venv/bin/python
RUN="/root/autodl-tmp/results/quad-a-visible-gap-resume-$(date +%Y%m%d-%H%M%S)"
"$PY" -u train.py --vehicle quad --stage A --tracking-reward visible-gap \
  --device cuda --environments 4 --rollout-steps 256 --learning-rate 0.0003 \
  --seed 1800 --steps 65536 --rtf 30 --output "$RUN" \
  --resume ../../artifacts/2026-11-15/quad-a-visible-gap/final.zip
```
