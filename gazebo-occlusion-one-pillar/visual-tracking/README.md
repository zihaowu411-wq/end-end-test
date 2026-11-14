# Gazebo 视觉跟车 Recurrent PPO

定高四旋翼、`visible-gap` 奖励及新 AutoDL 实例的启动命令见[场景与恢复说明](../README.md#定高四旋翼阶段-a)。

策略观测来自蓝车的前视 RGB，相机画面以 `3×72×128` 的 RGB 字节张量送入 CNN 和 LSTM。连续动作映射到 `0–1.5 m/s` 的线速度和 `±1.5 rad/s` 的角速度，每秒控制六次。目标车按仿真程序行驶。车距、方位、可见性和碰撞供奖励和评估使用。

## 在 AutoDL 运行

服务器路径为 `/root/autodl-tmp/cloud_research/simulators/gazebo-occlusion-one-pillar/visual-tracking`。场景模板 `one_pillar.sdf` 位于上一级目录。依赖是 Gazebo Sim 8、系统 Python 3.10 的 Gazebo Transport 13、Msgs 10、Pillow、FFmpeg，以及项目 `.venv` 中的 PyTorch 2.8、SB3 2.9、SB3-Contrib 2.9 和 TensorBoard。

```bash
cd /root/autodl-tmp/cloud_research/simulators/gazebo-occlusion-one-pillar/visual-tracking
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
.venv/bin/python -u smoke.py --rtf 20
```

重新训练默认用 CUDA，并为每次运行使用新的输出目录。

```bash
.venv-gpu/bin/python -u train.py --stage A --seed 17 --steps 12288 --rtf 20 --device cuda \
  --output runs/straight-rgb-recurrent-ppo-repeat
.venv/bin/python -u evaluate.py --stage A --rtf 20 --episodes 20 --first-seed 20000 \
  --model runs/straight-rgb-recurrent-ppo-repeat/final.zip \
  --output runs/straight-rgb-recurrent-ppo-repeat-eval
```

在单独的输出目录加上 `--video` 可以生成 MP4，视频上半部是前视和俯视画面，下半部是实际车距曲线。训练过程中用 `tail -f runs/straight-rgb-seed17.log` 看首次运行的日志。每 2,048 步保存一个滚动批次末尾的检查点。

## 评估

- 以每回合计划长度为分母，目标处于相机水平视野且两车中心距为 `2.5–3.5 m` 的步数为分子。提前结束后的剩余步数按未达标计。
- 阶段 A 无柱，目标车直行和变速。阶段 B 无柱，增加左右转。阶段 C 使用单柱，视线遮挡按相机到目标中心的平面线段与柱体交集计算。
- 阶段 A 的通过条件为 20 条指定评估路线的平均有效跟踪率至少 80%，且无碰撞。各回合的距离、可见性、动作及奖励可在 `trace.csv` 中检查。

[Gazebo Harmonic 仿真控制](https://gazebosim.org/api/sim/8/pause_run_simulation.html) · [Recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
