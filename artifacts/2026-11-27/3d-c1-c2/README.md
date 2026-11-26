# 三维 X3 跟踪与多柱避障

策略使用 256×144 前视 RGB 和 Recurrent PPO，控制频率 20 Hz。动作依次为前向速度、侧向速度、偏航角速度和竖直速度。名义跟踪距离 2 m，有效跟踪要求目标可见且三维距离在 1.5–2.5 m。各阶段使用 8 个并行环境、每轮每环境采样 256 步、学习率 0.0001、批大小 64、种子 17。

| 运行 | 模型 | 训练配置 | 独立评估 |
| --- | --- | --- | --- |
| B，512 轮 | [final.zip](stage-b-512/final.zip) | [配置](stage-b-512/train-config.json) | [报告](stage-b-512/evaluation.json) |
| C1，512 轮 | [final.zip](c1-512/final.zip) | [配置](c1-512/train-config.json) | [报告](c1-512/evaluation.json) |
| C1，768 轮 | [final.zip](c1-768/final.zip) | [配置](c1-768/train-config.json) | [报告](c1-768/evaluation.json) |
| C1，896 轮 | [final.zip](c1-896/final.zip) | [配置](c1-896/train-config.json) | [报告](c1-896/evaluation.json) |
| C2，205 轮恢复点 | [检查点](c2-205/checkpoint-3043328.zip) | [配置](c2-205/train-config.json) | 无 |
| C2，499 轮停止点 | [检查点](c2-499/checkpoint-3647488.zip) | [配置](c2-499/train-config.json) | 无 |

B 512 轮从零训练，20 局有效跟踪率 100%。C1 512 和 768 轮都从同一个 B 512 轮模型开始，各用 100 个固定种子做确定性评估。C1 768 的有效跟踪率为 98.22%，96/100 局无碰撞完成且有效跟踪率至少 95%。C1 896 轮从 C1 768 轮再续训 128 轮，评估有效跟踪率为 25.73%。

C2 从 C1 768 轮模型开始，随机使用穿行和八字路线。场地有 16 根柱子，目标每条路线经过其中 4 根，过柱时交替上升和下降。原训练在随机到过柱点落在回合起点的八字路线时中断。修复后的三维 C2 重新采样这类路线，平面 C2 沿用原采样规则。按同一批环境种子做的 [640 条路线验证](route-validation/validation-output.txt) 附有 [验证脚本](route-validation/check_routes.py)。

恢复训练从第 205 轮检查点开始，又完成 294 轮后按要求停止。最新检查点累计包含 **499 轮 C2 PPO 更新**，同时保存策略与优化器状态。[保存记录](c2-499/snapshot.json) 中的 SB3 采样计数还包含两次在更新前保存时已采集的 rollout。当前检查点没有独立评估结果。

## 复现 C2 训练设置

从仓库根目录进入 Gazebo 训练代码目录，下例传入归档的 499 轮检查点。原配置中的绝对路径来自实际服务器。剩余的 269 轮对应 550,912 步。

```bash
repo="$PWD"
cd gazebo-occlusion-one-pillar/visual-tracking
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
py=/opt/gazebo-visual-tracking/venv/bin/python
"$py" -u train.py --stage C2 --vehicle quad --ideal-velocity --free-flight-3d \
  --observation rgb --tracking-reward safe-effective-tracking --target-gap-m 2 \
  --target-visual-scale 1 --camera-resolution 256 144 --transverse-clearance-shaping \
  --c2-route mixed --rtf 30 --device cuda --seed 17 --environments 8 \
  --rollout-steps 256 --learning-rate 0.0001 --steps 550912 \
  --resume "$repo/artifacts/2026-11-27/3d-c1-c2/c2-499/checkpoint-3647488.zip" \
  --output /tmp/c2-resume-499
```

## 回放

- [B 512 轮，种子 28405，机载 RGB 和固定侧视](stage-b-512/side-seed-28405.mp4)
- [C1 768 轮，种子 28402，评估轨迹回放](c1-768/replay-seed-28402.mp4)

在本目录运行 `sha256sum -c SHA256SUMS` 可核对归档文件。
