# Gazebo 视觉跟车与 AutoDL 恢复

## 2026-11-14 工作记录

- 场景有两辆差速小车、一根遮挡柱、追踪车前视 RGB 和俯视相机。`one_pillar.sdf` 同时是训练场景模板，训练阶段 A 和 B 会在生成世界时移除柱子。
- 追踪策略为 RGB → CNN → LSTM → Recurrent PPO → 线速度与转向角速度。目标车脚本行驶，仿真真值用于奖励和评估。
- 阶段 A 的 CPU 首轮完成了 12,288 步。在 20 条评估路线中，有效跟踪率为 30.3%，目标全程可见，碰撞为 0。有效跟踪要求目标可见且车距在 2.5–3.5 m。第一轮尚未达到 80% 的阶段门槛。
- `artifacts/2026-11-14/` 保存了 CPU 最终模型、评估结果、车距采样记录、双视角回放和单柱遮挡截图。

## 新 AutoDL 实例

1. 在「租用新实例」选「我的镜像」，选择已保存的 `gazebo-harmonic-visual-tracking-cu126-20261114`，选择驱动兼容 CUDA 12.6 的 NVIDIA GPU。
2. 用 VS Code Remote-SSH 连接新实例。GitHub 仓库是私有仓库，需要在新实例使用自己的 GitHub 授权。也可以从已登录的 Windows 端把仓库传过去。
3. 将仓库克隆到数据盘，并执行检查。系统镜像中的 Python 路径是 `/opt/gazebo-visual-tracking/venv/bin/python`。

```bash
mkdir -p /root/autodl-tmp/repos
git clone git@github.com:zihaowu411-wq/end-end-test.git /root/autodl-tmp/repos/end-end-test
cd /root/autodl-tmp/repos/end-end-test/gazebo-occlusion-one-pillar/visual-tracking
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PY=/opt/gazebo-visual-tracking/venv/bin/python
gz sim --version
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
"$PY" -u smoke.py --rtf 20
```

## 恢复后续训练

以下命令仅在决定恢复训练时运行。它从已备份的阶段 A 模型继续训练，在数据盘写入新的运行目录。用 20 条独立路线评估，通过 80% 且零碰撞门槛后再进阶段 B。

```bash
RUN="/root/autodl-tmp/results/straight-rgb-3090-$(date +%Y%m%d-%H%M%S)"
"$PY" -u train.py --stage A --device cuda --environments 4 --rollout-steps 64 \
  --seed 17 --steps 16384 --rtf 35 --output "$RUN" \
  --resume ../../artifacts/2026-11-14/stage-a-cpu-final.zip
"$PY" -u evaluate.py --stage A --episodes 20 --first-seed 23000 --rtf 35 \
  --model "$RUN/final.zip" --output "${RUN}-eval"
```

AutoDL 的镜像记录系统盘，`/root/autodl-tmp` 是每台实例的数据盘。新实例从镜像获得依赖，从 GitHub 获取代码和已备份的模型。其他未归档的训练输出可使用 AutoDL 文件存储或跨实例拷贝。 [AutoDL 镜像文档](https://www.autodl.com/docs/image/) · [AutoDL 迁移文档](https://www.autodl.com/docs/migrate_instance/)
