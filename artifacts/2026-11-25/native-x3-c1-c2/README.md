# 正常尺寸红色 X3 的 RGB C1 与 C2

本目录保存 2026-11-25 训练的 Recurrent PPO 模型。策略输入为 256×144 前视 RGB。目标使用原尺寸全红 X3，视觉缩放为 1.0，名义跟踪距离为 2 m。

| 模型 | 训练起点 | 本段采样步数 | 模型累计步数 | 服务器原始训练配置 |
| --- | --- | ---: | ---: | --- |
| [c1-final.zip](c1-final.zip) | 从零初始化 | 262,144 | 262,144 | [C1 128 轮](c1-train-config.json) |
| [c2-round-64.zip](c2-round-64.zip) | C1 最终模型 | 131,072 | 393,216 | [C2 前 64 轮](c2-round-64-train-config.json) |
| [c2-final.zip](c2-final.zip) | C2 第 64 轮模型 | 131,072 | 524,288 | [C2 后 64 轮](c2-round-128-train-config.json) |

训练使用 RTX 3090、8 个并行环境、每环境每轮 256 步和固定学习率 `1e-4`。C1 使用随机单柱平滑路线及种子 17。C2 使用 16 根障碍物、随机抽取穿梭与八字路线，前后两段各 64 轮，训练种子分别为 17、25。C1 和 C2 均使用 `safe-effective-tracking` 和横向净空奖励。

C2 最终模型的完整训练过程对应表中两份 C2 配置。复跑时，按模型对应关系设置 `resume`，并将 `output` 改为本机的新输出目录。原始 JSON 保留服务器当时的路径。

评估使用最终权重的确定性动作。C1 的 20 局全部安全过柱，整局有效跟踪率为 97.1%。C2 的穿梭和八字各评估 10 局，20 局均无碰撞完成路线，有效跟踪率为 100%。逐局指标见 [C1 评估](c1-evaluation.json)和 [C2 评估](c2-evaluation.json)。

评估入口为 `gazebo-occlusion-one-pillar/visual-tracking/evaluate.py`。使用 `--vehicle quad --ideal-velocity --observation rgb --tracking-reward safe-effective-tracking --transverse-clearance-shaping --target-gap-m 2 --target-visual-scale 1`。C1 加 `--stage C1 --randomized-c1 --smooth-c1-route`，C2 加 `--stage C2` 并通过 `--c2-route` 选择路线。用对应的 `--model` 指定最终模型。

下载后可在本目录运行 `sha256sum -c SHA256SUMS` 核对三个权重文件。

同一 C2 最终模型在四个复杂轨迹场景中的结果与回放见 [V4 评估](../native-x3-c2-v4/)。
