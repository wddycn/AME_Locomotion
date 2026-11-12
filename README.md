# AME Locomotion Reproduction (Isaac Lab + Unitree G1)

Language: [中文](#中文) | [English](#english)

---

## 中文

### 项目简介

本项目是对论文 Attention-Based Map Encoding for Learning Generalized Legged Locomotion 中
基于注意力的地形编码器方法 AME 的复现实现。

- 仿真与训练平台: NVIDIA Isaac Sim 5.1.0 + Isaac Lab 2.3.0
- 机器人平台: Unitree G1 29DoF
- 强化学习框架: RSL-RL (含本项目自定义网络扩展)

核心目标是基于高程图和注意力机制，学习具备更强地形泛化能力的腿式运动策略。

### 方法实现位置

AME 的主要网络实现在:

- [rsl_rl/rsl_rl/modules/actor_critic_encoder.py](rsl_rl/rsl_rl/modules/actor_critic_encoder.py)

该文件包含:

- 地形图卷积特征提取
- 局部地形特征与本体状态的多头注意力融合
- 与 Actor/Critic 网络对接的编码输出

### 安装说明

1. 安装 Isaac Lab

- 参考官方文档: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- 建议使用 conda 环境。

2. 安装本项目扩展

```bash
python -m pip install -e source/ame_locomotion
```

3. 安装自定义 rsl_rl

```bash
python -m pip install -e rsl_rl
```

说明: 本项目使用了自定义 Actor-Critic 地形编码网络，若不安装本地 rsl_rl，可能无法正确导入或运行。

4. 预训练模型目录

- 项目内提供了预训练模型目录: [pretrained/](pretrained/)
- 本仓库已提供预训练检查点文件（.pt），可直接用于快速测试。
- ame1.pt: 当前默认配置训练得到（高程图 33×21，开启 CNN 下采样）。
- ame2.pt: 在默认配置基础上加入全局上下文（attach_global=True）。

### 快速开始

训练:

```bash
bash run_train.sh
```

测试与可视化:

```bash
bash run_play.sh
```
续训：
```
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-v0 \
  --resume \
  --load_run 2026-10-09_23-12-23 \
  --checkpoint model_14999.pt \
  --max_iterations 5000 \
  --num_envs 1024 \
  --headless
```
### XLL Stage 4 完整路线续训

Stage 4 从 Stage 3 checkpoint 续训，训练地形包含约 80% 的完整路线（地面、0.8 m 方台、梅花桩、0.8 m 方台、落地）和约 20% 的原 Stage 3 梅花桩地形。完整路线的台高按地形等级从约 0.10 m 增加到约 0.20 m。起步位置和朝向轻微随机；只有通过末台并落地，完整路线才升级。策略观测与动作维度不变。

在安装了 Isaac Sim / Isaac Lab 的 AME Python 环境中运行：

```bash
bash scripts/rsl_rl/train_xll_stage4.sh
```

默认加载 `logs/rsl_rl/xll_ame/2026-10-24_23-26-32/model_21599.pt`，继续训练 3000 次迭代，结果存到新的 `logs/rsl_rl/xll_ame/*_stage4_course/`。参数可在脚本命令后覆盖，例如 `--num_envs 256 --max_iterations 100`。续训的迭代数是**额外**迭代数，不是最终迭代编号。

Stage 4 的专用播放任务已移除。部署前仍需单独核对高程图和真机传感器观测。

### 两阶段训练说明

请注意 AME 采用两阶段训练流程:
第一阶段完成后，将 [velocity_env_cfg_29dof.py](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py) 中的 `FINETUNE` 设为 `True`，再启动第二阶段训练。

### 复现说明与实现调整

整体上我们遵循论文设计，同时尝试了三点小调整。

1. CNN 输入使用 xyz 坐标，而非仅 z 高度

- CNN 直接处理 xyz 三维输入。
- CNN 输出后不再额外拼接坐标。
- 我们观察到该设置通常有更好的训练表现。
- 可能原因是: 直接输入 xyz 能让网络更早学习到位置相关信息。

2. 用 CNN 步长下采样降低注意力计算开销

- 提高高程图分辨率或扩大范围时，MHA 序列长度会显著增加。
- 我们在 CNN 阶段通过步长下采样缩短后续注意力序列。
- 测试中未观察到明显的最终策略性能下降，同时训练开销更低。

3. 引入 AME2 中的全局上下文

- 我们加入了 AME2 提出的通过 MLP + max-pool 获取全局上下文信息的设计。
- 测试发现该设计能够提升策略表现。
- 代价是注意力权重的可解释性变差，并且训练开销明显增加。

### 关键文件

- AME 编码器实现: [rsl_rl/rsl_rl/modules/actor_critic_encoder.py](rsl_rl/rsl_rl/modules/actor_critic_encoder.py)
- 训练脚本: [run_train.sh](run_train.sh)
- 测试脚本: [run_play.sh](run_play.sh)

---

