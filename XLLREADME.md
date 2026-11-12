# XLL 四足机器人训练

本项目将 AME 高程图注意力策略用于 XLL 四足机器人，在 Isaac Lab 中分阶段训练。运行前需安装 Isaac Sim、Isaac Lab、本项目扩展和仓库内的 `rsl_rl`；安装方法见 [主 README](README.md#安装说明)。以下命令均在 `AME_Locomotion` 根目录执行。

# 一、梅花桩训练任务
## 训练阶段

| 阶段 | 训练任务 | 播放任务 | 目标 |
| --- | --- | --- | --- |
| 0 | `AME-XLL-Stage0-v0` | `AME-XLL-Stage0-Play-v0` | 楼梯、方块、坡面等基础地形 |
| 1 | `AME-XLL-Rough-v0` | `AME-XLL-Rough-Play-v0` | 踏石、沟壑等粗糙地形 |
| 2 | `AME-XLL-Stage2-v0` | `AME-XLL-Stage2-Play-v0` | 综合复杂地形与梅花桩 |
| 3 | `AME-XLL-Stage3-v0` | `AME-XLL-Stage3-Play-v0` | 梅花桩坑深课程及坑底触地惩罚 |
| 4 | `AME-XLL-Stage4-v0` | — | 上方台、过桩、下方台的完整路线 |

XLL 各阶段共用 12 关节策略的观测与动作接口，可以按顺序续训。Stage 2/3 的播放任务也会随机初始位置和朝向。Stage 4 训练地形约 80% 为完整路线、20% 为 Stage 3 桩地形；方台高随课程由约 0.10 m 增至 0.20 m。

## 训练与续训

从头训练 Stage 0：

```bash
python scripts/rsl_rl/train.py --task AME-XLL-Stage0-v0 \
  --max_iterations 10000 --num_envs 1024 --headless
```

后续阶段使用上一阶段的运行目录和 checkpoint，例如从 Stage 3 续训 Stage 4：

```bash
python scripts/rsl_rl/train.py --task AME-XLL-Stage4-v0 \
  --resume --load_run 2026-10-24_23-26-32 --checkpoint model_21599.pt \
  --run_name stage4_course --max_iterations 3000 --num_envs 1024 --headless
```

训练记录保存在 `logs/rsl_rl/xll_ame/`。`--load_run` 是其中的运行目录名；`--max_iterations` 表示本次**额外**训练的迭代数。训练其他阶段时，替换 `--task`、`--load_run` 和 `--checkpoint` 即可。

## 播放与导出

任务名应与 checkpoint 阶段对应。以 Stage 3 为例：

```bash
python scripts/rsl_rl/play.py --task AME-XLL-Stage3-Play-v0 \
  --checkpoint 'logs/rsl_rl/xll_ame/<stage3_run>/model_<iteration>.pt' \
  --num_envs 1
```

将占位路径换成实际文件。只导出策略时，在上述命令后加 `--export both --export-only`；输出位于该运行目录的 `exported/`。部署前需核对关节顺序、高程图、观测归一化与动作执行参数。

查看所有 XLL 训练记录：

```bash
tensorboard --logdir logs/rsl_rl/xll_ame --port 6006
```

只看某一次训练时，将 `--logdir` 指向该次运行目录，例如：

```bash
tensorboard --logdir logs/rsl_rl/xll_ame/2026-10-24_23-26-32 --port 6006
```

在 TensorBoard 的 Scalars 页面还可以只勾选一项指标，查看单独一条曲线。

## 主要文件

- [XLL 任务配置](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/xll/velocity_env_cfg_xll.py)：各阶段观测、动作、奖励、随机化及环境配置。
- [XLL 任务注册](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/xll/__init__.py)：训练与播放任务名。
- [Stage 4 地形](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/terrains/stage4_terrain_cfg.py)：完整路线地形。
- [XLL 机器人模型](source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/assets/robots/xll.py)：关节、站姿和执行器参数。

# 轮胎阵专项训练与播放

`AME-XLL-Tire-Array-v0` 继承 Stage 2 的机器人、观测、动作、奖励和物理随机化，新增固定轮胎阵地形。8 个轮胎按横向 2 个、前进方向 4 个排列，从短边沿 +X 穿过。每组阵列独立采样外径 0.50–0.60 m、内径 0.30–0.36 m、胎宽 0.145–0.185 m；平放高度等于胎宽。轮胎建模为平顶、垂直内外侧壁的空心圆柱环，不使用弧形胎面。同一阵列的横纵中心距均等于该组实际外径，所有相邻轮胎相切；仅每四个轮胎交界处保留菱形空隙。

训练使用 8 级地形课程。第 0 级固定为外径 0.50 m、内径 0.30 m、胎宽 0.145 m；每提高一级，三个尺寸的随机化上限线性扩大；最高级覆盖完整尺寸范围。每级有 8 组独立随机阵列。每级连续 3 次通过出口才升一级，连续 3 次跌倒、越界或超时才降一级；初始只从第 0–1 级抽样。尺寸在建场时采样，reset 不改变网格。

轮胎使用固定的闭合椭圆截面环形网格，保留中孔，碰撞与高程扫描使用同一实际地形；不模拟滚动、变形。机器人在轮胎阵前方 1 m 出生，只接收 0.4–0.8 m/s 前进命令，横移和转向目标均为零，无站立命令。沿途机身中心偏离中线超过 0.35 m 或倒退超过 0.5 m 会失败终止；该约束针对机身中心，不要求每只脚始终落在轮胎上。前进到 3.9 m、通过最远出口后完成回合。

轮胎任务保留 XLL 的 Stage 2 默认站姿，不强制左右脚分别落入哪一列孔位。一侧脚落在轮胎孔内、另一侧脚从两列轮胎间的菱形缝隙落在地面也是有效策略。训练奖励机身前向速度至少 0.20 m/s 时的地面支撑脚：轮胎中孔和菱形缝隙都属于地面支撑；承重脚接触高于地面 0.07 m 的轮胎橡胶表面会受到惩罚。训练目标是沿 +X 稳定穿过阵列，保持前进速度，且不从阵列侧面绕行；机身通过 `x = 3.9 m` 的出口线才成功重置。最大尺寸轮胎阵末端约在 `x = 3.4 m`，因此还必须在出口空地继续前进约 0.5 m；到达终点会获得一次 25 分成功奖励。

从 Stage 2 权重继续训练（不会自动选择 checkpoint）：

```bash
python scripts/rsl_rl/train.py --task AME-XLL-Tire-Array-v0 \
  --resume --load_run <stage2_run> --checkpoint <model.pt> \
  --run_name tire_array --num_envs 1024 --max_iterations 3000 --headless
```

播放新训练的轮胎策略：

```bash
python scripts/rsl_rl/play.py --task AME-XLL-Tire-Array-Play-v0 \
  --checkpoint <轮胎训练模型的完整路径> --num_envs 1 --real-time
```

播放固定前进速度 0.6 m/s，关闭训练扰动和策略观测噪声，保留通关与越界终止。尺寸范围可在 `terrains/tire_terrain_cfg.py` 中调整。

# 沟壑障碍
可以直接用 Stage 2 的训练策略，只是在控制上需要高要求
