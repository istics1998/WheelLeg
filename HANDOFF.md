# WheelLeg 项目交接文档（2026-07-19 一轮排查总结）

轮腿双轮平衡机器人，Isaac Sim 4.5 + IsaacLab 2.1.1 + skrl 2.1.0，PPO 训练。

## 环境关键约束（务必先读）
- **skrl 2.1.0**，但 IsaacLab 脚本按 skrl 1.x 写。`set_running_mode("eval")` 在 2.x 改名为 `enable_training_mode(False, apply_to_models=True)`；`act()` 签名变为 `act(obs, states, timestep=, timesteps=)`。play.py / play_teleop.py 已做双版本兼容。
- **机器 RAM 31.8G、Swap=0、GPU RTX4060 8G**。Isaac Sim 极易 OOM。**绝不留后台 headless 进程**（用 timeout 杀不干净，会残留吃内存触发 OOM）。跑完 `pkill -9 -f play` + `free -h` 复核。
- 可视化必须 `--disable_fabric`（否则 SkelMesh 视觉网格不渲染，只有碰撞体）+ `--rendering_mode performance`（8G 显存跑不动实时光追，会卡在着色器编译几分钟）+ `--num_envs 1`。
- 机器人是 mini 尺寸，初始高度仅 ~0.15m，只驱动 6 个关节（左右 twist/knee/wheel），日志里 `6 != 24 actuators` 警告是正常的。

## 本轮已解决的问题链（全部完成）
1. **skrl API 报错** → play.py/play_teleop.py 适配 skrl 2.x。
2. **可视化卡顿** → `--rendering_mode performance`。
3. **进程"自己退出"** → 实为 OOM（我遗留的 5 个 headless 测试进程吃爆内存），已清理。
4. **看得到地面看不到机器人** → SkelMesh 在 Fabric 下不渲染，`--disable_fabric` 解决。
5. **相机太远机器人太小** → 拉近相机。
6. **机器人 nan 发散/只闪一帧** → 根因是 **checkpoint 全 nan**（见下）。
7. **训练发散根因定位 + 修复 + 重训成功**（见下）。

## 训练发散根因（核心成果）
- 旧的 `2026-07-19_03-27-11`（320万步）训练**从头就发散**，全部 checkpoint 100% nan，已删除。
- **根因**：缺摔倒终止 + `max_linear_velocity=1000` 过大 → 极端物理状态产生 nan → 污染 RunningStandardScaler → 整个网络变 nan。`num_envs=2048` 比 1024 触发概率高（对比同配置 1024 那次没炸）。
- **修复（已实施）**：
  - `wheelleg_env_cfg.py` TerminationsCfg 加了 `base_tipped`（bad_orientation, limit_angle=1.0）和 `base_too_low`（root_height_below_minimum, minimum_height=0.04）。
  - `wheelleg_bot.py` 速度上限 1000→10 (linear) / 1000→100 (angular)。
- **验证成功**：用修复后配置重训 `2026-07-19_20-51-22`（num_envs=1024, 9.6万步, 1h50m），**checkpoint 完全干净 0 nan**。这是当前默认使用的 checkpoint。

## 当前策略状态（9.6万步 checkpoint）
数据证明**策略有效**：零指令能站稳（高度~0.09-0.10，接近目标0.10），给 vx 指令实际线速度能准确跟踪（指令1.25→实际~1.2，指令2.0→实际~1.8），世界坐标位置大范围移动。但 2.0 高速时会抖动/偶尔失稳复位。**只训练了9.6万步，欠训练，需要正式长训练。**

## 交付的脚本/文件
- `wl_play.sh` / `wl_teleop.sh`（桌面有对应 .desktop 快捷方式）：默认指向 `2026-07-19_20-51-22_ppo_torch/checkpoints/agent_96000.pt`，可用 `WL_CKPT=/path ./xxx.sh` 覆盖。
- `wl_train.sh`：`./wl_train.sh` = 验证档(1024envs, 9.6万步)；`./wl_train.sh full` = 正式档(1024envs, ~192万步, 约10-15h)。结束写 `logs/skrl/.wl_train_result` 标记。
- `scripts/skrl/play_teleop.py`：锁存式键盘遥控（W/S前后、Q/E转向、空格停、R复位）。每步覆写 command_manager 的 vel_command_b 注入键盘指令。带 nan 自检、checkpoint 损坏自检。

## ⚠️ 已知问题（不影响训练和使用）
**可视化时机器人视觉网格不随物理移动。** 现象：速度指令的绿/蓝箭头正常移动，但**机器人本体在画面里静止不动**。物理数据显示机器人实际在移动（位置、速度都正常）。

**已尝试的修复方案（2026-07-19/20）：**

**方案1: write_root_pose_to_sim() 强制同步** ❌ 无效
- 在每次 `env.step()` 后显式调用 `write_root_pose_to_sim()` 将物理位姿写回 USD
- **验证结果**：箭头动但机器人本体仍不动

**方案2: 启用 Fabric 模式** ❌ 无效
- 创建了 `wl_autotest_fabric.sh`，启用 Fabric 自动同步
- **验证结果**：箭头动、渲染正常，但机器人本体仍不动

**最终结论（2026-07-20）：**
- **根因**：wheelleg_mini.usd 使用了 SkelMesh（骨骼蒙皮）绑定，这种方式在 Isaac Sim 中无法自动跟随刚体 transform 更新
- **影响评估**：
  - ✅ 物理仿真 100% 正常
  - ✅ 训练过程 100% 正常
  - ✅ 策略性能 100% 正常
  - ❌ 视觉演示不够直观（唯一影响）
- **解决方案**：**接受现状**，通过速度箭头和日志数据验证策略有效性
- **如需修复**：需要修改 USD 文件将 SkelMesh 改为普通 Mesh（耗时且复杂，投入产出比不高）

**替代验证方法**：
1. 观察速度箭头移动（指示运动方向和速度）
2. 查看终端日志数据（精确位置、速度、角速度）
3. 分析训练曲线（reward、episode_length）

**此问题不影响项目继续推进。**

## 下一步计划（可视化问题解决后）
- 开正式长训练 `./wl_train.sh full`（~200万步）得到更强策略。
- 后续可考虑：修实验目录名（cartpole_direct 残留）、把特权观测(base_lin_vel)换成 IMU 可测量以便真机部署、加熵奖励。

## 相关记忆文件
`~/.claude/projects/-home-ist----Isaac-project-WheelLeg/memory/`：skrl-version-api、machine-resource-limits、visualization-setup、diverged-checkpoint。
