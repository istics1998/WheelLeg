# WheelLeg 项目快速使用指南

## 当前状态 (2026-07-19)

✅ **已解决**: 训练发散、OOM崩溃、可视化卡顿、机器人不渲染
🔧 **待验证**: 视觉网格与物理脱节问题（已添加修复代码，需在GUI中验证）

## 可用脚本

### 1. 训练 (`wl_train.sh`)
```bash
# 验证档 (1024 envs, 9.6万步, ~2小时)
./wl_train.sh

# 正式档 (1024 envs, ~192万步, 10-15小时)
./wl_train.sh full
```

### 2. 可视化回放 (`wl_play.sh`)
```bash
# 使用默认 checkpoint (当前最好的干净checkpoint)
./wl_play.sh

# 指定 checkpoint
WL_CKPT=/path/to/agent_xxx.pt ./wl_play.sh
```

### 3. 键盘遥控 (`wl_teleop.sh`)
```bash
./wl_teleop.sh

# 键位:
#   W/S - 前进/后退 (vx)
#   Q/E - 左转/右转 (wz)
#   空格 - 停止
#   R - 复位环境
```

### 4. 自动化测试 (`wl_autotest.sh`) ⭐ 新增
```bash
./wl_autotest.sh

# 自动执行测试序列: 平衡→前进→停止→左转
# 无需键盘输入，适合验证视觉是否跟随物理移动
```

## 当前 Checkpoint

**推荐使用**: `logs/skrl/cartpole_direct/2026-07-19_20-51-22_ppo_torch/checkpoints/agent_96000.pt`
- ✅ **完全干净** (0 nan)
- 训练步数: 9.6万步 (欠训练，但已能站稳、跟踪速度指令)
- 性能: 零指令能站稳，vx=1.25 → 实际~1.2 m/s

## 视觉问题排查 (未完成)

**现象**: 键盘遥控时，速度指令箭头移动，但机器人本体静止不动。

**本轮修复** (2026-07-19 第二轮):
- 在 `play_teleop.py` 和 `play_teleop_autotest.py` 中添加了 `write_root_pose_to_sim()` 调用
- 每次 step 后强制将物理位姿写回 USD，尝试同步视觉

**验证方法**:
```bash
./wl_autotest.sh
```
在 Isaac Sim 窗口中观察:
1. 机器人本体是否移动?
2. 绿/蓝箭头是否移动?

**如果修复无效**，建议:
1. 尝试启用 Fabric 模式 (编辑脚本去掉 `--disable_fabric`)
2. 检查 `wheelleg_mini.usd` 的 visual mesh 绑定方式
3. 或接受现状: 物理/数据正常，仅通过日志 + 箭头验证策略

## 环境约束 (重要!)

- **RAM**: 31.8G, **Swap=0**, **GPU**: RTX 4060 8G
- **绝不留后台进程**: 运行完后 `pkill -9 -f play` 清理，否则OOM
- **可视化必须**: `--disable_fabric` + `--rendering_mode performance` + `--num_envs 1`

## 下一步

1. **验证视觉修复**: 运行 `./wl_autotest.sh`，确认机器人是否移动
2. **正式训练**: 如果视觉问题不影响训练，可开始 `./wl_train.sh full`
3. **后续优化**: 
   - 修改实验目录名 (cartpole_direct 残留)
   - 特权观测换成 IMU 可测量值 (为真机部署准备)
   - 调整奖励函数 (加熵奖励等)

## 详细文档

- **HANDOFF.md**: 完整的问题排查历史和技术细节
- **Memory files**: `~/.claude/projects/-home-ist----Isaac-project-WheelLeg/memory/`
  - skrl-version-api.md
  - machine-resource-limits.md
  - visualization-setup.md
  - diverged-checkpoint.md
