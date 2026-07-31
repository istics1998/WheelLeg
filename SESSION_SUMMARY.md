# WheelLeg 项目工作总结 - 2026-07-19 第二轮

## 本轮任务
继续解决上一轮遗留的**机器人视觉网格不随物理移动**的问题。

## 问题分析

**现象**:
- 键盘遥控时，速度指令箭头（绿/蓝）明显移动和旋转
- 机器人本体在画面中静止不动
- `[DBG]` 日志显示机器人世界坐标大范围变化，速度跟踪正常
- 日志警告: `getBypassRenderSkelMeshProcessing ... has not been populated`

**根因假设**:
在 `--disable_fabric` 模式下，IsaacLab 的 Articulation 只更新物理刚体，但 SkelMesh 视觉网格的 USD transform 没有自动同步。Fabric 模式有自动同步机制，禁用后需要显式调用 `write_root_pose_to_sim()`。

## 实施的修复方案

### 1. 修改 `play_teleop.py` (行 294-299)
在每次 `env.step()` 之后添加强制视觉同步代码：

```python
# [FIX] 在 disable_fabric 模式下强制更新视觉 transform
if args_cli.disable_fabric and hasattr(robot, 'write_root_pose_to_sim'):
    root_pose = torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=-1)
    robot.write_root_pose_to_sim(root_pose)
```

**原理**: `write_root_pose_to_sim()` 显式将物理仿真中的 root 位姿写回 USD prim 的 xformOp，强制更新渲染用的 transform。

### 2. 创建自动化测试脚本

**新文件**:
- `scripts/skrl/play_teleop_autotest.py` - 自动化测试版 play_teleop，无需键盘输入
- `wl_autotest.sh` - 便捷启动脚本

**测试序列**:
1. 零指令平衡 50步
2. 前进 vx=1.0 持续 200步
3. 停止 50步  
4. 左转 wz=1.0 持续 100步

每个阶段打印位置、速度、移动距离，方便验证物理是否正常。

**用途**: 自动执行预定义的运动序列，让用户专注在 GUI 窗口中观察机器人本体是否移动，无需手动按键。

### 3. 文档更新

- **HANDOFF.md**: 更新"未解决问题"部分，记录本轮修复尝试和验证方法
- **QUICK_START.md**: 创建快速上手指南，汇总所有可用脚本和使用方法
- **Memory 文件**: 
  - 新增 `visual-sync-fix-attempt.md` 记录修复细节
  - 更新 `MEMORY.md` 索引

## 验证方法

由于需要在 Isaac Sim 的 GUI 窗口中实际观察，无法通过命令行完全自动化验证。需要用户执行：

```bash
./wl_autotest.sh
```

**观察要点**:
1. 机器人本体是否在画面中移动？
2. 速度指令箭头（绿/蓝）是否移动？

**判断标准**:
- ✅ **修复成功**: 机器人本体和箭头都移动
- ❌ **仍有问题**: 箭头移动但机器人静止 → 视觉脱节问题仍存在

## 备选方案（如果当前修复无效）

### 方案A: 启用 Fabric 模式
```bash
# 编辑 wl_teleop.sh 或 wl_autotest.sh，去掉 --disable_fabric
```
**优点**: Fabric 模式下 IsaacLab 自动同步 visual/physics  
**缺点**: SkelMesh 在 Fabric 下不渲染（但如果 USD 本身绑定有问题，这反而可能绕过问题）

### 方案B: 修改 USD 文件
检查 `wheelleg_mini.usd` 中 visual mesh 的绑定方式：
- 如果是 SkelRoot/SkelMesh（骨骼蒙皮），可能需要改为普通 Mesh
- 重新从 URDF/原始模型导出 USD

### 方案C: 接受现状
物理仿真和训练都正常，仅通过日志数据 + 箭头验证策略有效性，不依赖机器人本体的视觉渲染。

## 交付文件清单

**脚本**:
- ✅ `wl_autotest.sh` (新增) - 自动化视觉测试
- ✅ `scripts/skrl/play_teleop_autotest.py` (新增) - 自动化测试主程序
- ✅ `scripts/skrl/play_teleop.py` (修改) - 添加 `write_root_pose_to_sim()` 调用
- ✅ `scripts/debug_visual_sync.py` (新增) - 诊断脚本（参考用）
- ✅ `scripts/test_visual_movement.py` (新增) - 测试脚本（参考用）

**文档**:
- ✅ `HANDOFF.md` (更新) - 添加本轮修复尝试说明
- ✅ `QUICK_START.md` (新增) - 快速使用指南
- ✅ `memory/visual-sync-fix-attempt.md` (新增) - Memory 记录
- ✅ `memory/MEMORY.md` (更新) - 索引更新

**其他**:
- ✅ 所有脚本已 `chmod +x` 设置可执行权限
- ✅ `wl_autotest.sh` 已添加 `cd "$(dirname "$0")"` 确保相对路径正确

## 下一步行动

### 立即行动（用户）
1. 运行 `./wl_autotest.sh` 验证视觉修复是否有效
2. 在 Isaac Sim 窗口中观察机器人是否移动
3. 如果修复成功 → 可以开始正式长训练 `./wl_train.sh full`
4. 如果仍有问题 → 尝试备选方案或联系下一轮 Claude

### 后续优化（可选）
- 修改实验目录名（cartpole_direct 残留）
- 将特权观测（base_lin_vel）换成 IMU 可测量值
- 调整奖励函数（加熵奖励、微调权重）
- 增加 curriculum 难度梯度

## 技术要点回顾

**关键发现**:
- IsaacLab 的 Articulation 在 Fabric 模式下自动同步 visual/physics
- 禁用 Fabric 后，visual mesh 的 USD transform 不会自动更新
- `write_root_pose_to_sim(root_pose)` 可以显式写回物理位姿到 USD

**调试技巧**:
- 速度指令箭头用 VisualizationMarkers 绘制在 `root_pos_w`，它的移动证明物理正常
- 对比箭头（动）和机器人本体（不动）可以确认是视觉渲染问题，不是物理问题
- 日志 `[DBG]` 数据显示 `pos_w` 变化，进一步证明物理仿真正确

## 本轮状态

✅ **完成**: 代码修复实施、自动化测试脚本创建、文档更新  
🔧 **待验证**: 修复是否有效（需在 GUI 中实际运行）  
⏸️ **暂停**: 正式长训练（等视觉问题确认后再开始）

---

**交接给**: 下一轮 Claude 或用户自己验证  
**预计验证时间**: 5 分钟（运行 wl_autotest.sh 并观察）  
**文档版本**: 2026-07-19 23:40
