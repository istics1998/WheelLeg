# WheelLeg 项目文档索引

## 📚 文档位置

所有文档已整理到 `docs/` 目录下。

### 主要文档（项目根目录）

- **HANDOFF.md** - 完整技术历史和问题排查记录
- **QUICK_START.md** - 快速使用指南
- **SESSION_SUMMARY.md** - 第二轮详细工作记录
- **README_THIS_SESSION.md** - 本轮快速摘要

### 桌面文档（docs/目录）

#### 交接文档 ⭐ 重要
- **【交接】给下一轮Claude.txt** - 给下一轮 Claude 的完整交接文档
  - 核心问题描述
  - 已尝试方案（避免重复）
  - 建议解决方向
  - 项目完整状态

#### 工作总结
- **【完成】第二轮工作总结.txt** - 第二轮完整工作记录
- **【必读】项目状态.txt** - 项目当前状态总览

#### 问题分析
- **【当前状态】视觉问题.txt** - 视觉脱节问题状态
- **【最终方案】接受现状.txt** - 问题分析和影响评估
- **【备选方案】视觉修复.txt** - 备选解决方案

#### 使用说明
- **【使用说明】WheelLeg快捷方式.txt** - 桌面快捷方式使用指南

### Memory 文件（持久化记录）

位置: `~/.claude/projects/-home-ist----Isaac-project-WheelLeg/memory/`

- **MEMORY.md** - 索引文件
- **skrl-version-api.md** - skrl 版本兼容性
- **machine-resource-limits.md** - 机器资源限制
- **visualization-setup.md** - 可视化设置
- **diverged-checkpoint.md** - 训练发散问题
- **visual-sync-fix-attempt.md** - 视觉同步修复尝试

## 🎯 快速导航

### 新会话开始
→ 阅读 `docs/【交接】给下一轮Claude.txt`

### 了解项目历史
→ 阅读 `HANDOFF.md`

### 快速上手使用
→ 阅读 `QUICK_START.md`

### 查看详细工作记录
→ 阅读 `SESSION_SUMMARY.md`

### 了解视觉问题
→ 阅读 `docs/【最终方案】接受现状.txt`

## 📋 桌面快捷方式

桌面上保留了所有脚本快捷方式：
- WheelLeg 键盘遥控.desktop
- WheelLeg 可视化.desktop
- WheelLeg 自动测试.desktop
- WheelLeg 测试(Fabric模式).desktop
- 轮足平衡演示.desktop

## 🔧 项目脚本

项目根目录：
- `wl_train.sh` - 训练脚本
- `wl_play.sh` - 可视化回放
- `wl_teleop.sh` - 键盘遥控
- `wl_autotest.sh` - 自动化测试
- `wl_autotest_fabric.sh` - Fabric 模式测试

---

更新时间: 2026-07-20 01:15
状态: 文档已整理，桌面保留快捷方式
