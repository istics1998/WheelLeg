#!/usr/bin/env bash
# WheelLeg 键盘遥控可视化脚本（双击桌面图标调用）
# 焦点放在 Isaac Sim 视口窗口上，用方向键/小键盘控制机器人：
#   ↑/↓ 前进后退   ←/→ 左右平移   Z/X 左右转向   松开=原地平衡
set -e

WHEELLEG_DIR="/home/ist/桌面/Isaac_project/WheelLeg"
ISAACLAB_SH="/home/ist/IsaacLab/isaaclab.sh"
CONDA_ENV="isaaclab45"

# 加载 conda（桌面双击时不是登录shell，需手动 source）
source /home/ist/miniconda3/etc/profile.d/conda.sh

cd "$WHEELLEG_DIR"

# checkpoint 选择:
# 默认指向最新验证干净的训练结果(2026-07-19 晚,加了摔倒终止后首次成功训练,9.6万步)。
# 需要换模型时改这里,或用 WL_CKPT 环境变量覆盖: WL_CKPT=/path/to/xxx.pt ./wl_teleop.sh
CKPT="${WL_CKPT:-$WHEELLEG_DIR/logs/skrl/cartpole_direct/2026-07-19_20-51-22_ppo_torch/checkpoints/agent_96000.pt}"

if [ ! -f "$CKPT" ]; then
    echo "❌ 找不到 checkpoint: $CKPT"
    read -p "按回车键关闭..."
    exit 1
fi

echo "======================================"
echo " WheelLeg 键盘遥控 (锁存式:按一下加一档并保持)"
echo " checkpoint: $CKPT"
echo " 键位: W/S前后  Q/E转向  (A/D横移未训练)  空格=停"
echo "======================================"

# 注意:不加 --disable_fabric。GPU 仿真下必须走 fabric 原生渲染同步(PhysX->Fabric->Hydra),
# 这是 IsaacLab 所有任务显示机器人移动的标准路径。之前加 --disable_fabric 是基于已被推翻的
# SkelMesh 误诊;而 GPU 下手动写 USD 位姿会触发非法 setGlobalPose(eENABLE_DIRECT_GPU_API)刷屏报错。
conda run -n "$CONDA_ENV" --no-capture-output "$ISAACLAB_SH" -p scripts/skrl/play_teleop.py \
    --task Template-Wheelleg-v0 \
    --num_envs 1 \
    --rendering_mode performance \
    --checkpoint "$CKPT"

echo ""
echo "运行结束。"
read -p "按回车键关闭..."
