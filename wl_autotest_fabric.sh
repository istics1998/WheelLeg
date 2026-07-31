#!/usr/bin/env bash
# WheelLeg 自动化视觉移动测试 - Fabric 模式
# 尝试用 Fabric 自动同步解决视觉脱节问题
set -e

WHEELLEG_DIR="/home/ist/桌面/Isaac_project/WheelLeg"
ISAACLAB_SH="/home/ist/IsaacLab/isaaclab.sh"
CONDA_ENV="isaaclab45"

# 加载 conda
source /home/ist/miniconda3/etc/profile.d/conda.sh

cd "$WHEELLEG_DIR"

# checkpoint 选择
CKPT="${WL_CKPT:-$WHEELLEG_DIR/logs/skrl/cartpole_direct/2026-07-19_20-51-22_ppo_torch/checkpoints/agent_96000.pt}"

if [ ! -f "$CKPT" ]; then
    echo "❌ 找不到 checkpoint: $CKPT"
    read -p "按回车键关闭..."
    exit 1
fi

echo "======================================"
echo " WheelLeg 自动测试 (Fabric 模式)"
echo " checkpoint: $CKPT"
echo " 测试: 启用 Fabric 自动同步"
echo " 观察机器人本体是否移动!"
echo "======================================"

# 注意：去掉了 --disable_fabric 参数
conda run -n "$CONDA_ENV" --no-capture-output "$ISAACLAB_SH" -p scripts/skrl/play_teleop_autotest.py \
    --task Template-Wheelleg-v0 \
    --num_envs 1 \
    --rendering_mode performance \
    --checkpoint "$CKPT"

echo ""
echo "测试完成。"
read -p "按回车键关闭..."
