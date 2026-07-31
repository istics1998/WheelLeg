#!/usr/bin/env bash
# WheelLeg 自动化视觉移动测试
# 无需键盘操作，自动执行测试序列验证视觉修复是否有效
set -e

WHEELLEG_DIR="/home/ist/桌面/Isaac_project/WheelLeg"
ISAACLAB_SH="/home/ist/IsaacLab/isaaclab.sh"
CONDA_ENV="isaaclab45"

# 加载 conda（桌面双击时不是登录shell，需手动 source）
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
echo " WheelLeg 自动化视觉移动测试"
echo " checkpoint: $CKPT"
echo " 测试序列: 平衡→前进→停止→左转"
echo " 请在窗口中观察机器人是否移动!"
echo "======================================"

conda run -n "$CONDA_ENV" --no-capture-output "$ISAACLAB_SH" -p scripts/skrl/play_teleop_autotest.py \
    --task Template-Wheelleg-v0 \
    --num_envs 1 \
    --rendering_mode performance \
    --disable_fabric \
    --checkpoint "$CKPT"

echo ""
echo "测试完成。"
read -p "按回车键关闭..."
