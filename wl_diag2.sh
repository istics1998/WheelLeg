#!/usr/bin/env bash
# WheelLeg 策略运动诊断(diag2):强制多条速度指令,验证策略能否真正按指令运动。
# 用法:
#   ./wl_diag2.sh                 # 自动选最新一次训练的 best_agent.pt
#   ./wl_diag2.sh <checkpoint>    # 指定 checkpoint
#   HOLD=120 ./wl_diag2.sh        # 每条指令持续步数(默认 80)
# ⚠ 同时只能开一个 Isaac Sim,别和训练/play 并跑,否则 OOM。
set -e

WHEELLEG_DIR="/home/ist/桌面/Isaac_project/WheelLeg"
ISAACLAB_SH="/home/ist/IsaacLab/isaaclab.sh"
CONDA_ENV="isaaclab45"
HOLD="${HOLD:-80}"

source /home/ist/miniconda3/etc/profile.d/conda.sh
cd "$WHEELLEG_DIR"

CKPT="${1:-}"
if [ -z "$CKPT" ]; then
    LATEST_DIR=$(ls -td logs/skrl/wheelleg/*/ 2>/dev/null | head -1)
    CKPT="${LATEST_DIR}checkpoints/best_agent.pt"
fi

if [ ! -f "$CKPT" ]; then
    echo "找不到 checkpoint: $CKPT"
    exit 1
fi

echo "======================================"
echo " WheelLeg 策略诊断 diag2"
echo " checkpoint = $CKPT"
echo " hold = $HOLD 步/指令"
echo "======================================"

conda run -n "$CONDA_ENV" --no-capture-output "$ISAACLAB_SH" -p scripts/diag_policy_motion2.py \
    --checkpoint "$CKPT" --hold "$HOLD"
