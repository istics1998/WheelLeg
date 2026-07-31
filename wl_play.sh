#!/usr/bin/env bash
# WheelLeg 可视化运行脚本（双击桌面图标调用）
set -e

WHEELLEG_DIR="/home/ist/桌面/Isaac_project/WheelLeg"
ISAACLAB_SH="/home/ist/IsaacLab/isaaclab.sh"
CONDA_ENV="isaaclab45"

# 加载 conda（桌面双击时不是登录shell，需手动 source）
source /home/ist/miniconda3/etc/profile.d/conda.sh

cd "$WHEELLEG_DIR"

# checkpoint 选择:
# 默认指向 nan 发散修复后的正式档结果(2026-07-26,192万步跑满,11 checkpoint 全 CLEAN)。
# 修复内容:solver_velocity_iteration_count 0→4(闭环 mimic 需速度层收敛)+
# max_angular_velocity 100→50rad/s(砍连杆病态飙升)。全程 nan=0。
# 注意:nan 已解,但策略仍陷"站着不动"局部最优(play 静止),待调奖励/课程后重训。详见 START_HERE.md。
# 需要换模型时改这里,或用 WL_CKPT 环境变量覆盖: WL_CKPT=/path/to/xxx.pt ./wl_play.sh
CKPT="${WL_CKPT:-logs/skrl/wheelleg/2026-07-26_04-22-08_ppo_torch/checkpoints/best_agent.pt}"

if [ ! -f "$CKPT" ]; then
    echo "❌ 找不到 checkpoint: $CKPT"
    read -p "按回车键关闭..."
    exit 1
fi

echo "======================================"
echo " WheelLeg 可视化运行"
echo " checkpoint: $CKPT"
echo "======================================"

conda run -n "$CONDA_ENV" --no-capture-output "$ISAACLAB_SH" -p scripts/skrl/play.py \
    --task Template-Wheelleg-v0 \
    --num_envs 1 \
    --rendering_mode performance \
    --disable_fabric \
    --checkpoint "$CKPT"

echo ""
echo "运行结束。"
read -p "按回车键关闭..."
