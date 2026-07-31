#!/bin/bash
# 验证清单 - 在运行正式训练前确认所有功能正常

echo "========================================"
echo " WheelLeg 项目验证清单"
echo "========================================"
echo ""

echo "1. 检查 checkpoint 文件..."
CKPT="logs/skrl/cartpole_direct/2026-07-19_20-51-22_ppo_torch/checkpoints/agent_96000.pt"
if [ -f "$CKPT" ]; then
    echo "   ✓ checkpoint 存在: $(ls -lh $CKPT | awk '{print $5}')"
else
    echo "   ✗ checkpoint 不存在!"
    exit 1
fi

echo ""
echo "2. 检查可用脚本..."
for script in wl_play.sh wl_teleop.sh wl_train.sh wl_autotest.sh; do
    if [ -x "$script" ]; then
        echo "   ✓ $script"
    else
        echo "   ✗ $script (不存在或不可执行)"
    fi
done

echo ""
echo "3. 检查内存状态..."
MEM_FREE=$(free -h | grep Mem | awk '{print $4}')
echo "   可用内存: $MEM_FREE"
ISAAC_PROC=$(ps aux | grep -E "isaac|omni" | grep -v grep | wc -l)
if [ "$ISAAC_PROC" -gt 0 ]; then
    echo "   ⚠ 检测到 $ISAAC_PROC 个 Isaac Sim 进程在运行"
    echo "     建议执行: pkill -9 -f 'isaac|omni|play'"
else
    echo "   ✓ 无残留 Isaac Sim 进程"
fi

echo ""
echo "4. 检查文档..."
for doc in HANDOFF.md QUICK_START.md SESSION_SUMMARY.md; do
    if [ -f "$doc" ]; then
        echo "   ✓ $doc"
    else
        echo "   ✗ $doc"
    fi
done

echo ""
echo "========================================"
echo " 下一步操作"
echo "========================================"
echo ""
echo "【必做】验证视觉修复是否有效:"
echo "  ./wl_autotest.sh"
echo "  → 在 Isaac Sim 窗口观察机器人本体是否移动"
echo ""
echo "【可选】手动键盘遥控测试:"
echo "  ./wl_teleop.sh"
echo "  → 用 W/S/Q/E 测试前进/后退/转向"
echo ""
echo "【最终】开始正式训练 (10-15小时):"
echo "  ./wl_train.sh full"
echo ""
echo "========================================"
