#!/usr/bin/env bash
# WheelLeg 训练脚本
# 用法:
#   ./wl_train.sh            # 验证档:num_envs=1024, ~10万步(约3000迭代),先确认不再发散
#   ./wl_train.sh full       # 正式档:num_envs=1024, ~200万步(约60000迭代)
#   WL_ENVS=2048 ./wl_train.sh full   # 自定义环境数
set -e

WHEELLEG_DIR="/home/ist/桌面/Isaac_project/WheelLeg"
ISAACLAB_SH="/home/ist/IsaacLab/isaaclab.sh"
CONDA_ENV="isaaclab45"

source /home/ist/miniconda3/etc/profile.d/conda.sh
cd "$WHEELLEG_DIR"

# 完成标记文件:训练结束时写入结果,供外部检测提醒
MARKER="$WHEELLEG_DIR/logs/skrl/.wl_train_result"
rm -f "$MARKER"

# 环境数与迭代数(rollouts=32,故 timesteps = ITERS * 32)
MODE="${1:-verify}"
ENVS="${WL_ENVS:-1024}"
if [ "$MODE" = "full" ]; then
    ITERS="${WL_ITERS:-60000}"   # ~192万步
else
    ITERS="${WL_ITERS:-3000}"    # ~9.6万步,快速验证是否发散
fi

echo "======================================"
echo " WheelLeg 训练  [$MODE]"
echo " num_envs = $ENVS   max_iterations = $ITERS  (约 $((ITERS*32)) 步)"
echo " headless 无渲染训练。Ctrl-C 可随时中断。"
echo "======================================"

set +e
conda run -n "$CONDA_ENV" --no-capture-output "$ISAACLAB_SH" -p scripts/skrl/train.py \
    --task Template-Wheelleg-v0 \
    --headless \
    --num_envs "$ENVS" \
    --max_iterations "$ITERS"
TRAIN_RC=$?
set -e

# 训练结束后,自动检测最新 checkpoint 是否发散(全 nan)
RESULT="unknown"
NEWEST=$(find "$WHEELLEG_DIR/logs/skrl" -name "*.pt" -newer "$0" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$NEWEST" ]; then
    RESULT=$(conda run -n "$CONDA_ENV" python - "$NEWEST" <<'PY'
import sys, torch
f = sys.argv[1]
try:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    pol = ck.get("policy", {})
    nan = sum(int(torch.isnan(v).sum()) for v in pol.values() if hasattr(v,"is_floating_point") and v.is_floating_point())
    tot = sum(v.numel() for v in pol.values() if hasattr(v,"is_floating_point") and v.is_floating_point())
    print("DIVERGED_NAN" if nan==tot and tot>0 else ("PARTIAL_NAN" if nan>0 else "CLEAN"))
except Exception as e:
    print(f"CHECK_ERROR:{e}")
PY
)
fi

echo "$MODE|rc=$TRAIN_RC|result=$RESULT|ckpt=$NEWEST" > "$MARKER"

echo ""
echo "======================================"
echo " 训练结束。退出码=$TRAIN_RC  权重检测=$RESULT"
echo " 最新 checkpoint: $NEWEST"
echo "======================================"
read -p "按回车键关闭..."
