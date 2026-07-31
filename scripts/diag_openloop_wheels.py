"""开环轮子测试:绕开策略前进决策,直接强制两驱动轮固定速度,测车身能否平移。
目的:区分"前进不动"是机构/物理问题还是策略学习问题。
- 策略照常输出(维持平衡),但每步覆写 wheel_l/wheel_r 两个动作分量为固定值。
- 指令恒为 0(策略只想站住),纯看强制轮速能否推动 base 沿 x 平移。
- 记录每相位 base 世界坐标净位移 dx/dy + body系 vx/wz。
关键对照:两驱动轮镜像轴(左+Y/右-Y),物理上直线前进需关节速度反号。
  若"反号"相位车真前进而"同号"相位原地转 → 机构没问题,是策略没学会反号(学习问题)。
  若两种都推不动车(dx≈0) → 机构/摩擦/离地问题。
动作顺序: [twist_l, twist_r, knee_l, knee_r, wheel_l, wheel_r] = idx 0..5
wheel 动作 JointVelocityActionCfg scale=10 clip±10 → 目标轮速(rad/s)=clip(raw*10,±10)
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--hold", type=int, default=120, help="每条指令持续步数(*0.01s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from skrl.utils.runner.torch import Runner
import WheelLeg.tasks  # noqa: F401

TASK = "Template-Wheelleg-v0"
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)

import os, yaml
cfg_path = os.path.join(os.path.dirname(__file__), "..",
    "source/WheelLeg/WheelLeg/tasks/manager_based/wheelleg/agents/skrl_ppo_cfg.yaml")
with open(cfg_path) as f:
    agent_cfg = yaml.safe_load(f)

from isaaclab_rl.skrl import SkrlVecEnvWrapper
env = gym.make(TASK, cfg=env_cfg)
env = SkrlVecEnvWrapper(env)
runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
if hasattr(runner.agent, "set_running_mode"):
    runner.agent.set_running_mode("eval")

obs, _ = env.reset()
uenv = env.unwrapped
robot = uenv.scene["robot"]
cmd_term = uenv.command_manager.get_term("base_velocity")
wheel_l_idx = robot.find_joints("g6_l_to_wheel1")[0][0]
wheel_r_idx = robot.find_joints("g6_r_to_wheel2")[0][0]

# 强制轮动作(raw,乘10后即目标rad/s)。指令恒0。
# (raw_L, raw_R, 标签)
W = 0.6  # raw 0.6 -> 目标 6 rad/s
PHASES = [
    ( W,  W, "同号++  (raw+0.6/+0.6 -> 目标+6/+6)"),
    (-W, -W, "同号--  (raw-0.6/-0.6 -> 目标-6/-6)"),
    ( W, -W, "反号+-  (raw+0.6/-0.6 -> 目标+6/-6)"),
    (-W,  W, "反号-+  (raw-0.6/+0.6 -> 目标-6/+6)"),
    ( 0,  0, "轮停0   (纯站立参照)"),
]

def force_cmd0():
    cmd_term.vel_command_b[:, 0] = 0.0
    cmd_term.vel_command_b[:, 1] = 0.0
    cmd_term.vel_command_b[:, 2] = 0.0

print("\n===== 开环轮子测试开始 (指令恒0,强制轮速) =====")
print(f"轮半径参考~0.036m; 目标6rad/s单轮正常滚动应~0.216m/s\n")
for (rl, rr, label) in PHASES:
    force_cmd0()
    p0 = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
    vx_hist, wz_hist, wl_hist, wr_hist = [], [], [], []
    for i in range(args.hold):
        force_cmd0()
        fresh = uenv.observation_manager.compute()["policy"]
        with torch.inference_mode():
            out = runner.agent.act(fresh, None, timestep=0, timesteps=0)
            actions = out[-1].get("mean_actions", out[0]).clone()
        # 覆写两个轮动作分量(idx 4,5)
        actions[:, 4] = rl
        actions[:, 5] = rr
        env.step(actions)
        if i >= args.hold - 40:
            vx_hist.append(robot.data.root_lin_vel_b[0, 0].item())
            wz_hist.append(robot.data.root_ang_vel_b[0, 2].item())
            wl_hist.append(robot.data.joint_vel[0, wheel_l_idx].item())
            wr_hist.append(robot.data.joint_vel[0, wheel_r_idx].item())
    p1 = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
    dp = p1 - p0
    dist = float(np.hypot(dp[0], dp[1]))
    print(f"[{label}]")
    print(f"   净位移 dx={dp[0]:+.3f} dy={dp[1]:+.3f} 直线距离={dist:.3f} m  (over {args.hold*0.01:.1f}s)")
    print(f"   body系 vx={np.mean(vx_hist):+.3f} wz={np.mean(wz_hist):+.3f} | "
          f"实际轮速 L={np.mean(wl_hist):+7.2f} R={np.mean(wr_hist):+7.2f} rad/s\n")
print("===== 开环轮子测试结束 =====")
print("判读: 若某'反号'相位 dist 明显>0 而'同号'相位 dist≈0 → 机构正常,策略需学反号;")
print("      若所有相位 dist 都≈0(轮子空转) → 机构/摩擦/离地问题,需查装配。\n")
simulation_app.close()
