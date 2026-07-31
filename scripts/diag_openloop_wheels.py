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
# 主轮接触力传感器(prim wheel.$),用于判断主轮是否着地
contact = uenv.scene.sensors.get("wheel_contacts", None)

# 强制轮动作(raw,乘10后即目标rad/s)。指令恒0。use_policy=False 时连腿也全0(纯无控)。
# (raw_L, raw_R, use_policy_legs, 标签)
W = 0.6  # raw 0.6 -> 目标 6 rad/s
PHASES = [
    ( 0,  0, False, "全0无控   (腿轮都0,测能否自立)"),
    ( W,  W, True,  "同号++    (目标+6/+6,腿=策略)"),
    (-W, -W, True,  "同号--    (目标-6/-6,腿=策略)"),
    ( W, -W, True,  "反号+-    (目标+6/-6,腿=策略)"),
    (-W,  W, True,  "反号-+    (目标-6/+6,腿=策略)"),
]

def force_cmd0():
    cmd_term.vel_command_b[:, 0] = 0.0
    cmd_term.vel_command_b[:, 1] = 0.0
    cmd_term.vel_command_b[:, 2] = 0.0

def wheel_contact_forces():
    if contact is None:
        return float("nan"), float("nan")
    # net_forces_w: (N_env, N_bodies, 3) -> 取模长, 前两个 body 即 wheel1/wheel2
    f = contact.data.net_forces_w[0]  # (N_bodies,3)
    mags = torch.linalg.norm(f, dim=-1).cpu().numpy()
    a = float(mags[0]) if mags.shape[0] > 0 else float("nan")
    b = float(mags[1]) if mags.shape[0] > 1 else float("nan")
    return a, b

print("\n===== 开环轮子测试开始 (指令恒0,强制轮速) =====")
print(f"轮半径参考~0.036m; 目标6rad/s单轮正常滚动应~0.216m/s")
print(f"init 高度=0.15m; projected_gravity_z≈-1 表示直立, 接近0/正表示倾倒\n")
for (rl, rr, use_policy, label) in PHASES:
    force_cmd0()
    p0 = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
    vx_hist, wz_hist, wl_hist, wr_hist = [], [], [], []
    z_hist, gz_hist, cfl_hist, cfr_hist = [], [], [], []
    for i in range(args.hold):
        force_cmd0()
        fresh = uenv.observation_manager.compute()["policy"]
        with torch.inference_mode():
            out = runner.agent.act(fresh, None, timestep=0, timesteps=0)
            raw = out[-1].get("mean_actions", out[0])
        actions = torch.zeros_like(raw)
        if use_policy:
            actions.copy_(raw)          # 腿用策略
            actions[:, 4] = rl          # 覆写两轮
            actions[:, 5] = rr
        # use_policy=False 时 actions 全 0(含腿),纯无控自立测试
        env.step(actions)
        if i >= args.hold - 40:
            vx_hist.append(robot.data.root_lin_vel_b[0, 0].item())
            wz_hist.append(robot.data.root_ang_vel_b[0, 2].item())
            wl_hist.append(robot.data.joint_vel[0, wheel_l_idx].item())
            wr_hist.append(robot.data.joint_vel[0, wheel_r_idx].item())
            z_hist.append(robot.data.root_pos_w[0, 2].item())
            gz_hist.append(robot.data.projected_gravity_b[0, 2].item())
            cf = wheel_contact_forces()
            cfl_hist.append(cf[0]); cfr_hist.append(cf[1])
    p1 = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
    dp = p1 - p0
    dist = float(np.hypot(dp[0], dp[1]))
    print(f"[{label}]")
    print(f"   净位移 dx={dp[0]:+.3f} dy={dp[1]:+.3f} 距离={dist:.3f}m | "
          f"高度z={np.mean(z_hist):.3f} 倾倒gz={np.mean(gz_hist):+.2f}")
    print(f"   body vx={np.mean(vx_hist):+.3f} wz={np.mean(wz_hist):+.3f} | "
          f"轮速 L={np.mean(wl_hist):+7.2f} R={np.mean(wr_hist):+7.2f} | "
          f"主轮接触力 L={np.nanmean(cfl_hist):6.1f} R={np.nanmean(cfr_hist):6.1f} N\n")
print("===== 开环轮子测试结束 =====")
print("判读关键:")
print("  1) 全0无控相位:若 z 掉/gz→0 → 机器人静态不稳,需策略平衡(倒立摆型)")
print("  2) 主轮接触力≈0 → 主轮离地空转,前进推力无从谈起(查腿姿/轮位)")
print("  3) 有接触力但轮速不跟指令(命令0却转) → 阻尼太弱压不住,或被拖行\n")
simulation_app.close()
