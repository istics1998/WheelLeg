"""增强诊断:强制轮换多条速度指令,验证策略能否真正按指令运动(而非只测到原地转)。
- 强制指令序列:前进/后退/原地左转/原地右转/停,每条持续若干步给足 settling。
- 单独拆出两个驱动轮(g6_l_to_wheel1 / g6_r_to_wheel2)转速,区分"真驱动"vs"空转打滑"。
- 每条指令末尾统计 base 实际 vx/wz 相对指令的跟踪误差。
判据:
  - 各相位 base 实际速度朝指令方向且量级接近 → 策略会走 ✅
  - 前进指令下 vx≈0 但轮子高速转 → 打滑/空转(机构或摩擦问题)
  - base 实际与指令反向或乱 → 策略仍没学会
注意:heading_command=True 会在 command_manager.compute 里用航向误差重算 wz,
所以本脚本每步在 env.step 之后强制覆写 vel_command_b,并用 observation_manager 重算 obs,
确保策略看到的是我们强制的指令。
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--hold", type=int, default=80, help="每条指令持续步数(*0.01s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from skrl.utils.runner.torch import Runner
import WheelLeg.tasks  # noqa: F401 注册任务

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

# 定位两个驱动轮关节索引
wheel_l_idx = robot.find_joints("g6_l_to_wheel1")[0]
wheel_r_idx = robot.find_joints("g6_r_to_wheel2")[0]

# 强制指令序列: (vx, vy, wz, 标签)
PHASES = [
    (0.5, 0.0, 0.0, "前进 vx=+0.5"),
    (-0.5, 0.0, 0.0, "后退 vx=-0.5"),
    (0.0, 0.0, 1.57, "原地左转 wz=+1.57"),
    (0.0, 0.0, -1.57, "原地右转 wz=-1.57"),
    (0.0, 0.0, 0.0, "停 全0"),
]

def force_cmd(vx, vy, wz):
    cmd_term.vel_command_b[:, 0] = vx
    cmd_term.vel_command_b[:, 1] = vy
    cmd_term.vel_command_b[:, 2] = wz

print("\n===== 增强诊断开始 =====")
for (vx, vy, wz, label) in PHASES:
    # 采集本相位末尾若干步的实际状态求均值(避开切换瞬态)
    lin_hist, ang_hist, wl_hist, wr_hist = [], [], [], []
    for i in range(args.hold):
        force_cmd(vx, vy, wz)
        fresh = uenv.observation_manager.compute()["policy"]
        with torch.inference_mode():
            out = runner.agent.act(fresh, None, timestep=0, timesteps=0)
            actions = out[-1].get("mean_actions", out[0])
            env.step(actions)
        if i >= args.hold - 30:  # 末 30 步取均值
            lin = robot.data.root_lin_vel_b[0].cpu()
            ang = robot.data.root_ang_vel_b[0].cpu()
            jv = robot.data.joint_vel[0].cpu()
            lin_hist.append(lin[0].item())
            ang_hist.append(ang[2].item())
            wl_hist.append(jv[wheel_l_idx].item())
            wr_hist.append(jv[wheel_r_idx].item())
    import numpy as np
    vx_m = np.mean(lin_hist); wz_m = np.mean(ang_hist)
    wl_m = np.mean(wl_hist); wr_m = np.mean(wr_hist)
    # 跟踪判定
    if abs(vx) > 0.01:
        track = f"vx跟踪: 指令{vx:+.2f} 实际{vx_m:+.2f} 误差{abs(vx-vx_m):.2f}"
    elif abs(wz) > 0.01:
        track = f"wz跟踪: 指令{wz:+.2f} 实际{wz_m:+.2f} 误差{abs(wz-wz_m):.2f}"
    else:
        track = f"停: 实际vx{vx_m:+.2f} wz{wz_m:+.2f} (应≈0)"
    print(f"[{label:18s}] base实际 vx={vx_m:+.3f} wz={wz_m:+.3f} | "
          f"驱动轮速 L={wl_m:+7.2f} R={wr_m:+7.2f} rad/s | {track}")
print("===== 增强诊断结束 =====\n")
simulation_app.close()
