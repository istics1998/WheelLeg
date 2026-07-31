"""诊断:正式档策略为何 play 时静止。
加载 checkpoint,跑若干步,打印 指令速度 / 策略动作 / 实际 base 线速度+角速度 / 轮子转速。
判据:
  - 动作≈0 且 base_vel≈0 → 策略躺平(局部最优,需调奖励/课程)
  - 动作有值但 base_vel≈0 → 机构走不动(执行器/机构问题)
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=60)
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
# 用与训练相同的 skrl 配置
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
robot = env.unwrapped.scene["robot"]
cmd_mgr = env.unwrapped.command_manager

print("\n===== 诊断开始 =====")
for i in range(args.steps):
    with torch.inference_mode():
        out = runner.agent.act(obs, env.state() if hasattr(env, "state") else None,
                               timestep=0, timesteps=0)
        actions = out[-1].get("mean_actions", out[0])
        obs, _, _, _, _ = env.step(actions)
    if i % 10 == 0 or i == args.steps - 1:
        cmd = cmd_mgr.get_command("base_velocity")[0].cpu().numpy()
        lin = robot.data.root_lin_vel_b[0].cpu().numpy()
        ang = robot.data.root_ang_vel_b[0].cpu().numpy()
        act = actions[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        print(f"步{i:3d} | 指令[vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}] "
              f"| 动作{['%+.2f'%a for a in act]} "
              f"| base实际[vx={lin[0]:+.2f} vy={lin[1]:+.2f} wz={ang[2]:+.2f}] "
              f"| 轮速max={abs(jv).max():.1f}")
print("===== 诊断结束 =====\n")
simulation_app.close()
