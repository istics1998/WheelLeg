#!/usr/bin/env python3
"""
自动测试机器人视觉是否跟随物理移动。
不依赖键盘输入，直接注入速度指令并监控位置变化。
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="测试机器人视觉移动")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="WheelLeg-Flat-v0")
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import os
from packaging import version
import skrl
from skrl.utils.runner.torch import Runner
import isaaclab_tasks
import WheelLeg.tasks
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

def main():
    print("\n" + "=" * 70)
    print("自动视觉移动测试")
    print(f"Fabric 模式: {'禁用' if args_cli.disable_fabric else '启用'}")
    print("=" * 70 + "\n")

    # 环境配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # 禁用指令自动重采样
    try:
        vel_cmd = env_cfg.commands.base_velocity
        vel_cmd.resampling_time_range = (1.0e9, 1.0e9)
        vel_cmd.heading_command = False
        vel_cmd.rel_standing_envs = 0.0
    except AttributeError:
        pass

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # 加载checkpoint
    task_name = args_cli.task.split(":")[-1]
    try:
        experiment_cfg = load_cfg_from_registry(task_name, "skrl_ppo_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]))
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, run_dir=f".*_ppo_torch", other_dirs=["checkpoints"])

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] 加载 checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    base_env = env.unwrapped
    cmd_term = base_env.command_manager.get_term("base_velocity")
    robot = base_env.scene["robot"]

    # Reset
    obs, _ = env.reset()
    pos_init = robot.data.root_pos_w[0].clone()
    print(f"[INFO] 初始位置: x={pos_init[0]:.3f}, y={pos_init[1]:.3f}, z={pos_init[2]:.3f}\n")

    # 测试1: 零指令下平衡 (50步)
    print("=== 测试1: 零指令平衡 (50步) ===")
    cmd_term.vel_command_b[:] = torch.tensor([0.0, 0.0, 0.0], device=base_env.device)
    for step in range(50):
        if version.parse(skrl.__version__) >= version.parse("2.0.0"):
            states = env.state() if hasattr(env, "state") else None
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
        else:
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, _, _, _, _ = env.step(actions)

        # 在disable_fabric模式下尝试强制更新视觉
        if args_cli.disable_fabric and hasattr(robot, 'write_root_pose_to_sim'):
            root_pose = torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=-1)
            robot.write_root_pose_to_sim(root_pose)

    pos_after_balance = robot.data.root_pos_w[0].clone()
    drift = (pos_after_balance - pos_init).norm().item()
    print(f"位置漂移: {drift:.4f} m (零指令下应该很小)")

    # 测试2: 前进指令 vx=1.0 (200步)
    print("\n=== 测试2: 前进指令 vx=1.0 (200步) ===")
    cmd_term.vel_command_b[:] = torch.tensor([1.0, 0.0, 0.0], device=base_env.device)
    pos_before = robot.data.root_pos_w[0].clone()

    for step in range(200):
        if version.parse(skrl.__version__) >= version.parse("2.0.0"):
            states = env.state() if hasattr(env, "state") else None
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
        else:
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, _, _, _, _ = env.step(actions)

        # 在disable_fabric模式下尝试强制更新视觉
        if args_cli.disable_fabric and hasattr(robot, 'write_root_pose_to_sim'):
            root_pose = torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=-1)
            robot.write_root_pose_to_sim(root_pose)

        if step % 50 == 0:
            pos = robot.data.root_pos_w[0]
            vel = robot.data.root_lin_vel_b[0]
            print(f"  Step {step:3d}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) vel_x={vel[0]:.2f}")

    pos_after = robot.data.root_pos_w[0].clone()
    movement = (pos_after - pos_before).norm().item()
    print(f"\n总移动距离: {movement:.3f} m")

    # 判断结果
    print("\n" + "=" * 70)
    print("测试结果:")
    if movement > 1.0:
        print(f"✓ 物理正常: 机器人移动了 {movement:.2f} m")
        print(f"  Fabric模式={'禁用' if args_cli.disable_fabric else '启用'}")
        if args_cli.disable_fabric:
            print("  已尝试调用 write_root_pose_to_sim() 强制更新视觉")
        print("\n请在Isaac Sim窗口中观察:")
        print("  1. 机器人本体是否在画面中移动?")
        print("  2. 速度指令箭头(绿/蓝)是否在移动?")
        print("  如果箭头动但机器人不动 → 视觉与物理脱节")
        print("  如果都动 → 问题已解决!")
    else:
        print(f"✗ 物理异常: 机器人仅移动了 {movement:.3f} m (预期 >1.0m)")
    print("=" * 70 + "\n")

    print("窗口保持打开，按 Ctrl+C 退出...")
    try:
        while simulation_app.is_running():
            env.step(actions)
    except KeyboardInterrupt:
        pass

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
