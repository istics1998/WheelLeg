#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved. SPDX-License-Identifier: BSD-3-Clause

"""
自动化测试版 play_teleop: 无需键盘，自动注入速度指令测试机器人移动。
用于验证视觉是否跟随物理移动。

测试序列:
  1. 零指令平衡 50步
  2. 前进 vx=1.0 持续 200步
  3. 停止 50步
  4. 左转 wz=1.0 持续 100步
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="自动测试机器人视觉移动")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import os
import torch
import skrl
from packaging import version

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
import WheelLeg.tasks

algorithm = args_cli.algorithm.lower()

def main():
    print("\n" + "=" * 70)
    print("自动化视觉移动测试")
    print(f"Fabric 模式: {'禁用 (--disable_fabric)' if args_cli.disable_fabric else '启用'}")
    print("=" * 70)
    print("测试序列:")
    print("  1. 零指令平衡 50步")
    print("  2. 前进 vx=1.0 持续 200步")
    print("  3. 停止 50步")
    print("  4. 左转 wz=1.0 持续 100步")
    print("请在 Isaac Sim 窗口中观察机器人本体是否移动!")
    print("=" * 70 + "\n")

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    try:
        experiment_cfg = load_cfg_from_registry(task_name, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")

    # 禁用指令自动重采样
    try:
        vel_cmd = env_cfg.commands.base_velocity
        vel_cmd.resampling_time_range = (1.0e9, 1.0e9)
        vel_cmd.heading_command = False
        vel_cmd.rel_standing_envs = 0.0
        vel_cmd.rel_heading_envs = 0.0
    except AttributeError:
        pass

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] 加载 checkpoint: {resume_path}\n")
    runner.agent.load(resume_path)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    base_env = env.unwrapped
    cmd_term = base_env.command_manager.get_term("base_velocity")
    robot = base_env.scene["robot"]

    obs, _ = env.reset()

    # 强制相机固定到世界坐标
    if hasattr(base_env, 'viewport_camera_controller') and base_env.viewport_camera_controller is not None:
        base_env.viewport_camera_controller.update_view_to_world()

    # 测试序列定义: [(vx, vy, wz, steps, name), ...]
    test_sequence = [
        (0.0, 0.0, 0.0, 50, "零指令平衡"),
        (1.0, 0.0, 0.0, 200, "前进 vx=1.0"),
        (0.0, 0.0, 0.0, 50, "停止"),
        (0.0, 0.0, 1.0, 100, "左转 wz=1.0"),
    ]

    _step = 0
    for vx, vy, wz, num_steps, name in test_sequence:
        print(f"\n{'='*70}")
        print(f"测试阶段: {name} (持续 {num_steps} 步)")
        print(f"指令: vx={vx}, vy={vy}, wz={wz}")
        print(f"{'='*70}\n")

        cmd = torch.tensor([vx, vy, wz], dtype=torch.float32, device=base_env.device)
        cmd_term.vel_command_b[:] = cmd

        pos_start = robot.data.root_pos_w[0].clone()

        for i in range(num_steps):
            with torch.inference_mode():
                if version.parse(skrl.__version__) >= version.parse("2.0.0"):
                    states = env.state() if hasattr(env, "state") else None
                    outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                else:
                    outputs = runner.agent.act(obs, timestep=0, timesteps=0)

                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])

                obs, _, _, _, _ = env.step(actions)
                _step += 1

                # [FIX] 在 disable_fabric 模式下强制更新视觉 transform
                if args_cli.disable_fabric and hasattr(robot, 'write_root_pose_to_sim'):
                    root_pose = torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=-1)
                    robot.write_root_pose_to_sim(root_pose)

                # 每50步打印一次状态
                if i % 50 == 0 or i == num_steps - 1:
                    pos = robot.data.root_pos_w[0]
                    vel_b = robot.data.root_lin_vel_b[0]
                    ang_b = robot.data.root_ang_vel_b[0]
                    delta = (pos - pos_start).cpu().numpy()
                    print(
                        f"  Step {i:3d}/{num_steps}: "
                        f"位置=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                        f"位移=({delta[0]:.2f}, {delta[1]:.2f}) | "
                        f"速度_x={vel_b[0]:.2f} ang_z={ang_b[2]:.2f}",
                        flush=True
                    )

        pos_end = robot.data.root_pos_w[0]
        movement = (pos_end - pos_start).norm().item()
        print(f"\n阶段总移动距离: {movement:.3f} m")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("请确认在 Isaac Sim 窗口中:")
    print("  1. 机器人本体是否在画面中移动了?")
    print("  2. 速度指令箭头(绿/蓝)是否在移动?")
    print("\n如果箭头动但机器人不动 → 视觉与物理脱节 (问题仍存在)")
    print("如果两者都动 → 视觉修复成功!")
    print("=" * 70 + "\n")

    print("窗口保持打开，按 Ctrl+C 退出...")
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                obs, _, _, _, _ = env.step(actions)
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
