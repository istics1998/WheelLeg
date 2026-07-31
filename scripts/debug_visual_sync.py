#!/usr/bin/env python3
"""
诊断并修复机器人视觉网格与物理脱节问题。

测试方案：
1. 检查 Articulation 的 visual/physics 同步状态
2. 尝试强制刷新视觉变换
3. 对比 Fabric vs non-Fabric 模式下的差异
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="诊断视觉同步问题")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="WheelLeg-Flat-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks
import WheelLeg.tasks
from isaaclab_tasks.utils import parse_env_cfg

def main():
    print("\n" + "=" * 70)
    print("视觉同步诊断")
    print(f"Fabric 模式: {'禁用' if args_cli.disable_fabric else '启用'}")
    print("=" * 70 + "\n")

    # 创建环境
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    # reset
    env.reset()
    print(f"[INFO] 环境已创建，机器人类型: {type(robot).__name__}")
    print(f"[INFO] Prim 路径: {robot.cfg.prim_path}")

    # 检查初始状态
    pos_init = robot.data.root_pos_w[0].clone()
    print(f"[INFO] 初始位置: {pos_init.tolist()}")

    # 给一个简单的前进指令
    cmd_term = base_env.command_manager.get_term("base_velocity")
    cmd_term.vel_command_b[:] = torch.tensor([1.0, 0.0, 0.0], device=base_env.device)
    print(f"[INFO] 设置速度指令: vx=1.0 m/s\n")

    # step 100次，每20步打印位置
    print("开始 stepping...")
    actions = torch.zeros(env.action_space.shape, device=base_env.device)

    for step in range(100):
        obs, _, _, _, _ = env.step(actions)

        if step % 20 == 0:
            pos = robot.data.root_pos_w[0]
            lin_vel = robot.data.root_lin_vel_b[0]
            delta = (pos - pos_init).cpu().numpy()
            print(f"Step {step:3d}: pos={pos.tolist()} | "
                  f"delta_xy=({delta[0]:.3f}, {delta[1]:.3f}) | "
                  f"vel_x={lin_vel[0]:.3f}")

    print("\n" + "=" * 70)
    print("诊断结果:")
    pos_final = robot.data.root_pos_w[0]
    movement = (pos_final - pos_init).norm().item()
    print(f"总移动距离: {movement:.3f} m")

    if movement > 0.1:
        print("✓ 物理 simulation 正常 (机器人发生了位移)")
        print("\n如果视觉上机器人静止不动，说明是渲染同步问题。")
        print("尝试以下修复方案:")
        print("  1. 使用 Fabric 模式 (去掉 --disable_fabric)")
        print("  2. 检查 USD 文件的 visual mesh 是否正确绑定到 root")
        print("  3. 在 Isaac Sim GUI 中手动检查 Xform 的 translate 属性是否更新")
    else:
        print("✗ 物理 simulation 异常 (机器人没有移动)")
        print("可能原因: 策略未加载/动作全零/物理参数问题")

    print("=" * 70 + "\n")

    # 保持窗口打开，让用户观察
    print("按 Ctrl+C 退出...")
    try:
        while simulation_app.is_running():
            env.step(actions)
    except KeyboardInterrupt:
        pass

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
