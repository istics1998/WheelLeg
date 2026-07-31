#!/usr/bin/env python3
"""无头验证 wheelleg_mini_loop.usd 的闭环稳定性(单 env, headless)。

诊断版:用【真实训练执行器设置】(stiffness/damping、hip init=0.52、mimic 一致的
从动关节初值),而不是零阻尼自由落——后者会让未耦合的轮关节自由空转到 ~90rad/s,
造成"发散"假象。关键输出:每步 |vel| 最大的【具体关节名】,分清是机构发散还是轮子空转。

reset 后保持驱动目标不变(hold),自由演化 steps 步,检查:
  - nan/inf
  - 各关节速度(逐关节报 argmax)
  - base 高度是否维持(站得住 vs 塌了)

用法(需 conda activate isaaclab45):
  isaaclab.sh -p scripts/verify_loop_usd.py --usd <loop.usd> --steps 300
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str,
    default="/home/ist/桌面/Isaac_project/WheelLeg/source/WheelLeg/WheelLeg/robots/wheelleg_mini_loop.usd")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--free", action="store_true", help="零阻尼自由落(旧行为,仅调试)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# 真实训练初值(见 wheelleg_bot.py)
INIT_DRIVERS = {"body_to_sprocket2": 0.52, "body_to_sprocket4": -0.52}
# mimic 耦合斜率(follower -> (ref, gearing));ref=hip 时初值随 hip,否则 0
MIMIC = {
    "body_to_sprocket1": ("body_to_sprocket2", 1.0),
    "sprocket1_to_link1_l": ("body_to_sprocket2", 0.106),
    "link1_l_to_link2_l": ("body_to_sprocket2", -0.272),
    "sprocket2_to_link3_l": ("body_to_sprocket2", -0.434),
    "body_to_sprocket_link1": ("body_to_sprocket2", 1.0),
    "body_to_sprocket_link2": ("body_to_sprocket2", 1.0),
    "body_to_sprocket3": ("body_to_sprocket4", 1.0),
    "sprocket3_to_link1_r": ("body_to_sprocket4", 0.106),
    "link1_r_to_link2_r": ("body_to_sprocket4", -0.272),
    "sprocket4_to_link3_r": ("body_to_sprocket4", -0.434),
    "body_to_sprocket_link3": ("body_to_sprocket4", 1.0),
    "body_to_sprocket_link4": ("body_to_sprocket4", 1.0),
}


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120.0, device=args.device))
    sim.set_camera_view([2.0, 2.0, 1.0], [0.0, 0.0, 0.2])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=800.0).func("/World/light", sim_utils.DomeLightCfg(intensity=800.0))

    if args.free:
        acts = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)}
        init_z = 0.35
    else:
        # 贴近训练:hip/knee stiffness=40 damping=1,轮 stiffness=0 damping=5
        acts = {
            "hipknee": ImplicitActuatorCfg(
                joint_names_expr=["body_to_sprocket2", "body_to_sprocket4",
                                  "sprocket2_to_g6_l", "sprocket4_to_g6_r"],
                stiffness=40.0, damping=1.0),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["g6_l_to_wheel1", "g6_r_to_wheel2"],
                stiffness=0.0, damping=5.0),
            "passive": ImplicitActuatorCfg(  # 从动关节:给点阻尼免得数值抖
                joint_names_expr=[".*"], stiffness=0.0, damping=0.5),
        }
        init_z = 0.30

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
                max_linear_velocity=10.0, max_angular_velocity=100.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, init_z)),
        actuators=acts,
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    print(f"[VERIFY] usd={args.usd}", flush=True)
    print(f"[VERIFY] 关节数={robot.num_joints} 连杆数={robot.num_bodies} free={args.free}", flush=True)
    names = robot.joint_names

    # 构造 mimic 一致的初始关节角
    q0 = robot.data.default_joint_pos.clone()
    if not args.free:
        for i, n in enumerate(names):
            if n in INIT_DRIVERS:
                q0[:, i] = INIT_DRIVERS[n]
        for i, n in enumerate(names):
            if n in MIMIC:
                ref, g = MIMIC[n]
                base = INIT_DRIVERS.get(ref, 0.0)
                q0[:, i] = g * base
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    robot.set_joint_position_target(q0)
    robot.reset()

    max_vel = 0.0
    nan_hit = False
    for i in range(args.steps):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        qpos, qvel = robot.data.joint_pos, robot.data.joint_vel
        if torch.isnan(qpos).any() or torch.isnan(qvel).any() or torch.isinf(qvel).any():
            print(f"[VERIFY] ❌ 第 {i} 步 nan/inf!", flush=True)
            nan_hit = True
            break
        vabs = qvel.abs()
        vmax, jidx = vabs.max().item(), vabs.argmax().item() % len(names)
        max_vel = max(max_vel, vmax)
        if i % 50 == 0:
            h = robot.data.root_pos_w[0, 2].item()
            print(f"  step {i:3d}: |vel|max={vmax:7.2f} rad/s @{names[jidx]:22s} base_z={h:.3f}m", flush=True)

    print("\n===== 验证结果 =====", flush=True)
    final_h = robot.data.root_pos_w[0, 2].item()
    # 报告收尾各关节速度分布
    vend = robot.data.joint_vel.abs()[0]
    order = torch.argsort(vend, descending=True)
    print("末步各关节 |vel| 前 6:", flush=True)
    for k in order[:6].tolist():
        print(f"    {names[k]:24s} {vend[k].item():.2f} rad/s", flush=True)
    if nan_hit:
        print("❌ nan/inf — 过约束或锚点错误。", flush=True)
    elif max_vel > 50.0:
        print(f"⚠️  max|vel|={max_vel:.1f} rad/s — 看上面是哪个关节:若是 wheel 属正常空转;"
              f"若是 mimic 连杆则确为机构发散。base_z={final_h:.3f}m", flush=True)
    else:
        print(f"✅ 稳定:{args.steps} 步无 nan,max|vel|={max_vel:.2f} rad/s,base_z={final_h:.3f}m", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
