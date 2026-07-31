"""确认 robot 真实 body prim 路径,并验证:把物理位姿写进这些 USD prim 后,
USD ComputeLocalToWorldTransform 是否真的更新(即手动同步能否让网格动)。
无缓冲输出,结果写到 /tmp/wl_fix_result.txt。
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-Wheelleg-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from pxr import Usd, UsdGeom, Gf
import isaaclab_tasks
import WheelLeg.tasks
from isaaclab_tasks.utils import parse_env_cfg
import isaacsim.core.utils.stage as stage_utils

OUT = open("/tmp/wl_fix_result.txt", "w")
def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, file=OUT, flush=True)
    print(msg, flush=True)

def main():
    # disable_fabric 路径(手动写 USD 的场景)
    env_cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=False)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    stage = stage_utils.get_current_stage()
    obs, _ = env.reset()

    say("="*60)
    say("robot.cfg.prim_path =", robot.cfg.prim_path)
    say("robot.body_names =", robot.body_names)

    # 方法 A: root_physx_view.prim_paths (每个 body 的真实路径)
    try:
        pv_paths = robot.root_physx_view.prim_paths
        say("root_physx_view.prim_paths[:3] =", pv_paths[:3], "... total", len(pv_paths))
    except Exception as e:
        say("root_physx_view.prim_paths 失败:", repr(e))
        pv_paths = None

    # body prim 的真实路径 = root 路径去掉正则,替换成 env_0
    import re
    root_regex = robot.cfg.prim_path  # e.g. /World/envs/env_.*/Robot
    root_concrete = re.sub(r"env_\.\*|env_\[[^\]]*\]|\{ENV_REGEX_NS\}", "env_0",
                           root_regex).replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    # 若 root_regex 用的是 {ENV_REGEX_NS} 形式
    if "{ENV_REGEX_NS}" in root_regex:
        root_concrete = root_regex.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    say("推导出的具体 root =", root_concrete)

    # 检查每个 body prim 是否可拿到 + 可写 xformOp
    ok = 0
    ops = []
    for bn in robot.body_names:
        path = f"{root_concrete}/{bn}"
        p = stage.GetPrimAtPath(path)
        v = bool(p and p.IsValid())
        ta = p.GetAttribute("xformOp:translate") if v else None
        oa = p.GetAttribute("xformOp:orient") if v else None
        good = v and ta and ta.IsValid()
        if good:
            ok += 1
            ops.append((bn, p, ta, oa))
        say(f"  {path} valid={v} translate_ok={bool(ta and ta.IsValid())}")
    say(f">>> 可写连杆数 = {ok} / {len(robot.body_names)}")

    if ok == 0:
        say("!!! 仍然 0 个,路径推导还是错的")
        env.close(); return

    # ---- 决定性验证:强制机器人移动,手动写 USD,看 body 的世界变换是否更新 ----
    body_prim = ops[0][1]  # 第一个连杆
    xf = UsdGeom.Xformable(body_prim)
    def usd_world_t():
        return Gf.Vec3d(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())

    t0 = usd_world_t()
    zero = torch.zeros(env.action_space.shape, device=base_env.device)
    origin = base_env.scene.env_origins[0]
    for i in range(60):
        rv = robot.data.root_vel_w.clone(); rv[:, 0] = 2.0
        robot.write_root_velocity_to_sim(rv)
        env.step(zero)
        # 手动写 USD(修正后的路径)
        bpos = robot.data.body_pos_w[0] - origin
        bquat = robot.data.body_quat_w[0]
        for k, (bn, p, ta, oa) in enumerate(ops):
            bi = robot.body_names.index(bn)
            pp = bpos[bi].tolist(); qq = bquat[bi].tolist()
            if ta and ta.IsValid(): ta.Set(Gf.Vec3d(pp[0], pp[1], pp[2]))
            if oa and oa.IsValid(): oa.Set(Gf.Quatd(qq[0], qq[1], qq[2], qq[3]))
    t1 = usd_world_t()
    phys = float((robot.data.root_pos_w[0] - origin).norm())
    usd_moved = (t1 - t0).GetLength()
    say(f"USD body 世界变换: start={tuple(round(v,3) for v in t0)} end={tuple(round(v,3) for v in t1)}")
    say(f">>> USD moved={usd_moved:.4f}m | phys root dist~={phys:.4f}m")
    if usd_moved > 0.05:
        say("VERDICT: ✅ 修正路径后手动写 USD 生效 —— 网格会跟着动!")
    else:
        say("VERDICT: ❌ USD 仍不动,写位姿没被渲染采纳")
    say("="*60)
    env.close()

if __name__ == "__main__":
    main()
    OUT.close()
    simulation_app.close()
