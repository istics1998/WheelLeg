"""Minimal, low-memory diagnostic: spawn ONLY the robot articulation in a bare
sim (no RL managers, no sensors), give the root a velocity, step, and check
whether the body's USD render transform tracks the physics body.

Run e.g.:
  isaaclab.sh -p scripts/rt_diag_min.py --headless               # fabric ON
  isaaclab.sh -p scripts/rt_diag_min.py --headless --nofabric    # fabric OFF
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--nofabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pxr import Usd, UsdGeom, Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
import isaacsim.core.utils.stage as stage_utils

from WheelLeg.robots.wheelleg_bot import WHEELLEG_CONFIG


def main():
    use_fabric = not args_cli.nofabric
    print("\n" + "=" * 70)
    print(f"[MINDIAG] use_fabric = {use_fabric}")
    sim_cfg = SimulationCfg(dt=0.005, device=args_cli.device, use_fabric=use_fabric)
    sim = SimulationContext(sim_cfg)

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    # spawn robot exactly like the scene does (single instance)
    robot_cfg = WHEELLEG_CONFIG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    print("[MINDIAG] sim reset done. bodies:", robot.num_bodies, "joints:", robot.num_joints)

    stage = stage_utils.get_current_stage()
    body_prim = stage.GetPrimAtPath("/World/Robot/body")
    print("[MINDIAG] /World/Robot/body valid:", body_prim.IsValid())
    # count meshes actually under robot at runtime
    nmesh = sum(1 for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"), Usd.TraverseInstanceProxies())
                if p.GetTypeName() == "Mesh")
    print("[MINDIAG] runtime Mesh prims under /World/Robot:", nmesh)
    xf = UsdGeom.Xformable(body_prim)

    def world_t():
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return Gf.Vec3d(m.ExtractTranslation())

    # fabric/usdrt stage = what Hydra actually renders. In fabric mode the live
    # transform lives here, not on the USD stage.
    fabric_reader = None
    try:
        import usdrt
        from usdrt import Usd as RtUsd, Sdf as RtSdf, Gf as RtGf
        stage_id = stage_utils.get_current_stage_id() if hasattr(stage_utils, "get_current_stage_id") else None
        if stage_id is None:
            import omni.usd
            stage_id = omni.usd.get_context().get_stage_id()
        rt_stage = RtUsd.Stage.Attach(stage_id)

        def _fabric_t():
            rp = rt_stage.GetPrimAtPath("/World/Robot/body")
            if not rp:
                return None
            for attr_name in ("_worldPosition", "xformOp:translate", "_worldMatrix"):
                a = rp.GetAttribute(attr_name)
                if a and a.HasValue():
                    v = a.Get()
                    try:
                        return Gf.Vec3d(v[0], v[1], v[2])
                    except Exception:
                        try:
                            t = v.ExtractTranslation()
                            return Gf.Vec3d(t[0], t[1], t[2])
                        except Exception:
                            pass
            return None
        fabric_reader = _fabric_t
        _rt_dbg_prim = rt_stage.GetPrimAtPath("/World/Robot/body")
        print("[MINDIAG] usdrt fabric stage attached OK; body attrs:",
              [a.GetName() for a in _rt_dbg_prim.GetAttributes()] if _rt_dbg_prim else "NO PRIM")
    except Exception as e:
        print("[MINDIAG] usdrt unavailable:", repr(e))

    # give the root a strong forward+up velocity so it clearly moves
    root_vel = robot.data.default_root_state[:, 7:13].clone()
    root_vel[:, 0] = 3.0  # vx
    robot.write_root_velocity_to_sim(root_vel)

    t0 = world_t()
    f0 = fabric_reader() if fabric_reader else None
    p0 = robot.data.root_pos_w[0].clone()
    print(f"[MINDIAG] start: USD_T={tuple(round(v,3) for v in t0)} FABRIC_T={tuple(round(v,3) for v in f0) if f0 else None} phys={[round(float(v),3) for v in p0]}")

    # ---- FIX MECHANISM UNDER TEST: write physics body world-poses to USD each frame ----
    body_names = robot.body_names
    body_ops = []  # (translate_attr, orient_attr) per body
    for bn in body_names:
        bp = stage.GetPrimAtPath(f"/World/Robot/{bn}")
        if not bp or not bp.IsValid():
            body_ops.append(None)
            continue
        ta = bp.GetAttribute("xformOp:translate")
        oa = bp.GetAttribute("xformOp:orient")
        body_ops.append((ta, oa))

    def push_poses_to_usd():
        pos = robot.data.body_pos_w[0]   # (num_bodies, 3), world (env at origin)
        quat = robot.data.body_quat_w[0]  # (num_bodies, 4) wxyz
        for k, ops in enumerate(body_ops):
            if ops is None:
                continue
            ta, oa = ops
            p = pos[k].tolist()
            q = quat[k].tolist()  # w,x,y,z
            if ta and ta.IsValid():
                ta.Set(Gf.Vec3d(p[0], p[1], p[2]))
            if oa and oa.IsValid():
                oa.Set(Gf.Quatd(q[0], q[1], q[2], q[3]))
    # -------------------------------------------------------------------------------------

    for i in range(60):
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim_cfg.dt)
        # only apply the USD-write fix in non-fabric mode (matches the real fix gating);
        # in fabric mode we measure NATIVE fabric behavior with no manual writes.
        if not use_fabric:
            push_poses_to_usd()

    t1 = world_t()
    f1 = fabric_reader() if fabric_reader else None
    p1 = robot.data.root_pos_w[0].clone()
    print(f"[MINDIAG] end:   USD_T={tuple(round(v,3) for v in t1)} FABRIC_T={tuple(round(v,3) for v in f1) if f1 else None} phys={[round(float(v),3) for v in p1]}")

    usd_d = (t1 - t0).GetLength()
    fab_d = (f1 - f0).GetLength() if (f0 and f1) else -1.0
    phys_d = float((p1 - p0).norm())
    print(f"[MINDIAG] >>> USD moved={usd_d:.4f}m | FABRIC moved={fab_d:.4f}m | physics moved={phys_d:.4f}m")

    render_d = fab_d if (use_fabric and fab_d >= 0) else usd_d
    if nmesh <= 0:
        verdict = "NO MESH AT RUNTIME"
    elif phys_d < 0.02:
        verdict = "physics barely moved (inconclusive)"
    elif render_d < 0.02:
        verdict = "BUG: physics moves but RENDER transform FROZEN -> mesh appears static"
    else:
        verdict = "OK: render transform tracks physics -> mesh moves visually"
    print(f"[MINDIAG] VERDICT[fabric={use_fabric}]: {verdict}")
    print("=" * 70 + "\n", flush=True)
    sim.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()
