"""Full-env probe: does the REAL ManagerBasedRLEnv, with fabric ON, update the
render transform (fabric _worldPosition) when the robot moves? Uses the trained
checkpoint's env but forces motion via root-velocity writes so we don't need the policy.

  isaaclab.sh -p scripts/rt_fullenv_probe.py --task Template-Wheelleg-v0 --headless            # fabric ON
  isaaclab.sh -p scripts/rt_fullenv_probe.py --task Template-Wheelleg-v0 --headless --nofabric  # fabric OFF
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-Wheelleg-v0")
parser.add_argument("--nofabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from pxr import Usd, UsdGeom, Gf

import isaaclab_tasks  # noqa
import WheelLeg.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg


def main():
    use_fabric = not args_cli.nofabric
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=use_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    robot = base.scene["robot"]
    import isaacsim.core.utils.stage as stage_utils
    stage = stage_utils.get_current_stage()

    obs, _ = env.reset()
    print(f"\n[FULLENV] use_fabric={use_fabric} bodies={robot.num_bodies}")

    body_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/body")
    xf = UsdGeom.Xformable(body_prim)
    def usd_t():
        return Gf.Vec3d(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())

    fab = None
    try:
        from usdrt import Usd as RtUsd
        import omni.usd
        sid = omni.usd.get_context().get_stage_id()
        rt = RtUsd.Stage.Attach(sid)
        def _fab():
            p = rt.GetPrimAtPath("/World/envs/env_0/Robot/body")
            if not p: return None
            a = p.GetAttribute("_worldPosition")
            if a and a.HasValue():
                v = a.Get(); return Gf.Vec3d(v[0], v[1], v[2])
            return None
        fab = _fab
    except Exception as e:
        print("[FULLENV] usdrt NA:", repr(e))

    zero = torch.zeros(env.action_space.shape, device=base.device)
    t0 = usd_t(); f0 = fab() if fab else None
    p0 = robot.data.root_pos_w[0].clone()
    print(f"[FULLENV] start USD={tuple(round(v,3) for v in t0)} FAB={tuple(round(v,3) for v in f0) if f0 else None} phys={[round(float(v),3) for v in p0]}")

    for i in range(60):
        # force the root to move so we can see if render tracks it (bypass policy)
        rv = robot.data.root_vel_w.clone()
        rv[:, 0] = 2.0
        robot.write_root_velocity_to_sim(rv)
        env.step(zero)

    t1 = usd_t(); f1 = fab() if fab else None
    p1 = robot.data.root_pos_w[0].clone()
    print(f"[FULLENV] end   USD={tuple(round(v,3) for v in t1)} FAB={tuple(round(v,3) for v in f1) if f1 else None} phys={[round(float(v),3) for v in p1]}")
    usd_d = (t1 - t0).GetLength()
    fab_d = (f1 - f0).GetLength() if (f0 and f1) else -1.0
    phys_d = float((p1 - p0).norm())
    render_d = fab_d if (use_fabric and fab_d >= 0) else usd_d
    print(f"[FULLENV] >>> USD={usd_d:.4f} FAB={fab_d:.4f} phys={phys_d:.4f}")
    if phys_d < 0.02:
        v = "phys barely moved (inconclusive)"
    elif render_d < 0.02:
        v = "RENDER FROZEN"
    else:
        v = "RENDER TRACKS PHYSICS (fabric works in full env!)"
    print(f"[FULLENV] VERDICT[fabric={use_fabric}]: {v}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
