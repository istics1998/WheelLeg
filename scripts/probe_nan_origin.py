"""前台 nan 溯源探针。

目的:训练从第 1 个 checkpoint 就全 nan(即时爆炸),加阻尼+加 clip 均无效 →
说明物理层直接产出 nan(clamp(nan)=nan,故 obs clip 救不了),会瞬间毒化
RunningStandardScaler。此脚本直接建真 env,分两种驱动各跑若干步,每步后逐张量
检查 nan/inf 与极值,报告**第一次**变坏的 step、张量、以及具体关节/体索引。

用法(前台):
  conda run -n isaaclab45 --no-capture-output /home/ist/IsaacLab/isaaclab.sh \
      -p scripts/probe_nan_origin.py --headless --num_envs 16
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import WheelLeg.tasks  # noqa: F401  注册 Template-Wheelleg-v0
from isaaclab_tasks.utils import parse_env_cfg


def scan(tag, t, step, names=None):
    """返回 True 表示发现异常并已打印。"""
    if not torch.is_tensor(t):
        return False
    nan = torch.isnan(t)
    inf = torch.isinf(t)
    if nan.any() or inf.any():
        idx = torch.nonzero(nan | inf)
        first = idx[0].tolist()
        col = first[-1]
        nm = names[col] if (names and col < len(names)) else f"col{col}"
        print(f"  !! [{tag}] step={step} nan={int(nan.sum())} inf={int(inf.sum())} "
              f"first_at={first} joint/comp='{nm}'")
        return True
    mx = t.abs().max().item()
    if mx > 1e4:
        am = torch.argmax(t.abs().reshape(-1)).item()
        col = am % t.shape[-1]
        nm = names[col] if (names and col < len(names)) else f"col{col}"
        print(f"  ~~ [{tag}] step={step} |max|={mx:.3e} at col='{nm}' (还未 nan,快爆了)")
    return False


def run(env, robot, jn, mode):
    print(f"\n===== 驱动模式: {mode} =====")
    obs, _ = env.reset()
    print("  reset 后即时检查:")
    scan("joint_vel", robot.data.joint_vel, 0, jn)
    scan("joint_pos", robot.data.joint_pos, 0, jn)
    scan("root_vel_w", robot.data.root_vel_w, 0)
    scan("obs", obs["policy"] if isinstance(obs, dict) else obs, 0)

    nact = env.action_space.shape[-1]
    bad = False
    for step in range(1, args.steps + 1):
        if mode == "zero":
            act = torch.zeros((args.num_envs, nact), device=env.unwrapped.device)
        else:  # random,复现策略探索的大动作
            act = torch.rand((args.num_envs, nact), device=env.unwrapped.device) * 2 - 1
        obs, rew, term, trunc, info = env.step(act)
        o = obs["policy"] if isinstance(obs, dict) else obs
        hit = False
        hit |= scan("joint_vel", robot.data.joint_vel, step, jn)
        hit |= scan("joint_pos", robot.data.joint_pos, step, jn)
        hit |= scan("root_vel_w", robot.data.root_vel_w, step)
        hit |= scan("reward", rew, step)
        hit |= scan("obs", o, step)
        if hit:
            print(f"  >>> 首次异常出现在 step={step},模式={mode}。停止本模式。")
            bad = True
            break
        if step % 25 == 0:
            jv = robot.data.joint_vel
            am = torch.argmax(jv.abs().reshape(-1)).item()
            col = am % jv.shape[-1]
            print(f"  step={step} ok  |joint_vel|max={jv.abs().max().item():.2f} "
                  f"@{jn[col]}  |obs|max={o.abs().max().item():.2f}")
    if not bad:
        print(f"  === 模式 {mode} 跑完 {args.steps} 步,无 nan/inf/爆炸 ===")


# 只建一次 env(冷启动只付一次),两种驱动复用,跑完 zero 后 reset 再跑 random。
env_cfg = parse_env_cfg("Template-Wheelleg-v0", num_envs=args.num_envs)
env = gym.make("Template-Wheelleg-v0", cfg=env_cfg)
robot = env.unwrapped.scene["robot"]
jn = robot.data.joint_names
print("  joint_names:", jn)

run(env, robot, jn, "zero")
run(env, robot, jn, "random")

env.close()
simulation_app.close()
