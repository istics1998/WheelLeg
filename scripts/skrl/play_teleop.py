# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
键盘遥控回放 skrl checkpoint。

在可视化窗口里用键盘实时给机器人下达速度指令(v_x, v_y, ω_z),
用来验证平衡策略能否前进/后退/转向/抗推挤。

键位(焦点需在 Isaac Sim 视口窗口上):
    ↑ / Numpad 8   前进 (+x)        ↓ / Numpad 2   后退 (-x)
    ← / Numpad 4   左移 (+y)        → / Numpad 6   右移 (-y)
    Z / Numpad 7   左转 (+yaw)      X / Numpad 9   右转 (-yaw)
    松开按键即回到零速指令(原地平衡)。
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="键盘遥控回放 skrl checkpoint。")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="The ML framework used for training the skrl agent.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="The RL algorithm used for training the skrl agent.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="遥控灵敏度倍率。")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
import weakref

import carb
import omni.appwindow

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import WheelLeg.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()


class LatchedSe2Keyboard:
    """锁存式 SE(2) 键盘遥控。

    与 IsaacLab 的 Se2Keyboard(按住才有速度)不同,这里是"按一下加一档,保持不变":
    每按一次方向键,对应速度分量增/减一个步长并持续保持,直到再次按键或归零。
    这样单击就能让机器人持续运动,便于观察平衡策略在稳态速度下的表现。

    键位:
        ↑ / ↓        v_x  +/-        ← / →        v_y  +/-
        Z / X        omega_z +/-     空格 / 0     全部归零(停)
    """

    def __init__(self, v_x_step=0.25, v_y_step=0.25, omega_z_step=0.5,
                 v_x_max=2.0, v_y_max=1.0, omega_z_max=6.28):
        self._cmd = np.zeros(3)
        self._steps = np.array([v_x_step, v_y_step, omega_z_step])
        self._max = np.array([v_x_max, v_y_max, omega_z_max])
        # +/- 增量映射到 [vx, vy, wz] 的哪个分量、什么方向 (WASD + QE,避免与视口箭头键冲突)
        self._key_delta = {
            "W": (0, +1),   # 前进 +vx
            "S": (0, -1),   # 后退 -vx
            "A": (1, +1),   # 左移 +vy (注意:本策略训练时 lin_vel_y 恒为 0,不会响应横移)
            "D": (1, -1),   # 右移 -vy (同上)
            "Q": (2, +1),   # 左转 +wz
            "E": (2, -1),   # 右转 -wz
        }
        self._stop_keys = {"SPACE", "NUMPAD_0"}
        self.reset_requested = False  # 按 R 请求复位环境
        # 订阅键盘事件
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

    def __del__(self):
        try:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def reset(self):
        self._cmd.fill(0.0)

    def advance(self) -> np.ndarray:
        return self._cmd

    def _on_event(self, event, *args, **kwargs):
        # 仅在按下(KEY_PRESS)时响应一次,松开不改变(锁存)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            name = event.input.name
            if name == "R":
                self.reset_requested = True
            elif name in self._stop_keys:
                self._cmd.fill(0.0)
            elif name in self._key_delta:
                idx, sign = self._key_delta[name]
                self._cmd[idx] = float(np.clip(self._cmd[idx] + sign * self._steps[idx],
                                                -self._max[idx], self._max[idx]))
        return True

    def __str__(self):
        return (
            "锁存式键盘遥控 (按一下加一档,保持不变):\n"
            "  W / S        前进/后退 (v_x  ±0.25, 上限 2.0)   ← 训练过,主要测这个\n"
            "  Q / E        左转/右转 (omega_z ±0.5)          ← 训练过\n"
            "  A / D        左移/右移 (v_y)  ※本策略未训练横移,不会响应\n"
            "  空格          全部归零(停,原地平衡)\n"
            "  R            复位环境(机器人失稳/发散后用它重来)"
        )


def main():
    """键盘遥控回放。"""
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    task_name = args_cli.task.split(":")[-1]

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(task_name, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")

    # 关闭指令自动重采样 / heading / 站立环境,让键盘指令不被内部逻辑覆盖
    try:
        vel_cmd = env_cfg.commands.base_velocity
        vel_cmd.resampling_time_range = (1.0e9, 1.0e9)  # 实际上永不重采样
        vel_cmd.heading_command = False
        vel_cmd.rel_standing_envs = 0.0
        vel_cmd.rel_heading_envs = 0.0
    except AttributeError:
        skrl.logger.warning("未找到 base_velocity 指令项,键盘指令可能被内部逻辑覆盖。")

    # checkpoint 路径
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode (skrl 2.x: enable_training_mode(False); 1.x: set_running_mode)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    # 键盘遥控设备(锁存式:按一下加一档并保持,便于观察稳态速度下的平衡表现)
    s = args_cli.sensitivity
    keyboard = LatchedSe2Keyboard(
        v_x_step=0.25 * s,
        v_y_step=0.25 * s,
        omega_z_step=0.5 * s,
    )
    keyboard.reset()
    print("\n" + "=" * 60)
    print(keyboard)
    print("=" * 60 + "\n")

    # 拿到底层 env 与指令项,用于每步覆写速度指令
    base_env = env.unwrapped
    cmd_term = base_env.command_manager.get_term("base_velocity")
    robot = base_env.scene["robot"]

    # [视觉同步] 在 --disable_fabric 模式下,Hydra 从 USD stage 取渲染变换,但 PhysX
    # 不会把刚体位姿写回 USD stage(只更新物理缓冲),导致机器人本体网格静止不动
    # (而速度箭头是每帧显式绘制的,所以会动)。这里在每步之后把物理算得的各连杆世界
    # 位姿显式写进对应 USD prim 的 xformOp,让视觉网格跟上物理。已用无头诊断验证有效。
    # ⚠️ GPU 保护:在 GPU 仿真下 PhysX 开启 eENABLE_DIRECT_GPU_API,对关节连杆 prim
    # 调 setGlobalPose(写 xformOp 会触发)是非法操作,会每帧刷屏报错且写入被拒、网格照样不动。
    # 因此这套"手动写 USD"只在 CPU 仿真下才启用;GPU 下应走 fabric 原生同步(不加 --disable_fabric)。
    _is_cpu = str(base_env.device).startswith("cpu")
    _usd_sync_ops = None
    if args_cli.disable_fabric and _is_cpu:
        try:
            import re
            from pxr import Gf, UsdGeom  # noqa: F401
            import isaacsim.core.utils.stage as _stage_utils
            _stage = _stage_utils.get_current_stage()
            # 运行时 robot.cfg.prim_path 是带正则的 "/World/envs/env_.*/Robot"
            # (不是 "{ENV_REGEX_NS}"),必须把 env 正则段替换成具体的 env_0,
            # 否则拼出的路径非法(Ill-formed SdfPath),GetPrimAtPath 全失败 -> 0 个连杆。
            _robot_root = re.sub(r"env_\.\*|env_\[[^\]]*\]", "env_0", robot.cfg.prim_path)
            _robot_root = _robot_root.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
            _ops = []
            for _bn in robot.body_names:
                _bp = _stage.GetPrimAtPath(f"{_robot_root}/{_bn}")
                if _bp and _bp.IsValid():
                    _ops.append((_bp.GetAttribute("xformOp:translate"),
                                 _bp.GetAttribute("xformOp:orient")))
                else:
                    _ops.append(None)
            _usd_sync_ops = _ops
            print(f"[TELEOP] 视觉同步已启用: 将每帧把 {sum(o is not None for o in _ops)} 个连杆位姿写入 USD。", flush=True)
        except Exception as _e:
            print(f"[TELEOP] 视觉同步初始化失败(继续运行): {_e!r}", flush=True)
            _usd_sync_ops = None

    # 启动前自检:确认 checkpoint 没有加载出 nan 权重(训练发散的 checkpoint 会全是 nan,
    # 导致策略输出 nan 动作、物理瞬间发散)。
    with torch.inference_mode():
        _st = env.state() if hasattr(env, "state") else None
        obs, _ = env.reset()
        _out = (runner.agent.act(obs, _st, timestep=0, timesteps=0)
                if version.parse(skrl.__version__) >= version.parse("2.0.0")
                else runner.agent.act(obs, timestep=0, timesteps=0))
        _act = _out[-1].get("mean_actions", _out[0])
        if not torch.isfinite(_act).all():
            print("\n" + "!" * 60)
            print("[FATAL] 策略输出 nan 动作 —— 这个 checkpoint 已损坏(训练发散)。")
            print("        换一个干净的 checkpoint: WL_CKPT=/path/to/xxx.pt")
            print("!" * 60 + "\n", flush=True)
            env.close()
            return

    # reset environment
    obs, _ = env.reset()

    # 强制相机固定在世界坐标,不跟随机器人(让机器人在画面里真正移动)
    if hasattr(base_env, 'viewport_camera_controller') and base_env.viewport_camera_controller is not None:
        base_env.viewport_camera_controller.update_view_to_world()

    # simulate environment
    _last_dbg = None
    _step = 0
    _nan_reported = False
    _nan_streak = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 手动复位:按 R 键(在 step 前处理)
            if getattr(keyboard, "reset_requested", False):
                keyboard.reset_requested = False
                keyboard.reset()
                obs, _ = env.reset()
                _step = 0
                _nan_reported = False
                _nan_streak = 0
                print("[TELEOP] 手动复位。", flush=True)
                continue

            # 读取键盘指令 [v_x, v_y, omega_z] 并写入所有环境的指令 buffer
            raw_cmd = keyboard.advance()
            _cur = tuple(round(float(v), 3) for v in raw_cmd)
            if _cur != _last_dbg:
                print(f"[TELEOP] keyboard cmd (vx, vy, wz) = {_cur}", flush=True)
                _last_dbg = _cur
            kb_cmd = torch.tensor(raw_cmd, dtype=torch.float32, device=base_env.device)
            cmd_term.vel_command_b[:] = kb_cmd

            # agent stepping (deterministic). skrl 2.x: act(obs, states, ...); 1.x: act(obs, ...)
            if version.parse(skrl.__version__) >= version.parse("2.0.0"):
                states = env.state() if hasattr(env, "state") else None
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            else:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
            _step += 1

            # [视觉同步] disable_fabric 下把物理位姿写回 USD,让本体网格跟随物理移动。
            # 各连杆 prim 是 Robot(位于 env 原点)的子节点,其 xformOp 为相对原点的局部
            # 位姿;单环境时 env_0 原点为 (0,0,0),故世界位姿减去 env 原点即为局部位姿。
            if _usd_sync_ops is not None:
                from pxr import Gf
                _origin = base_env.scene.env_origins[0]
                _bpos = (robot.data.body_pos_w[0] - _origin)
                _bquat = robot.data.body_quat_w[0]  # (num_bodies, 4) wxyz
                for _k, _op in enumerate(_usd_sync_ops):
                    if _op is None:
                        continue
                    _ta, _oa = _op
                    _p = _bpos[_k].tolist()
                    _q = _bquat[_k].tolist()
                    if _ta and _ta.IsValid():
                        _ta.Set(Gf.Vec3d(_p[0], _p[1], _p[2]))
                    if _oa and _oa.IsValid():
                        _oa.Set(Gf.Quatd(_q[0], _q[1], _q[2], _q[3]))
                # env.step 里的渲染发生在写位姿之前,视口拿到的是旧位姿;写完后再强制渲染
                # 一次,让视口立即刷新到本帧物理位姿(否则本体只在停下/空闲时才"跳"到位)。
                base_env.sim.render()

            # step 之后读取状态
            pos_w = robot.data.root_pos_w[0]
            lin_b = robot.data.root_lin_vel_b[0]
            is_nan = not (torch.isfinite(pos_w).all() and torch.isfinite(lin_b).all())

            # 连续 nan 计数:只有持续发散才判定为真炸(单步瞬时 nan 常是缓冲读取假象)
            if is_nan:
                _nan_streak += 1
            else:
                _nan_streak = 0

            # 连续 30 步都 nan 才认为真发散,提示复位(避免瞬时假 nan 误判)
            if _nan_streak >= 30 and not _nan_reported:
                print(f"[WARN] step {_step}: 连续 {_nan_streak} 步 nan,物理确已发散。按 R 复位重试。", flush=True)
                _nan_reported = True

            # 正常状态下每 50 步打印一次指令 vs 实际
            if not is_nan and _step % 50 == 0:
                injected = base_env.command_manager.get_command("base_velocity")[0].tolist()
                lin = lin_b.tolist()
                ang_b = robot.data.root_ang_vel_b[0].tolist()
                pos = pos_w.tolist()
                print(
                    f"[DBG] 指令={[round(v,2) for v in injected]} | "
                    f"实际线速度_b=[{lin[0]:.2f},{lin[1]:.2f},{lin[2]:.2f}] "
                    f"角速度z={ang_b[2]:.2f} | 位置xy=[{pos[0]:.2f},{pos[1]:.2f}] 高度z={pos[2]:.2f}",
                    flush=True,
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
