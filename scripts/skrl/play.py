# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

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

# skrl 2.x changed the agent API (running mode + act signature)
_skrl_v2 = version.parse(skrl.__version__) >= version.parse("2.0.0")

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import WheelLeg.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
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

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    # skrl >= 2.0 renamed `set_running_mode("eval")` to `enable_training_mode(False)`
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # [视觉同步] --disable_fabric 下 Hydra 从 USD stage 取渲染变换,但 PhysX 不会把刚体
    # 位姿写回 USD stage,导致机器人本体网格静止不动。这里每步把物理位姿写进各连杆 USD
    # prim 的 xformOp,让视觉网格跟上物理移动。(fabric 模式无需此操作)
    # ⚠️ GPU 保护:GPU 仿真下 PhysX 开 eENABLE_DIRECT_GPU_API,写关节连杆 xformOp 会触发
    # 非法的 setGlobalPose,每帧刷屏报错且无效。手动写 USD 只在 CPU 下启用;GPU 走 fabric。
    _usd_sync_ops = None
    _sync_robot = None
    if args_cli.disable_fabric and str(env.unwrapped.device).startswith("cpu"):
        try:
            import re
            from pxr import Gf  # noqa: F401
            import isaacsim.core.utils.stage as _stage_utils
            _base_env = env.unwrapped
            _sync_robot = _base_env.scene["robot"]
            _stage = _stage_utils.get_current_stage()
            # 运行时 prim_path 是带正则的 "/World/envs/env_.*/Robot",必须替换成具体 env_0,
            # 否则拼出非法路径,GetPrimAtPath 全失败 -> 0 个连杆(网格不跟随物理移动)。
            _robot_root = re.sub(r"env_\.\*|env_\[[^\]]*\]", "env_0", _sync_robot.cfg.prim_path)
            _robot_root = _robot_root.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
            _ops = []
            for _bn in _sync_robot.body_names:
                _bp = _stage.GetPrimAtPath(f"{_robot_root}/{_bn}")
                _ops.append((_bp.GetAttribute("xformOp:translate"), _bp.GetAttribute("xformOp:orient"))
                            if (_bp and _bp.IsValid()) else None)
            _usd_sync_ops = _ops
            print(f"[PLAY] 视觉同步已启用: 每帧写入 {sum(o is not None for o in _ops)} 个连杆位姿到 USD。", flush=True)
        except Exception as _e:
            print(f"[PLAY] 视觉同步初始化失败(继续运行): {_e!r}", flush=True)
            _usd_sync_ops = None

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # skrl >= 2.0 requires `states` as a positional arg to `act()`
            if _skrl_v2:
                states = env.state() if hasattr(env, "state") else None
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            else:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

            # [视觉同步] 把物理位姿写回 USD,让本体网格跟随移动(仅 disable_fabric)
            if _usd_sync_ops is not None:
                from pxr import Gf
                _origin = env.unwrapped.scene.env_origins[0]
                _bpos = _sync_robot.data.body_pos_w[0] - _origin
                _bquat = _sync_robot.data.body_quat_w[0]  # (num_bodies, 4) wxyz
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
                # 写完位姿后强制渲染,让视口立即刷新到本帧物理位姿(否则只在空闲时才"跳"到位)
                env.unwrapped.sim.render()
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
