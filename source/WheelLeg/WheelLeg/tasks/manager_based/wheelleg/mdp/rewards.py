# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def upright_positive_z_l2(env, asset_cfg: SceneEntityCfg, alpha: float = 1.0):
    """
    保持机器人基座竖直且 z 轴朝上。
    - alpha 控制倒立惩罚的强度。
    - 基于 asset.data.projected_gravity_b，不依赖 IMU。
    """
    asset = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b   # (num_envs, 3), 单位向量

    # 横向倾斜惩罚：roll/pitch 偏差
    tilt_pen = torch.sum(torch.square(g_b[:, :2]), dim=1)

    # 倒立惩罚：g_bz > 0 表示重力指向机体正 z（即机体倒了）
    inverted_pen = torch.relu(g_b[:, 2]) ** 2

    return tilt_pen + alpha * inverted_pen


def flat_orientation_l2_signed(env, asset_cfg: SceneEntityCfg, invert_scale: float = 1.0,   # 反面朝上时的额外惩罚强度
):
    """
    区分正反面的平放（水平朝向）惩罚：
    - 倾斜惩罚: g_x^2 + g_y^2 （越水平越小）
    - 倒置惩罚: relu(g_z)^2  （正面朝上 g_z≈-1 → 0；反面朝上 g_z>0 → 惩罚）
    返回：非负惩罚值（越大越糟）。用法上建议在 RewardTerm 里配负权重。
    """
    asset = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b             # (N, 3), unit vector of gravity in body frame

    # 1) 横向倾斜惩罚（和官方 flat_orientation_l2 一致的主项）
    tilt_pen = torch.sum(torch.square(g_b[:, :2]), dim=1)  # g_x^2 + g_y^2

    # 2) 反面朝上惩罚（只有 g_z > 0，即倒置时才生效）
    inverted_pen = torch.relu(g_b[:, 2]) ** 2              # 0..1

    # 总惩罚
    penalty = tilt_pen + invert_scale * inverted_pen
    return penalty

def track_lin_vel_x_exp_penalty(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return 1.0 - torch.exp(-lin_vel_error / std**2)

def track_lin_vel_y_exp_penalty(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return 1.0 - torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_exp_penalty(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return 1.0 - torch.exp(-ang_vel_error / std**2)




def track_yaw_penalty(
    env,
    imu_cfg: SceneEntityCfg,
    command_name: torch.Tensor,
):
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi
    # 1) 取 yaw 指令
    yaw_cmd = command_name[:, 2] if (command_name.ndim == 2 and command_name.shape[1] >= 3) else command_name.reshape(-1)

    # 2) 获取 IMU 的四元数
    imu = env.scene[imu_cfg.name]
    q = imu.data.quat_w   # (N,4)，需确认顺序是 (w, x, y, z)

    # 3) 四元数 → 世界系 yaw (Z轴旋转角)
    w, x, y, z = q.unbind(-1)  # (N,)
    yaw_now = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # 4) yaw 误差
    yaw_err = _wrap_to_pi(yaw_cmd - yaw_now)

    # 5) 映射到 [0,1]：对齐=0，反向=1
    cos_err = torch.cos(yaw_err).clamp(-1.0, 1.0)
    reward = 0.5 * (1.0 - cos_err)

    return reward
