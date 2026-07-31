from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

def root_height_over_maximum(
    env: ManagerBasedRLEnv, maximum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is over the maximum height.

    Note:
        This is currently only supported for flat terrains, i.e. the maximum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height

def terminate_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_min: float = -1.0, x_max: float = 1.0,
    y_min: float = -1.0, y_max: float = 1.0,
) -> torch.Tensor:
    """Terminate if the asset's base leaves the given (x, y) bounds."""
    # 获取机器人根位置 (world frame)
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w[:, :2]  # 只取 x, y

    # 判断是否超出边界
    out_x = (root_pos[:, 0] < x_min) | (root_pos[:, 0] > x_max)
    out_y = (root_pos[:, 1] < y_min) | (root_pos[:, 1] > y_max)
    out_of_bounds = out_x | out_y

    return out_of_bounds