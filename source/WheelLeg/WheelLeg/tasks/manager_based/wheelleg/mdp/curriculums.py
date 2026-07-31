# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs import mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def override_command_range(env, env_ids, old_value, value, num_steps):
    if getattr(env, "common_step_counter", 0) > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE

