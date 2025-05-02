# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import LejuWBCRoughEnvCfg


@configclass
class LejuWBCFlatEnvCfg(LejuWBCRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # Commands
        # self.commands.base_velocity.ranges.gaits = (2.0, 3.0)
        # self.commands.base_velocity.ranges.gaits = (1.0, 2.0)
        # self.commands.base_velocity.ranges.gaits = (0.0, 1.0)
        # self.commands.base_velocity.ranges.walk_lin_vel_x = (0.5, 0.5)
        # self.commands.base_velocity.ranges.run_lin_vel_x = (2.0, 2.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.4, 0.4)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.4, 0.4)
        # self.commands.base_velocity.ranges.jump_lin_vel_x = (1.5, 1.5)
