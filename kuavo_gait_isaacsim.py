# -*- coding: utf-8 -*-
# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
'''
@File    : getting_started_kuavo.py
@Time    : 2025/04/15 19:38:34
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : None
'''

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import torch
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view

import pandas as pd
import time

asset_path = "biped_s42_fixed.usd"
legged_mapping = {
    'knee': 'leg_*4_joint',
    'hip': 'leg_*3_joint',
    'ankle': 'leg_*5_joint'
}

def trun_sin(x, min: float, max: float):
    return (np.sin(x) + 1) / 2 * (max - min) + min

legged_dof_pos_func = {
    'hip': lambda phi: trun_sin(2 * np.pi * phi, -0.7854, -0),
    'knee': lambda phi: trun_sin(2 * np.pi * (phi+3/16), 0.5236, 1.7453),
    'ankle': lambda phi: trun_sin(2 * np.pi * (phi-3/8), -0.4363, 0.0873)
}

class Runner:
    def __init__(self):
        self.hz = 1000
        self.world = World(physics_dt=1/self.hz, stage_units_in_meters=1.0, backend='torch', device='cpu')
        self.world.scene.add_default_ground_plane()  # add ground plane
        set_camera_view(
            eye=[0.0, 4.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
        )  # set camera view

        prim_path = "/World/bot"
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)  # add robot to stage
        self.bot = Articulation(prim_paths_expr=prim_path, name='kauvo')
        self.bot.set_world_poses(positions=torch.tensor([[0.0, 0.0, 0.1]]) / get_stage_units())
        self.world.reset()

        self.last_dof_pos = [0] * len(self.bot.dof_names)
        self.cycle_time = 1.2

    def run(self):
        while True:
            phi = self.world.current_time / self.cycle_time % 1
            for key in ['knee', 'hip', 'ankle']:
                for idx, name in zip((0, 1), ('l', 'r')):
                    rad = legged_dof_pos_func[key]((phi + 0.375 * idx) % 1)
                    dof_idx = self.bot.get_dof_index(legged_mapping[key].replace('*', name))
                    self.last_dof_pos[dof_idx] = rad
            # self.bot.set_joint_position_targets([self.last_dof_pos])
            # self.bot.set_body_coms(positions=torch.tensor([[[0.0, 0.0, 1.89]]]) / get_stage_units(), 
            #                        orientations = torch.tensor([[[0.7835, 0.0000, 0.0000, -0.6214]]])/ get_stage_units())
            if self.world.current_time / self.cycle_time >= 1:
                self.bot.set_joint_positions(torch.tensor([self.last_dof_pos], dtype=torch.float))
            self.world.step(render=True)


if __name__ == '__main__':
    runner = Runner()
    runner.run()
    simulation_app.close()
