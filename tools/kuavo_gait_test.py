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

asset_path = "./models/biped_s42_fine/xml/biped_s42_fixed.usd"
legged_mapping = {
    'knee': 'leg_*4_joint',
    'hip': 'leg_*3_joint',
    'ankle': 'leg_*5_joint'
}

def trun_sin(x, min: float, max: float):
    return (np.sin(x) + 1) / 2 * (max - min) + min

def dif(x, min, max):
    if x < min:
        y = min - x
    elif x > max:
        y = max - x
    else:
        y = 0
    return y

def jump_hip(phi):
    return 0.7127 * np.sin(2 * np.pi * phi - 0.7077) - 0.9902

def jump_knee(phi):
    return -0.5537 * np.sin(34 / 9 * np.pi *
                        (phi - 3 / 17 + 0.35 * dif(phi, 0.25, 0.6)) + 0.0142) + 1.5105
    
def linear_fit(phi):
    y = jump_knee(12/17) - (jump_knee(12/17) - jump_knee(3/17)) / (8/17) * (phi - 12/17 + float(phi < 3 / 17))
    y = y + (1.6 - y) * min(phi - 12/17 + float(phi < 3 / 17), 3/17 - phi + float(phi > 12 / 17)) / (8/17)
    return y
    
def jump_knee_final(phi):
    if phi > 3 / 17 and phi < 12/17:
        y = jump_knee(phi)
    else:
        y = linear_fit(phi)
    return y
    
def jump_ankle(phi):
    if phi > 5 / 16 and phi < 10 / 16:
        return 0.3568 * np.sin(32 / 5 * np.pi * (phi - 5/16) - 1.3156) - 0.2421
    elif phi >= 10 / 16 and phi <= 1:
        return 0.1443 * np.sin(32 / 11 * np.pi * (phi - 10/16) - 0.0940) - 0.6516
    else:
        return 0.1443 * np.sin(32 / 11 * np.pi * (phi + 11/16) - 0.0940) - 0.6516
    
legged_dof_pos_func = {
    'hip': lambda phi: trun_sin(2 * np.pi * phi, -0.7854, -0),
    'knee': lambda phi: trun_sin(2 * np.pi * (phi+3/16), 0.5236, 1.7453),
    'ankle': lambda phi: trun_sin(2 * np.pi * (phi-3/8), -0.4363, 0.0873)
}

jumped_dof_pos_func = {
    'hip': lambda phi: jump_hip(phi),
    'knee': lambda phi: jump_knee_final(phi),
    'ankle': lambda phi: jump_ankle(phi)
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
        self.cycle_time = 2

    def run(self, gait):
        while True:
            phi = self.world.current_time / self.cycle_time % 1
            if gait == 'walk':
                for key in ['knee', 'hip', 'ankle']:
                    for idx, name in zip((0, 1), ('l', 'r')):
                        rad = legged_dof_pos_func[key]((phi + 0.375 * idx) % 1)
                        dof_idx = self.bot.get_dof_index(legged_mapping[key].replace('*', name))
                        self.last_dof_pos[dof_idx] = rad
            # self.bot.set_joint_position_targets([self.last_dof_pos])
            # self.bot.set_body_coms(positions=torch.tensor([[[0.0, 0.0, 1.89]]]) / get_stage_units(), 
            #                        orientations = torch.tensor([[[0.7835, 0.0000, 0.0000, -0.6214]]])/ get_stage_units())
            elif gait == 'jump':
                for key in ['knee', 'hip', 'ankle']:
                    for idx, name in zip((0, 1), ('l', 'r')):
                        rad = jumped_dof_pos_func[key]((phi) % 1)
                        dof_idx = self.bot.get_dof_index(legged_mapping[key].replace('*', name))
                        self.last_dof_pos[dof_idx] = rad
            if self.world.current_time / self.cycle_time >= 1:
                self.bot.set_joint_positions(torch.tensor([self.last_dof_pos], dtype=torch.float))
            self.world.step(render=True)


if __name__ == '__main__':
    runner = Runner()
    runner.run('jump')
    simulation_app.close()
