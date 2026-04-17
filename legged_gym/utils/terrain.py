# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        # self.wjk_robot_terrain()
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def wjk_robot_terrain(self):
        terrain = self.wjk_make_terrain(type="slope1")
        self.add_terrain_to_map(terrain, 1, 0)
        self.add_terrain_to_map(terrain, 1, 1)
        self.add_terrain_to_map(terrain, 1, 2)
        self.add_terrain_to_map(terrain, 1, 3)
        terrain = self.wjk_make_terrain(type="platform")
        self.add_terrain_to_map(terrain, 0, 0)
        self.add_terrain_to_map(terrain, 0, 1)
        self.add_terrain_to_map(terrain, 0, 2)
        self.add_terrain_to_map(terrain, 0, 3)
        terrain = self.wjk_make_terrain(type="platform",param = 100)
        self.add_terrain_to_map(terrain, 2, 0)
        self.add_terrain_to_map(terrain, 2, 1)
        self.add_terrain_to_map(terrain, 2, 2)
        self.add_terrain_to_map(terrain, 2, 3)
        terrain = self.wjk_make_terrain(type="slope2",param=200)
        self.add_terrain_to_map(terrain, 3, 0)
        self.add_terrain_to_map(terrain, 3, 1)
        self.add_terrain_to_map(terrain, 3, 2)
        self.add_terrain_to_map(terrain, 3, 3)
        terrain = self.wjk_make_terrain(type="platform",param = 200)
        self.add_terrain_to_map(terrain, 4, 0)
        self.add_terrain_to_map(terrain, 4, 1)
        self.add_terrain_to_map(terrain, 4, 2)
        self.add_terrain_to_map(terrain, 4, 3)

    def wjk_make_terrain(self,type="slope1",param = 1500):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        if type == "slope1":
            wjk_slope_terrain_1(terrain)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        if type == "slope2":
            wjk_slope_terrain_2(terrain,param)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        if type == "platform":
            wjk_platform_terrain(terrain,platform_height = param)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.2)
        return terrain

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        terrain_cfg = self.cfg.level_property

        platform_size = terrain_cfg.platform_size
        slope = terrain_cfg.slope_angle * difficulty
        def linear_map(range, r):
            return range[0]+(range[1]-range[0])*r
        step_height_up = linear_map(terrain_cfg.stairs_height_up, difficulty)
        step_height_down = linear_map(terrain_cfg.stairs_height_down, difficulty)
        discrete_obstacles_height = linear_map(terrain_cfg.obstacles_height, difficulty)
        gap_size = terrain_cfg.gap_size * difficulty
        pit_depth = terrain_cfg.pit_depth * difficulty

        if choice < self.proportions[0]:
            #flat terrain
            pass
        elif choice < self.proportions[1]:
            # smooth slope down
            # 中间高四周低 下坡
            terrain_utils.pyramid_sloped_terrain(terrain, slope, platform_size)
        elif choice < self.proportions[2]:
            # smooth slope up
            terrain_utils.pyramid_sloped_terrain(terrain, -slope, platform_size)
        elif choice < self.proportions[3]:
            # rough slope down
            terrain_utils.pyramid_sloped_terrain(terrain, slope, platform_size)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[4]:
            # rough slope up
            terrain_utils.pyramid_sloped_terrain(terrain, -slope, platform_size)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[5]:
            # stair down
            # 中间高四周低 下楼梯
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height_down, platform_size=platform_size)
        elif choice < self.proportions[6]:
            # stair up
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=-step_height_up, platform_size=platform_size)
        elif choice < self.proportions[7]:
            # discrete obstacles
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=platform_size)
        elif choice < self.proportions[8]:
            # 跨沟
            gap_terrain(terrain, gap_size, platform_size)
        else:
            # 坑/上高台
            pit_terrain(terrain, pit_depth, platform_size)
        
        return terrain

    def add_terrain_to_map(self, terrain: terrain_utils.SubTerrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain: terrain_utils.SubTerrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain: terrain_utils.SubTerrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def wjk_slope_terrain_1(terrain: terrain_utils.SubTerrain, platform_height=1500.):
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = 1 - xx / terrain.width
    xx = xx.reshape(terrain.width, 1)
    max_height = int(platform_height)
    terrain.height_field_raw += np.clip((max_height * xx ),a_min=100,a_max = max_height).astype(terrain.height_field_raw.dtype)
    return terrain

def wjk_slope_terrain_2(terrain: terrain_utils.SubTerrain, platform_height=1500.):
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    xx = xx / terrain.width
    max_height = int(platform_height)
    terrain.height_field_raw += np.clip((max_height * xx ),a_min=100,a_max = max_height).astype(terrain.height_field_raw.dtype)
    return terrain

def wjk_platform_terrain(terrain: terrain_utils.SubTerrain, platform_height=1500.):
    max_height = int(platform_height)
    terrain.height_field_raw += max_height
    #terrain.height_field_raw += max_height.astype(terrain.height_field_raw.dtype)
    return terrain
