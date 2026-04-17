from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.legged_math import (quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float,
                                          fit_plane, get_quat_yaw, torch_rand_float_1d)
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import legged_gym.envs.base.rewards as rewards
import legged_gym.envs.base.observations as observations
from legged_gym.envs.base.observation_buffer_3d import ObservationBuffer3D

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self.reward_module = rewards
        self._prepare_reward_function()
        self.init_done = True
        self.create_control_panel() # 控制窗口

    def create_control_panel(self):
        pass

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])
        obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids

    def get_observations(self):
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        return policy_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync:
            self.draw_debug_vis()

        return env_ids

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        self.randomize_at_reset(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['reward/' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain/terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["command/max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i](self) * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = rewards.reward_termination(self) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        add_noise = self.cfg.noise.add_noise
        self.privileged_obs_buf = torch.cat((
                                    observations.obs_base_lin_vel(self,add_noise), # 3
                                    observations.obs_base_ang_vel(self,add_noise), # 3
                                    observations.obs_projected_gravity(self,add_noise), # 3
                                    observations.obs_commands(self), # 3
                                    observations.obs_dof_pos(self,add_noise), # 12
                                    observations.obs_dof_vel(self,add_noise), # 12
                                    observations.obs_last_actions(self) # 12
                                ),dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.add_height_observation:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, observations.obs_measure_heights(self,add_noise)), dim=-1)

        self.obs_buf = torch.clone(self.privileged_obs_buf)

    def get_global_noise_level(self):
        if self.cfg.noise.noise_curriculum:
            # 根据iter改变noise level
            current_iter=self.common_step_counter//24
            difficulty=float(current_iter//1000) /8 # 每1000it 提升1等级 满难度4000iter
            difficulty=min(difficulty,1.0) # 0~1
            init_exp = 1.3 #初始值10^-1.3=0.05
            progress=(difficulty-1)*init_exp # -1.3~0
            scale=math.pow(10,progress) #指数增加 10^-x
            return self.cfg.noise.noise_level * scale
        else:
            return self.cfg.noise.noise_level

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            total_mass = 0
            print("====== mass before randomization ========")
            for i, p in enumerate(props):
                total_mass += p.mass
                print(f"body {i}: {p.mass:.3f}")
            print(f"Total mass: {total_mass:.3f}")

            # 记录原始base质量
            self.original_base_mass = props[0].mass
            self.original_total_mass = total_mass
            self.real_mass = torch.zeros(self.num_envs,dtype=torch.float,device=self.device,requires_grad=False)
            self.real_mass[:] = self.original_total_mass

        if self.cfg.domain_rand.randomize_base_mass:
            props[0].mass = self.original_base_mass + self.added_mass[env_id]
            self.real_mass[env_id] += self.added_mass[env_id]
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
        地形法线计算
        接触状态、步态变量buffer更新
        """

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.terrain_normal_GCS, self.base_height = self.compute_terrain_normal_and_distance()
            self.body_normal_GCS=quat_rotate(self.base_quat,self.gravity_vec)

        if self.cfg.domain_rand.push_robots:
            push_flag = torch.remainder(torch.tensor(self.common_step_counter), self.push_interval_rand) == 0
            push_ids = push_flag.nonzero(as_tuple=False).flatten()
            self._push_robots_by_id(push_ids)

        self.compute_contacts_and_gaits()

    def compute_terrain_normal_and_distance(self):
        "计算机器人附近地形的法向量"
        if self.cfg.terrain.mesh_type == 'plane':
            # flat terrain
            n_GCS=torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device,requires_grad=False)
            n_GCS[:,2]=-1
            distance = torch.mean(self.root_states[:, 2].unsqueeze(1)-self.measured_heights, dim=1)
            return n_GCS, distance

        coef_GCS, n_GCS=fit_plane(self.terrain_grid_x,
                                self.terrain_grid_y,
                                self.measured_heights)

        DEBUG=False
        if DEBUG:
            # 用两种计算方式验证
            _,n_BCS_yaw=fit_plane(self.height_points[:,:,0],
                                self.height_points[:,:,1],
                                self.measured_heights)
            # GCS -> rotate = BCS
            quat_yaw = get_quat_yaw(self.base_quat)
            n_BCS_yaw2=quat_rotate_inverse(quat_yaw, n_GCS)
            error = n_BCS_yaw - n_BCS_yaw2

        # 计算法线修正后的base_height
        # ax + by + z = c
        # 垂直距离公式: distance = |ax + by - c + z| / sqrt(a^2 + b^2 + 1)
        # 质心COG坐标
        p_x = self.root_states[:, 0]
        p_y = self.root_states[:, 1]
        p_z = self.root_states[:, 2]
        a = coef_GCS[0]
        b = coef_GCS[1]
        c = coef_GCS[2]
        distance = torch.abs(a*p_x + b*p_y - c + p_z) / torch.sqrt(a**2 + b**2 + 1)
        
        return n_GCS, distance

    def compute_contacts_and_gaits(self):
        "update contact and gait buffer"
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.current_contacts_filtered = torch.logical_or(contacts, self.last_contacts) # 两个time step的contact滤波

        #判断模式切换
        phase_change=torch.logical_xor(self.current_contacts_filtered,self.last_contacts_filtered)

        #在phase change的时候保存last air time, last contact time
        save_contact_time=torch.logical_and(phase_change,self.last_contacts_filtered)
        save_air_time=torch.logical_and(phase_change,~self.last_contacts_filtered)
        self.feet_last_air_time[:][save_air_time]=self.feet_air_time[:][save_air_time]
        self.feet_last_contact_time[:][save_contact_time]=self.feet_contact_time[:][save_contact_time]
        
        #当前phase air/contact time更新
        self.feet_air_time += self.dt
        self.feet_air_time[:][self.current_contacts_filtered] = 0
        self.feet_contact_time += self.dt
        self.feet_contact_time[:][~self.current_contacts_filtered] = 0

        # update global variable
        self.last_contacts_filtered=self.current_contacts_filtered
        self.last_contacts = contacts

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        control_type = self.cfg.control.control_type
        if control_type=="P": # 绝对式位置控制器
            if self.cfg.control.use_dof_limit_normalize:
                # action在关节limit内归一化 
                # 零点居中的对称映射
                l_low=self.default_dof_pos-self.dof_pos_limits[:,0]
                l_up=self.dof_pos_limits[:,1]-self.default_dof_pos
                dof_pos_scale=torch.max(l_low,l_up) # limit不对称时 取较大的 不然映射不到
                dof_pos_out = actions * dof_pos_scale.unsqueeze(0) + self.default_dof_pos
            else:
                dof_pos_out = actions * self.cfg.control.action_scale + self.default_dof_pos

            #clip到pos limit
            #pos[env_num,18] limit[18,2]
            self.pos_clipped_flag = torch.logical_or(dof_pos_out>self.dof_pos_limits[:,1],
                                                     dof_pos_out<self.dof_pos_limits[:,0])
            dof_pos_out = torch.clip(dof_pos_out, self.dof_pos_limits[:,0], self.dof_pos_limits[:,1])

            if self.cfg.domain_rand.randomize_motor_offset:
                # 零点不准导致实际给pid的角度有偏差
                dof_pos_out += self.motor_offsets

            if self.cfg.domain_rand.randomize_kpkd_factor:
                torques = self.p_gains_rand*(dof_pos_out - self.dof_pos) - self.d_gains_rand*self.dof_vel
            else:
                torques = self.p_gains*(dof_pos_out - self.dof_pos) - self.d_gains*self.dof_vel

        elif control_type=="P_add": # 增量式位置控制器(比较类似力矩)
            dof_pos_out = actions * self.cfg.control.action_scale + self.dof_pos

            #clip到pos limit
            self.pos_clipped_flag = torch.logical_or(dof_pos_out>self.dof_pos_limits[:,1],
                                                     dof_pos_out<self.dof_pos_limits[:,0])
            dof_pos_out = torch.clip(dof_pos_out, self.dof_pos_limits[:,0], self.dof_pos_limits[:,1])

            if self.cfg.domain_rand.randomize_kpkd_factor:
                torques = self.p_gains_rand*(dof_pos_out - self.dof_pos) - self.d_gains_rand*self.dof_vel
            else:
                torques = self.p_gains*(dof_pos_out - self.dof_pos) - self.d_gains*self.dof_vel

        elif control_type=="V":
            actions_scaled = actions * self.cfg.control.action_scale
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt

        elif control_type=="T":
            actions_scaled = actions * self.cfg.control.action_scale
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        # 计算no clip reward的clip flag
        self.torque_clipped_flag = torques.abs() > self.pid_torque_limits

        # torque clip 防止仿真器的硬限制不靠谱
        torques_clipped = torch.clip(torques, -self.pid_torque_limits, self.pid_torque_limits)

        return torques_clipped

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if len(env_ids)==0:return

        # self.dof_pos[env_ids,:] = 0
        self.dof_pos[env_ids] = self.default_dof_pos.unsqueeze(0) * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids,:] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if len(env_ids)==0:return

        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # root quat随机yaw旋转
        yaw=torch_rand_float_1d(-torch.pi,torch.pi,len(env_ids),device=self.device)
        zero=torch.zeros(len(env_ids),device=self.device)
        q1=quat_from_euler_xyz(zero,zero,yaw)
        q0=self.base_init_state[3:7].unsqueeze(0).repeat(len(env_ids),1)
        q2=quat_mul(q0,q1) #yaw最后在BCS下旋转
        self.root_states[env_ids, 3:7] = q2

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        push_ids = np.arange(self.num_envs)
        self._push_robots_by_id(push_ids)

    def _push_robots_by_id(self,env_ids):
        if len(env_ids)==0: return
        max_vel = self.cfg.domain_rand.max_push_vel_xyz
        v_xyz = torch_rand_float(-max_vel, max_vel, (len(env_ids), 3), device=self.device) 
        v_xyz[:,2] = 0 # 如果不需要z向速度打开这个
        # v_xyz[:,2] = v_xyz[:,2].abs() # z向速度朝下是往地面推 没用

        # 覆盖改成加算 相当于施加一个冲量 更合理
        # self.root_states[env_ids, 7:10] = v_xyz
        self.root_states[env_ids, 7:10] += v_xyz
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # shape: num_envs, 13=3(pos)+4(rot-quat)+3(linear vel)+3(rot vel)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # shape: num_envs, num_dof x 2
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state[:, self.feet_indices, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points() # 机器人要采样的x,y grid(BCS)
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        # gait
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_last_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_last_contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        
        # contact
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.current_contacts_filtered = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts_filtered = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        self.terrain_normal_GCS=torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device,requires_grad=False)

        # hard limit惩罚flag
        self.pos_clipped_flag=torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float,device=self.device,requires_grad=False)
        self.torque_clipped_flag=torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float,device=self.device,requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            func_name = 'reward_' + name
            self.reward_functions.append(getattr(self.reward_module, func_name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.vhacd_enabled = self.cfg.asset.vhacd_enabled
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 500000

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names) # 4x4+1
        self.num_dofs = len(self.dof_names) # 4x3
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.randomize_at_boot()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))) #平方根？
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1) # x,y加随机
            start_pose.p = gymapi.Vec3(*pos) # starred expression是把tensor中的值依次取出来
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i) #随机摩擦
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.actor_name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            max_init_level = min(max_init_level, self.cfg.terrain.num_rows-1)
            if not self.cfg.terrain.init_terrain_at_max_level:
                self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device) # [low,high)
            else:
                self.terrain_levels = torch.ones(size=(self.num_envs,), dtype=torch.int, device=self.device)*max_init_level
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            #平地
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self.gym.clear_lines(self.viewer)

        if self.cfg.debug.plot_heights:
            # 画高度采样点
            if not self.terrain.cfg.measure_heights:
                return
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy() # BCS->GCS
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)
        机器人的地形采样点(BCS)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        # z=points[:, :, 2]=0
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # 机器人的地形采样点(BCS)转换到GCS下
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # 保存GCS网格用于计算法向量
        self.terrain_grid_x=torch.clone(points[:, :, 0])
        self.terrain_grid_y=torch.clone(points[:, :, 1])

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def create_randomize_buffer(self):
        if self.cfg.domain_rand.randomize_kpkd_factor:
            self.p_gains_rand = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device, requires_grad=False)
            self.d_gains_rand = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device, requires_grad=False)

        self.control_latency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_control_latency:
            # 创建传感器delay用的buffer
            max_latency=self.cfg.domain_rand.control_latency_range[1]
            include_obs_steps = int(max_latency/self.dt) + 2
            self.sensor_delayed_dim = self.cfg.env.num_actions*2+3+3
            # 18(q_pos)+18(q_vel)+3(imu_pos)+3(imu_vel)
            self.sensor_obs_delayed_buf = ObservationBuffer3D(
                self.num_envs, self.sensor_delayed_dim,
                include_obs_steps, self.device)
            
        if self.cfg.domain_rand.randomize_apply_force:
            self.applied_force=torch.zeros((self.num_envs,self.num_bodies,3),dtype=torch.float,device=self.device,requires_grad=False)
            self.applied_torque=torch.zeros((self.num_envs,self.num_bodies,3),dtype=torch.float,device=self.device,requires_grad=False)
        
        self.pid_torque_limits=torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float,device=self.device,requires_grad=False)

        if self.cfg.domain_rand.randomize_motor_offset:
            self.motor_offsets=torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float,device=self.device,requires_grad=False)

    def randomize_at_boot(self):
        self.create_randomize_buffer()

        # random once
        if self.cfg.domain_rand.randomize_dof_armature:
            self.dof_armature = torch_rand_float(*self.cfg.domain_rand.armature_range,
                                                 shape=(self.num_envs,self.num_dofs),device=self.device)
        
        if self.cfg.domain_rand.randomize_friction:
            friction_range = self.cfg.domain_rand.friction_range
            restitution_range = self.cfg.domain_rand.restitution_range
            self.friction_coeffs=torch_rand_float_1d(friction_range[0], friction_range[1], self.num_envs, device=self.device)
            self.restitution_coeffs=torch_rand_float_1d(restitution_range[0], restitution_range[1], self.num_envs, device=self.device)

        if self.cfg.domain_rand.randomize_base_mass:
            added_mass_range = self.cfg.domain_rand.added_mass_range
            self.added_mass = torch_rand_float_1d(added_mass_range[0], added_mass_range[1], self.num_envs, device=self.device)

        if self.cfg.domain_rand.push_robots:
            # 随机interval
            upper = self.cfg.domain_rand.push_interval
            lower = upper/2
            self.push_interval_rand = torch.randint(int(lower), int(upper), [self.num_envs], device=self.device)

    def randomize_at_reset(self, env_ids):
        "每次env reset时随机一次"

        if len(env_ids)==0:
            return

        if self.cfg.domain_rand.randomize_motor_offset:
            self.motor_offsets[env_ids, :] = torch_rand_float_1d(*self.cfg.domain_rand.motor_offset_range,
                                                                 shape=len(env_ids),device=self.device)

        #重新生成控制延时
        if self.cfg.domain_rand.randomize_control_latency:
            self.control_latency[env_ids] = torch_rand_float_1d(*self.cfg.domain_rand.control_latency_range,
                                                                shape=len(env_ids),device=self.device)
            self.control_latency[env_ids]=torch.floor(self.control_latency[env_ids]/self.dt)*self.dt #实际实现是整数倍dt的延迟

             
        if self.cfg.domain_rand.randomize_kpkd_factor:
            scale_kp = torch_rand_float(*self.cfg.domain_rand.kp_factor_range,
                                        shape=(len(env_ids),self.num_dofs),device=self.device)
            self.p_gains_rand[env_ids, :] = self.p_gains[:].repeat(len(env_ids),1) * scale_kp
            
            scale_kd = torch_rand_float(*self.cfg.domain_rand.kd_factor_range,
                                        shape=(len(env_ids),self.num_dofs),device=self.device)
            self.d_gains_rand[env_ids, :] = self.d_gains[:].repeat(len(env_ids),1) * scale_kd
            
        if self.cfg.domain_rand.randomize_apply_force:
            base_link_index=0
            self.applied_force[env_ids, base_link_index, :]=torch_rand_float(*self.cfg.domain_rand.force_range,
                                                                             shape=(len(env_ids),3),device=self.device)
            self.applied_torque[env_ids, base_link_index, :]=torch_rand_float(*self.cfg.domain_rand.torque_range,
                                                                              shape=(len(env_ids),3),device=self.device)

        if self.cfg.domain_rand.randomize_limit:
            current_iter=self.common_step_counter//24
            difficulty=(current_iter//1000) /8 # 每1000it 提升1等级 满难度8000iter
            difficulty= min(difficulty, 1.0)
            s=self.cfg.domain_rand.smallest_torque_percent
            upper = self.torque_limits.repeat(len(env_ids),1) * (1-(1-s[1])*difficulty) # 1->s[1]
            lower = self.torque_limits.repeat(len(env_ids),1) * (1-(1-s[0])*difficulty) # 1->s[0]
            self.pid_torque_limits[env_ids, :] = torch.rand((len(env_ids),self.num_dofs),device=self.device) * (upper-lower) + lower
        else:
            self.pid_torque_limits = self.torque_limits.repeat(self.num_envs,1)