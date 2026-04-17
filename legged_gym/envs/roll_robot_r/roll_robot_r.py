import numpy as np
import sys
import torch
import threading
import math

from isaacgym.torch_utils import (to_torch, quat_rotate_inverse,
                                  quat_rotate, torch_rand_float, get_axis_params, quat_apply)
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.envs import LeggedRobot
from legged_gym.envs.roll_robot_r.env_cfg.default import rollRobotRCfg
from legged_gym.utils.command_receive import CommandReceiver
from legged_gym.utils.legged_math import quat_apply_yaw, wrap_to_pi, exp_interp
import legged_gym.envs.roll_robot_r.rewards as rewards
import legged_gym.envs.roll_robot_r.observations as observations
from legged_gym.utils.helpers import print_colored


class rollRobotR(LeggedRobot):
    cfg : rollRobotRCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def create_control_panel(self):
        if not self.cfg.env.is_train: # 训练的时候是headless 不需要
            # 创建第二线程来运行命令控制窗口
            self.command_receiver=CommandReceiver()
            self.window_thread = threading.Thread(target=self.command_receiver.create_window)
            self.window_thread.daemon = True #守护线程，主线程退出时，自动关闭窗口
            self.window_thread.start()

    def _init_buffers(self):
        super()._init_buffers()

        # 统计地形等级用的变量
        self.terrain_proportions=self.cfg.terrain.terrain_proportions
        self.terrain_cumulative_proportions = []
        cumulative_probability = 0
        for proportion in self.terrain_proportions:
            cumulative_probability += proportion
            self.terrain_cumulative_proportions.append(cumulative_probability)
        print_colored("=======terrain========")
        print(self.cfg.terrain.terrain_names)
        print(self.cfg.terrain.terrain_proportions)

        self.power_efficiency_buf=torch.zeros((self.num_envs,50),dtype=torch.float,device=self.device,requires_grad=False)
        self.command_level=torch.zeros((self.num_envs),dtype=torch.float,device=self.device,requires_grad=False)
        self.trap_static_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # 计算soft/hard limit
        self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            # soft limits
            upper = self.dof_pos_limits[i, 1] - self.default_dof_pos[i]
            lower = self.dof_pos_limits[i, 0] - self.default_dof_pos[i]
            hard_percent = 0.99
            soft_percent = self.cfg.rewards.soft_dof_pos_percent
            # 以default_dof_pos为中心
            self.soft_dof_pos_limits[i, 0] = self.default_dof_pos[i] + lower * soft_percent
            self.soft_dof_pos_limits[i, 1] = self.default_dof_pos[i] + upper * soft_percent
            self.hard_dof_pos_limits[i, 0] = self.default_dof_pos[i] + lower * hard_percent
            self.hard_dof_pos_limits[i, 1] = self.default_dof_pos[i] + upper * hard_percent
        
        self.stand_joint_pos=torch.tensor(self.cfg.init_state.stand_joint_pos,dtype=torch.float,device=self.device,requires_grad=False)

        self.orientation_good=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.height_good=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.pose_good=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    def _resample_commands(self, env_ids):
        """ 重写父类,原本是随机生成指令【x速度,y速度,旋转速度,机头朝向】
        改为从控制器读取指令，影响运行速度
        所以只在play时采用,train时按照原方式运行

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids)==0:
            return
        
        if self.cfg.env.is_train:
            # 第一次随机全范围
            # 有可能随机到lin和ang都很小(stand cmd)的情况
            # lin_vel圆形分布,先随机大小,后随机方向
            lin_cmd_upper=self.cfg.commands.ranges.max_curriculum*self.command_level[env_ids]
            lin_cmd_lower=torch.zeros_like(lin_cmd_upper,device=self.device)
            r1=torch_rand_float(0,1, (len(env_ids), 1), device=self.device).squeeze(1)
            lin_vel=lin_cmd_lower+(lin_cmd_upper-lin_cmd_lower)*r1
            angle=torch_rand_float(0,2*torch.pi,(len(env_ids), 1),device=self.device).squeeze(1)
            self.commands[env_ids, 0] = torch.cos(angle)*lin_vel
            self.commands[env_ids, 1] = torch.sin(angle)*lin_vel
            
            # ang_vel
            self.commands[env_ids, 2] = torch_rand_float(-self.cfg.commands.ranges.max_ang_vel_yaw,
                                                          self.cfg.commands.ranges.max_ang_vel_yaw,
                                                          (len(env_ids), 1), device=self.device).squeeze(1)
            
            if not self.cfg.commands.generate_stand_cmd:
                # 判断stand cmd
                lin_cmd_small = lin_vel < self.cfg.commands.cmd_stand_lin
                ang_cmd_small = self.commands[env_ids, 2].abs() < self.cfg.commands.cmd_stand_ang
                stand_cmd_id = torch.logical_and(lin_cmd_small,ang_cmd_small).nonzero(as_tuple=False).flatten()
                stand_cmd_env_id = env_ids[stand_cmd_id]
                if len(stand_cmd_env_id)!=0:
                    # 重新随机这部分, 不要随机到stand
                    lin_cmd_upper=self.cfg.commands.ranges.max_curriculum*self.command_level[stand_cmd_env_id]
                    lin_cmd_lower=torch.zeros_like(lin_cmd_upper,device=self.device) + self.cfg.commands.cmd_stand_lin
                    r1=torch_rand_float(0,1, (len(stand_cmd_env_id), 1), device=self.device).squeeze(1)
                    lin_vel=lin_cmd_lower+(lin_cmd_upper-lin_cmd_lower)*r1
                    angle=torch_rand_float(0,2*torch.pi,(len(stand_cmd_env_id), 1),device=self.device).squeeze(1)
                    self.commands[stand_cmd_env_id, 0] = torch.cos(angle)*lin_vel
                    self.commands[stand_cmd_env_id, 1] = torch.sin(angle)*lin_vel
                    
                    r2 = torch_rand_float(self.cfg.commands.cmd_stand_ang,
                                          self.cfg.commands.ranges.max_ang_vel_yaw, (len(stand_cmd_env_id), 1), device=self.device).squeeze(1)
                    s = torch.rand(len(stand_cmd_env_id), device=self.device) > 0.5
                    r2[s] *= -1.0 # 随机ang_vel取反
                    self.commands[stand_cmd_env_id, 2] = r2
            else:
                # 需要训练stand cmd,但是自然随机到的概率很小,增加一些
                zero_flag = torch.rand(len(env_ids),device=self.device) < self.cfg.commands.stand_prob
                zero_envs = env_ids[zero_flag]
                self.commands[zero_envs,:3] = 0
        else:
            #从控制器输入
            x,y,r,h=self.command_receiver.get_values()
            self.commands[env_ids, :3] = torch.tensor([x,y,r],device=self.device)

    def compute_observations(self):
        """ Computes observations
        """
        if not self.cfg.env.is_train:
            #play的时候一直读取控制器的指令
            self._resample_commands(list(range(self.num_envs)))

        add_noise = self.cfg.noise.add_noise
        locomotion_obs_buf = torch.cat((
                            observations.obs_dof_pos_normalized(self,add_noise) if self.cfg.control.use_dof_limit_normalize else observations.obs_dof_pos(self), # 18
                            observations.obs_dof_vel(self,add_noise), # 18
                            observations.obs_last_actions(self), # 18
                            observations.obs_base_lin_vel(self,add_noise), # 3
                            observations.obs_base_ang_vel(self,add_noise), # 3
                            observations.obs_projected_gravity(self,add_noise), # 3
                            observations.obs_commands(self), # 3
                        ),dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.add_height_observation:
            locomotion_obs_buf = torch.cat((locomotion_obs_buf, observations.obs_measure_heights(self,add_noise)), dim=-1)

        clip_range = self.cfg.normalization.clip_observations
        locomotion_obs_buf = torch.clip(locomotion_obs_buf, -clip_range, clip_range)

        self.obs_buf = locomotion_obs_buf
        self.locomotion_obs_group={
            "policy":locomotion_obs_buf,
            "critic_obs":locomotion_obs_buf,
        }
    
    def update_command_curriculum(self, env_ids):
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        track_reward=self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length / self.reward_scales["tracking_lin_vel"]/ self.cfg.rewards.reward_global_coef
        move_up = track_reward> 0.8
        move_down = track_reward< 0.2
        self.command_level[env_ids]=self.command_level[env_ids]+move_up*0.1-move_down*0.1
        self.command_level[env_ids]=self.command_level[env_ids].clip(min=0.0,max=1.0)

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
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)
        else:
            self.command_level[:] = 1.0
        
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
        self.power_efficiency_buf[env_ids] = 0
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['reward/' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain/terrain_level"] = torch.mean(self.terrain_levels.float())
            terrain_names=self.cfg.terrain.terrain_names
            for terrain_id in range(len(terrain_names)):
                name1="terrain/mean_Lv_"+terrain_names[terrain_id]
                name2="terrain/max_Lv_"+terrain_names[terrain_id]
                up=self.num_envs*self.terrain_cumulative_proportions[terrain_id]
                low=up-self.num_envs*self.terrain_proportions[terrain_id]
                if up-low<=1: continue
                self.extras["episode"][name1]=torch.mean(self.terrain_levels[int(low):int(up)].float())
                self.extras["episode"][name2]=torch.max(self.terrain_levels[int(low):int(up)].float())
        
        command_norm=self.commands[:,:2].norm(dim=1)
        self.extras["episode"]["command/max_command"] = torch.max(command_norm)
        self.extras["episode"]["command/mean_command"] = torch.mean(command_norm)
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
            if (self.cfg.rewards.reward_curriculum)and(name in self.cfg.rewards.curriculum_reward_list):
                current_iter=self.common_step_counter//24
                difficulty=float(current_iter//1000) /8 # 每1000it 提升1等级 满难度4000iter
                difficulty=min(difficulty,1.0) # 0~1
                scale=exp_interp(low=0.05,high=1.0,x=difficulty)
                rew*=scale

            # stand情况下屏蔽不在stand列表中的reward
            if name not in self.cfg.rewards.stand_reward_list:
                rew[self.stand]=0

            # 全局系数
            if hasattr(self.cfg.rewards, "reward_global_coef"):
                rew *= self.cfg.rewards.reward_global_coef

            self.rew_buf += rew
            self.episode_sums[name] += rew

        # if self.cfg.rewards.only_positive_rewards:
        #     self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = rewards.reward_termination(self) * self.reward_scales["termination"]
            if hasattr(self.cfg.rewards, "reward_global_coef"):
                rew *= self.cfg.rewards.reward_global_coef
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        #这里用到原点距离不准确，一初始点位有随机，二如果有rotate command 就会边走边转 直线距离不等于路程
        # distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)

        # move_up = (distance > self.cfg.commands.max_curriculum*self.max_episode_length_s*0.6) 
        # move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.2) * ~move_up

        track_reward=self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length / self.reward_scales["tracking_lin_vel"]/ self.cfg.rewards.reward_global_coef
        time_fix=0.7 # 整个episode求和因为resample等因素导致的系数
        error_std=(-torch.log((track_reward/time_fix).clip(max=1.0))).sqrt()
        move_up = error_std < 0.5
        move_down = error_std > 1.1
        # e^(-x^2)值和std表
        # 0.95 -> std=0.22
        # 0.9 -> std=0.32
        # 0.8 -> std=0.47
        # 0.7 -> std=0.59
        # 0.6 -> std=0.71
        # 0.5 -> std=0.83
        # 0.4 -> std=0.95
        # 0.3 -> std=1.09
        # 0.2 -> std=1.26
        # 0.1 -> std=1.51

        #如果command level没满 也不要动地形等级
        command_level_complete=self.command_level[env_ids]>0.95
        move_up=torch.logical_and(move_up,command_level_complete)
        move_down=torch.logical_and(move_down,command_level_complete)

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        long_time_trap = self.trap_static_time > self.cfg.commands.trap_time  # seconds
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= (~self.orientation_good)
        self.reset_buf |= long_time_trap

    def _process_dof_props(self, props, env_id):
        # dof_pos_limit存放urdf的limit
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
        return props
    
    def draw_debug_vis(self):
        super().draw_debug_vis()

        if self.cfg.debug.plot_normals:
            # 地形法线和机身法线
            v1=self.terrain_normal_GCS
            v2=body_normal_GCS=quat_rotate(self.base_quat,self.gravity_vec)
            root_pos = self.root_states[:, :3]

            line_length=-0.5
            l1_start=root_pos
            l1_end=root_pos+v1*line_length
            l2_start=root_pos
            l2_end=root_pos+v2*line_length
            line_vertices=torch.concat([l1_start,l1_end,l2_start,l2_end],dim=1).reshape(self.num_envs*4,3) # start end相邻放

            num_lines = line_vertices.shape[0]//2

            line_colors_v1 =torch.zeros((self.num_envs,3),dtype=torch.float32)
            line_colors_v1[:,0]=1.0 # red
            line_colors_v2 =torch.zeros((self.num_envs,3),dtype=torch.float32)
            line_colors_v2[:,1]=1.0 # green
            line_colors=torch.concat([line_colors_v1,line_colors_v2],dim=1).reshape(self.num_envs*2,3)
            # line_vertices:[num_lines * 2 , 3] ndarray线段起点和终点
            # line_colors:[num_lines, 3] RGB颜色0~1
            self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().numpy(), line_colors.cpu().numpy())
    
    def _create_envs(self):
        super()._create_envs()

        penalized_self_collision_names = []
        for name in self.cfg.asset.penalize_self_collision:
            penalized_self_collision_names.extend([s for s in self.body_names if name in s])

        self.penalised_self_collision_indices = torch.zeros(len(penalized_self_collision_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_self_collision_names)):
            self.penalised_self_collision_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_self_collision_names[i])

        # actor armature randomize
        if self.cfg.domain_rand.randomize_dof_armature:
            dof_armature = self.dof_armature.cpu().numpy()

            for i in range(self.num_envs):
                env_handle=self.envs[i]
                actor_handle=self.actor_handles[i]
                actor_dof_properties = self.gym.get_actor_dof_properties(env_handle,actor_handle)
                actor_dof_properties["armature"] = dof_armature[i]
                self.gym.set_actor_dof_properties(env_handle,actor_handle,actor_dof_properties)

    def _process_rigid_shape_props(self, props, env_id):
        "只在创建环境时调用, 每创建一个env, 调用一次"
        if self.cfg.domain_rand.randomize_friction:
            for element in props:
                element.friction = self.friction_coeffs[env_id]
                element.restitution = self.restitution_coeffs[env_id]
                element.rolling_friction = self.cfg.terrain.rolling_friction
                element.torsion_friction = self.cfg.terrain.torsion_friction
        return props

    def _prepare_reward_function(self):
        self.reward_module = rewards
        LeggedRobot._prepare_reward_function(self)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
        """
        LeggedRobot._post_physics_step_callback(self)

        # 更新trap_time
        lin_trap_static = torch.logical_and(torch.norm(self.base_lin_vel[:, :2], dim=-1) < self.cfg.commands.trap_vel,
                                            torch.norm(self.commands[:, :2], dim=-1) > self.cfg.commands.trap_vel)
        ang_trap_static = torch.logical_and(torch.abs(self.base_ang_vel[:, 2]) < self.cfg.commands.trap_vel,
                                            torch.abs(self.commands[:, 2]) > self.cfg.commands.trap_vel)
        trap_env_mask = torch.logical_or(lin_trap_static, ang_trap_static)
        self.trap_static_time[trap_env_mask] += self.dt
        self.trap_static_time[~trap_env_mask] = 0.
        
        # 计算姿态/站立高度
        self.angle_rad=torch.acos(torch.sum(self.body_normal_GCS*self.terrain_normal_GCS,dim=1).clip(min=-1,max=1))
        self.angle_deg=torch.rad2deg(self.angle_rad) # 0=正常站立 180=倒立
        self.orientation_good=self.angle_deg<self.cfg.rewards.orientation_threshold
        self.height_good=self.base_height>self.cfg.rewards.base_height_threshold
        self.pose_good=(self.orientation_good &
                        self.height_good)

        lin_cmd_small = self.commands[:, :2].norm(dim=1) < self.cfg.commands.cmd_stand_lin
        ang_cmd_small = self.commands[:, 2].abs() < self.cfg.commands.cmd_stand_ang
        self.cmd_small = torch.logical_and(lin_cmd_small,ang_cmd_small)

        self.stand = torch.logical_and(self.cmd_small, self.pose_good)

        # 更新power efficiency
        power=torch.clip(self.dof_vel*self.torques,min=0.) # 过滤负功
        power_all=torch.sum(power,dim=1)
        self.power_efficiency_buf = self.power_efficiency_buf.roll(shifts=1,dims=1)
        self.power_efficiency_buf[:,0]=power_all