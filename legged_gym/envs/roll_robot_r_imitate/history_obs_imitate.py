import torch
from legged_gym.envs.roll_robot_r.env_cfg.default import rollRobotRCfg
from legged_gym.envs.roll_robot_r.roll_robot_r import rollRobotR
import legged_gym.envs.roll_robot_r_imitate.observations as observations
from legged_gym.envs.base.observation_buffer_3d import ObservationBuffer3D
import numpy as np

class rollRobotR_history_imitate(rollRobotR): # 注意: 优先继承前面的方法
    cfg : rollRobotRCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def compute_observations(self):
        """ Computes observations
        """
        if not self.cfg.env.is_train:
            #play的时候一直读取控制器的指令
            self._resample_commands(list(range(self.num_envs)))
        teacher_obs_buf = torch.cat((
                            observations.obs_dof_pos_normalized(self,add_noise=False) if self.cfg.control.use_dof_limit_normalize else observations.obs_dof_pos(self,add_noise=False), # 18
                            observations.obs_dof_vel(self,add_noise=False),
                            observations.obs_last_actions(self),
                            observations.obs_base_lin_vel(self,add_noise=False),
                            observations.obs_base_ang_vel(self,add_noise=False),
                            observations.obs_projected_gravity(self,add_noise=False),
                            observations.obs_commands(self),
                        ),dim=-1)
        # 计算学生的观测
        add_noise_cfg = self.cfg.noise.add_noise
        obs_real_time = torch.cat((
            observations.obs_dof_pos_normalized(self,add_noise_cfg) if self.cfg.control.use_dof_limit_normalize else observations.obs_dof_pos(self,add_noise_cfg), # 18
            observations.obs_dof_vel(self,add_noise_cfg),
            observations.obs_projected_gravity(self,add_noise_cfg),
            observations.obs_base_ang_vel(self,add_noise_cfg),
        ),dim=-1)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # TCN_INPUT=STATE_DIM+ACTION_DIM # 60 不要给command
        if self.cfg.domain_rand.randomize_control_latency:
            # 把real_time数据放入buf用来计算delayed数据
            self.sensor_obs_delayed_buf.insert(obs_real_time)
            # reset的环境把所有历史数据都置为当前observation
            # TODO: 确认reset_dof,reset_root后obs有更新 否则要考虑加refresh 或者把这个reset放到下一个循环
            self.sensor_obs_delayed_buf.reset(reset_env_ids, obs_real_time[reset_env_ids])

            # 获取
            n_steps_ago = torch.floor(self.control_latency/self.dt).int()
            obs_delayed = self.sensor_obs_delayed_buf.get_obs_by_t(n_steps_ago)

            # 当前帧的tcn_obs 有延迟的传感器数据+无延迟的action和cmd
            student_obs_current=torch.concat((
                obs_delayed,
                observations.obs_last_actions(self),
                observations.obs_commands(self),
            ),dim=-1)
        else:
            # 没有延迟就直接放入实时传感器数据
            student_obs_current=torch.concat((
                obs_real_time,
                observations.obs_last_actions(self),
                observations.obs_commands(self),
            ),dim=-1)

        self.student_obs_history_buf.reset(reset_env_ids, student_obs_current[reset_env_ids])
        self.student_obs_history_buf.insert(student_obs_current)

        student_obs_buf = self.student_obs_history_buf.get_latest_obs(self.cfg.env.include_history_step)
        student_obs_buf = student_obs_buf.flatten(1,2)
        # 正则化
        clip_range = self.cfg.normalization.clip_observations
        teacher_obs_buf = torch.clip(teacher_obs_buf, -clip_range, clip_range)
        student_obs_buf = torch.clip(student_obs_buf, -clip_range, clip_range)


        self.teacher_obs_buf = teacher_obs_buf
        self.student_obs_buf = student_obs_buf


    def _init_buffers(self):
        super()._init_buffers()
        self.last_lin_vel = torch.clone(self.base_lin_vel)
        self.lin_acc = torch.zeros_like(self.last_lin_vel)
        self.teacher_obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_teacher_obs, device=self.device, dtype=torch.float)
        self.student_obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_observations, device=self.device, dtype=torch.float)
        self.student_obs_history_buf = ObservationBuffer3D(self.num_envs, self.cfg.env.single_history_obs_dim, self.cfg.env.include_history_step, self.device)
        self.low_level_controller = torch.jit.load('logs/locomotion.pt').to(self.device)


    def post_physics_step(self):
        env_ids = super().post_physics_step()
        self.lin_acc = (self.base_lin_vel - self.last_lin_vel)/self.dt
        self.last_lin_vel[:] = self.base_lin_vel[:]
        return env_ids
    
    def get_teacher_obs(self):
        return self.teacher_obs_buf
    
    def get_student_obs(self):
        return self.student_obs_buf
    
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def check_termination(self):
        """ Check if environments need to be reset
        """
        long_time_trap = self.trap_static_time > self.cfg.commands.trap_time  # seconds
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # self.reset_buf |= (~self.orientation_good)
        self.reset_buf |= long_time_trap

    def nav_step(self, actions):
        self.commands = actions
        self.compute_observations()
        obs = self.get_student_obs()
        actions = self.low_level_controller(obs)
        self.step(actions)

        self.compute_nav_obs()
        self.compute_nav_rewards()
        self.compute_nav_done()
        return self.nav_obs, self.nav_rewards,self.nav_done,self.nav_extras

    def compute_nav_obs():
        obs = 1
        return obs
    
    def compute_nav_rewards():
        rewards = 1
        return rewards
    
    def compute_nav_done():
        done = 1
        return done




    # def step(action):
    #     # 接收速度命令，即导航策略的action
    #     # 把速度命令导入到底层网络的obs中
    #     # 底层网络决策输出目标关节位置
    #     # 执行目标关节位置命令
    #     # 计算导航策略所需的obs和rew，,,,,,，用来return
    #     self.nav_obs = 
    #     return self.nav_obs