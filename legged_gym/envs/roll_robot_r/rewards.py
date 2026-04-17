import torch
from isaacgym.torch_utils import quat_rotate
from legged_gym.envs.base.rewards import * # 把base环境的reward函数import过来

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from legged_gym.envs.roll_robot_r.roll_robot_r_amp import rollRobotRAMP as env1
else: # 运行的时候如果直接import会循环引用
    from legged_gym.envs.base.base_task import BaseTask as env1

def reward_dof_pos_limits(env: env1):
    # Penalize dof positions too close to the limit
    flag1=env.dof_pos > env.soft_dof_pos_limits[:,1]
    flag2=env.dof_pos < env.soft_dof_pos_limits[:,0]

    #硬限制
    flag3=env.dof_pos > env.hard_dof_pos_limits[:,1]
    flag4=env.dof_pos < env.hard_dof_pos_limits[:,0]

    #硬限制x3
    out_of_limits=flag1+flag2+(flag3+flag4)*10.0
    penalty=torch.sum(out_of_limits,dim=1) # 18个dof求和 
    penalty/=18 #soft 18 hard 72 归一化

    return penalty

def reward_dof_position(env: env1):
    #关节相对target位置的误差
    error=torch.sum(torch.square(env.dof_pos-env.stand_joint_pos),1)
    return error

def reward_dof_vel_limits(env: env1):
    # Penalize dof velocities too close to the limit
    soft_percent=env.cfg.rewards.soft_dof_vel_percent
    hard_percent=0.95
    soft_limit = soft_percent*env.dof_vel_limits
    hard_limit = hard_percent*env.dof_vel_limits

    flag1 = torch.abs(env.dof_vel) > soft_limit
    flag2 = torch.abs(env.dof_vel) > hard_limit

    out_of_limits=flag1+flag2*10.0
    penalty=torch.sum(out_of_limits,dim=1) # 18个dof求和 
    penalty/=18 #soft 18 hard 72 归一化

    return penalty

def reward_torque_limits(env: env1):
    # penalize torques too close to the limit
    soft_percent=env.cfg.rewards.soft_torque_percent
    hard_percent=0.95
    soft_limit = soft_percent*env.torque_limits
    hard_limit = hard_percent*env.torque_limits

    flag1 = torch.abs(env.torques) > soft_limit
    flag2 = torch.abs(env.torques) > hard_limit

    out_of_limits=flag1+flag2*10.0
    penalty=torch.sum(out_of_limits,dim=1) # 18个dof求和 
    penalty/=18 #soft 18 hard 72 归一化

    return penalty

def reward_cot(env: env1):
    #功耗越大，惩罚越大
    power_efficiency_average=torch.mean(env.power_efficiency_buf,dim=1)

    mass=12;g=9.8
    vel=torch.sqrt(torch.sum(torch.square(env.base_lin_vel[:, :2]),dim=1))
    cot=power_efficiency_average/(mass*g*vel)
    return cot

def reward_power_consumption(env: env1):
    #功耗
    mech_power=torch.clip(env.dof_vel*env.torques,min=0.) #注意过滤负功项
    hot_power=torch.square(env.torques * 1.08) * 1.8 # P=I^2 R
    # 堵转12Nm=13A=300W, k=1.08, R=1.8ohm

    power_all=torch.sum(mech_power+hot_power,dim=1)
    return power_all
    
def reward_power_peak(env: env1):
    #惩罚功率峰值
    power_efficiency_average=torch.mean(env.power_efficiency_buf,dim=1)
    power=torch.clip(env.dof_vel*env.torques,min=0.) #注意过滤负功项
    instant_power=torch.sum(power,dim=1)
    #计算瞬时功率超出平均功率的比例
    peak_ratio=(instant_power-power_efficiency_average)/power_efficiency_average
    #过滤小于平均功率的值
    peak_ratio=peak_ratio.clip(min=0.,max=5.)

    #episode刚开始,平均功率还不准
    episode_start=env.episode_length_buf<25
    peak_ratio[episode_start]=0

    return peak_ratio

def reward_tracking_lin_vel(env: env1):
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error/env.cfg.rewards.tracking_lin_vel_std)

def reward_tracking_ang_vel(env: env1):
    # Tracking of angular velocity commands (yaw) 
    ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error/env.cfg.rewards.tracking_ang_vel_std)

def reward_feet_contact_forces_z(env: env1):
    #惩罚腿和地面的z向接触力
    forces_z=torch.clip(env.contact_forces[:, env.feet_indices, 2],min=0) #z up
    #正常站立时的足端力12kg*9.8=117.6
    force_bias=12*9.8
    all_forces=torch.sum(forces_z,dim=1)-force_bias
    return all_forces

def reward_trap_static(env: env1):
    return env.trap_static_time.clip(max=env.cfg.commands.trap_time)

def reward_self_collision(env: env1):
    return torch.sum(1.*(torch.norm(env.contact_forces[:, env.penalised_self_collision_indices, :], dim=-1) > 0.1), dim=1)

def reward_no_clip(env: env1):
    penalty1 = torch.sum(env.pos_clipped_flag, dim=1)
    penalty2 = torch.sum(env.torque_clipped_flag, dim=1)
    return penalty1 + penalty2

def reward_feet_air_time(env: env1):
    "奖励step time越长越好"
    # 判断全触地相和全腾空相
    ground_phase = torch.all(env.current_contacts_filtered==True,dim=1)
    air_phase = torch.all(env.current_contacts_filtered==False,dim=1)
    walk_phase = ~torch.logical_or(ground_phase,air_phase)

    # 奖励行走周期(腾空时间+触地时间)越长越好
    step_time = env.feet_last_air_time.clip(max=1.0) + env.feet_last_contact_time.clip(max=1.0) # 这个会训练出对角步态 50%占空比
    # step_time = env.feet_last_air_time.clip(max=0.5) + env.feet_last_contact_time.clip(max=0.5) # 这个是高速2-4步态 33%占空比
    reward = torch.mean(step_time, dim=1)
    # rew_airTime = torch.mean(env.feet_last_air_time.clip(max=1.0), dim=1) # 只奖励腾空时间
    reward[~walk_phase] = 0
    return reward

def reward_gait_std(env:env1):
    # 每个腿的步态统计量应该一样 防止有几个腿一直抬着或者触地 (这个也有助于对称性)
    gait_std = torch.std(env.feet_last_air_time,dim=1) + \
               torch.std(env.feet_last_contact_time,dim=1)
    return gait_std

def reward_gait_duty(env:env1):
    # contact/(air+contact) 
    # duty 5-1=0.84, 4-2=0.66(slow), trot=0.5, 2-4=0.33(fast), 1-5=0.16
    gait_type_duty = env.cfg.rewards.gait_type_duty
    duty = env.feet_last_contact_time / (env.feet_last_air_time + env.feet_last_contact_time + 1e-6)
    duty_error = ((duty-gait_type_duty)/0.16666).square().mean(dim=1)

    # type 5, 4, 3 ,2 ,1
    # gait_type = (duty/0.16666).round()
    # duty_error = (gait_type!=3).mean(dim=1,dtype=torch.float32)
    return duty_error

def reward_stand_feet_on_ground(env: env1):
    #要求六个脚在静止时触地
    all_contact=torch.sum(env.current_contacts_filtered,dim=1)/6 #0~6 -> 0~1
    # all_contact[all_contact<0.5]=0 #小于一半腿触地不稳定
    
    all_contact[~env.stand]=0 # 只在静止时计算
    return all_contact

def reward_stand_dof_pos(env: env1):
    # 站立时维持default姿态
    # 正常行走时
    # hip [-0.6,0.6]
    # thigh [-0.2,0.2]
    # shank [-0.6,0.4]
    std = [0.1, 0.05, 0.1]*6 # 不同关节不同dof_pos_std
    std = torch.tensor(std, device=env.device)

    error = (env.dof_pos - env.stand_joint_pos)/std
    # reward = torch.exp(-error.square()).mean(dim=-1) # exp激活 [0,1]
    reward = -error.square().mean(dim=-1) # 平方项惩罚

    reward[~env.stand]=0
    return reward

def reward_stand_dof_vel(env: env1):
    #惩罚结束时还在动的关节
    std=0.5

    error=env.dof_vel/std
    # reward=torch.exp(-error.square()).mean(dim=-1)
    reward = -error.square().mean(dim=-1)

    reward[~env.stand]=0
    return reward

def reward_orientation(env: env1):
    """崎岖地形的机身角度 1-(cosa)^2=(sina)^2 \\
    注意:不适用与angle>90度倒立的情况
    """
    # angle_rad=torch.acos(torch.sum(env.body_normal_GCS*env.terrain_normal_GCS,dim=1).clip(min=-1,max=1))
    # angle = angle_rad*180/torch.pi #正立0 倒立180
    # angle_std = env.cfg.rewards.orientation_std
    # reward = torch.exp(-(angle/angle_std).square())

    # error = torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1) # x^2+y^2 = (sina)^2

    angle_cos=torch.sum(env.body_normal_GCS*env.terrain_normal_GCS,dim=1)
    error = 1 - angle_cos.square()
    return error

def reward_base_height(env: env1):
    "崎岖地形的高度reward"
    base_height = env.base_height
    return torch.square(base_height - env.cfg.rewards.base_height_target)