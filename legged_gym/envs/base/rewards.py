import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from legged_gym.envs.base.legged_robot import LeggedRobot as env1
else: # 运行的时候如果直接import会循环引用
    from legged_gym.envs.base.base_task import BaseTask as env1

def reward_lin_vel_z(env: env1):
    # Penalize z axis base linear velocity
    return torch.square(env.base_lin_vel[:, 2])

def reward_ang_vel_xy(env: env1):
    # Penalize xy axes base angular velocity
    return torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

def reward_orientation(env: env1):
    # Penalize non flat base orientation
    return torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)

def reward_base_height(env: env1):
    # Penalize base height away from target
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    return torch.square(base_height - env.cfg.rewards.base_height_target)

def reward_torques(env: env1):
    # Penalize torques
    return torch.sum(torch.square(env.torques), dim=1)

def reward_dof_vel(env: env1):
    # Penalize dof velocities
    return torch.sum(torch.square(env.dof_vel), dim=1)

def reward_dof_acc(env: env1):
    # Penalize dof accelerations
    dof_acc = ((env.last_dof_vel - env.dof_vel) / env.dt)
    return torch.sum(torch.square(dof_acc), dim=1)

def reward_action_rate(env: env1):
    # Penalize changes in actions
    diff_1 = env.last_actions - env.actions
    return torch.sum(torch.square(diff_1),dim=1)

def reward_collision(env: env1):
    # Penalize collisions on selected bodies
    force = torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1)
    penalty = torch.sum(1.*(force > 0.1), dim=1)
    return penalty

def reward_termination(env: env1):
    # Terminal reward / penalty
    return env.reset_buf * ~env.time_out_buf # reset 但不是因为time out而reset(意外reset)

def reward_dof_pos_limits(env: env1):
    # Penalize dof positions too close to the limit
    out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)

def reward_dof_vel_limits(env: env1):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    return torch.sum((torch.abs(env.dof_vel) - env.dof_vel_limits*env.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

def reward_torque_limits(env: env1):
    # penalize torques too close to the limit
    return torch.sum((torch.abs(env.torques) - env.torque_limits*env.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

def reward_tracking_lin_vel(env: env1):
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error/env.cfg.rewards.tracking_sigma)

def reward_tracking_ang_vel(env: env1):
    # Tracking of angular velocity commands (yaw) 
    ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error/env.cfg.rewards.tracking_sigma)

def reward_feet_air_time(env: env1):
    # Reward long steps
    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    contact = env.contact_forces[:, env.feet_indices, 2] > 1.
    contact_filt = torch.logical_or(contact, env.last_contacts) 
    env.last_contacts = contact
    first_contact = (env.feet_air_time > 0.) * contact_filt
    env.feet_air_time += env.dt
    rew_airTime = torch.sum((env.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #腾空时间>0.5s reward+ <0.5s reward- 奖励较长的摆动相
    rew_airTime *= torch.norm(env.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    env.feet_air_time *= ~contact_filt
    return rew_airTime

def reward_stumble(env: env1):
    # Penalize feet hitting vertical surfaces
    return torch.any(torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2) >\
            5 *torch.abs(env.contact_forces[:, env.feet_indices, 2]), dim=1)
    
def reward_stand_still(env: env1):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(env.dof_pos - env.default_dof_pos), dim=1) * (torch.norm(env.commands[:, :2], dim=1) < 0.1)

def reward_feet_contact_forces(env: env1):
    # penalize high contact forces
    return torch.sum((torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1) -  env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)