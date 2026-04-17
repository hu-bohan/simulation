from legged_gym.envs.roll_robot_r.env_cfg.default import rollRobotRCfg

class rollRobotR_history_imitate_Cfg(rollRobotRCfg):
    class env(rollRobotRCfg.env):
        num_envs =2048
        num_observations = (18+18+3+3+18+3)*8 
        single_history_obs_dim = (18+18+3+3+18+3)
        num_teacher_obs = 66
        sample_history_obs_length = 8
        include_history_step = 8

    class noise(rollRobotRCfg.noise):
        class noise_scales(rollRobotRCfg.noise.noise_scales):
            lin_acc = 0.05
    class terrain(rollRobotRCfg.terrain):
        mesh_type = 'plane'
    class normalization(rollRobotRCfg.normalization):
        class obs_scales(rollRobotRCfg.normalization.obs_scales):
            lin_acc = 0.005
    
    class domain_rand(rollRobotRCfg.domain_rand):
        randomize_control_latency = False
        control_latency_range = [0.00, 0.081]  # 取整原因0.08必须多一点
        smallest_torque_percent = [0.85,1.0] # [12,12] -> [10,12]
                 