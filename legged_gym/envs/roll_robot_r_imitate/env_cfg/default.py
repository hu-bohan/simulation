from legged_gym.envs.roll_robot_r.env_cfg.default import rollRobotRCfg

class rollRobotR_imitate_Cfg(rollRobotRCfg):
    class env(rollRobotRCfg.env):
        num_envs =2048
        num_observations = 63
        num_teacher_obs = 66
    class noise(rollRobotRCfg.noise):

        class noise_scales(rollRobotRCfg.noise.noise_scales):
            lin_acc = 0.05

    class normalization(rollRobotRCfg.normalization):

        class obs_scales(rollRobotRCfg.normalization.obs_scales):
            lin_acc = 0.005
                 