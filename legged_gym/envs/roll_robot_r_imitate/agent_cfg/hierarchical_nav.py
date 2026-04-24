from legged_gym.envs.roll_robot_r.agent_cfg.default import rollRobotRCfgPPO


class rollRobotR_hierarchical_nav_CfgPPO(rollRobotRCfgPPO):
    class runner(rollRobotRCfgPPO.runner):
        experiment_name = "roll_robot_r_hierarchical_nav"
        run_name = ""
