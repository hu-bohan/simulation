from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .roll_robot_r.roll_robot_r import rollRobotR
from .roll_robot_r.env_cfg.default import rollRobotRCfg
from .roll_robot_r.agent_cfg.default import rollRobotRCfgPPO

import os
from legged_gym.utils.task_registry import task_registry

task_registry.register("roll_robot_r", rollRobotR, rollRobotRCfg(), rollRobotRCfgPPO())

# 教师学生模仿学习
#学生采用单帧和多帧本体感知输入，模仿行走
from .roll_robot_r_imitate.agent_cfg.imitate import rollRobotR_imitate_CfgPPO
from .roll_robot_r_imitate.env_cfg.history_obs_imitate_cfg import  rollRobotR_history_imitate_Cfg
from .roll_robot_r_imitate.history_obs_imitate import rollRobotR_history_imitate
task_registry.register("roll_robot_r_history_imitate", rollRobotR_history_imitate, rollRobotR_history_imitate_Cfg(), rollRobotR_imitate_CfgPPO())
