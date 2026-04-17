from legged_gym.envs.roll_robot_r.agent_cfg.default import rollRobotRCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

class rollRobotR_imitate_CfgPPO(rollRobotRCfgPPO):

    runner_class_name = 'imitate_Runner'
    class algorithm:
        num_learning_epochs=4
        num_mini_batches=4
        learning_rate=1e-3
        max_grad_norm=1.0
        use_clipped_value_loss=True
        clip_param=0.1
    class runner(rollRobotRCfgPPO.runner):
        max_iterations = 20000
        experiment_name = 'roll_robot_r_imitate'
        teacher_model = os.path.join(LEGGED_GYM_ROOT_DIR,"logs/example/sym4_trot_fixed.pt")
        teacher_actor_hidden_dims=[1024, 512, 256] # locomotion
        teacher_critic_hidden_dims=[1024, 512, 256] # locomotion
        teacher_input_dim = 66
        save_interval = 1000
        use_teacher_act = False
        


