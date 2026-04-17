from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class rollRobotRCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'OnPolicyRunner'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005
        num_learning_epochs = 4
        num_mini_batches = 4 # mini batch size = num_envs * num_steps / num_minibatches
        learning_rate = 3.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99 #discount factor
        lam = 0.95 #GAE
        max_grad_norm = 1.
        max_std = 1.0

    class policy(LeggedRobotCfgPPO.policy):
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_activation='tanh'
        actor_hidden_dims = [1024, 512, 256]
        critic_hidden_dims = [1024, 512, 256]
        init_noise_std=0.3 #训练初始std
        init_std_on_load=False

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'roll_robot_r'
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCritic'
        num_steps_per_env = 24 # per iteration
        max_iterations = 16000 # number of policy updates

        min_normalized_std = [0.0, 0.0, 0.0] * 6 #限制训练能收敛到的最小的std
        mul_std_on_load=1.0

        load_run = -1
        checkpoint = -1
        
        resume = False
        save_interval = 250 # check for potential saves every this many iterations