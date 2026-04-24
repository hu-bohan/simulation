from legged_gym.envs.roll_robot_r_imitate.env_cfg.history_obs_imitate_cfg import (
    rollRobotR_history_imitate_Cfg,
)


class rollRobotR_hierarchical_nav_Cfg(rollRobotR_history_imitate_Cfg):
    class env(rollRobotR_history_imitate_Cfg.env):
        num_envs = 64
        num_observations = 8 + 4 * 4
        episode_length_s = 30
        include_history_steps = None

    class terrain(rollRobotR_history_imitate_Cfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = True
        add_height_observation = False

    class commands(rollRobotR_history_imitate_Cfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 0.1

    class domain_rand(rollRobotR_history_imitate_Cfg.domain_rand):
        push_robots = False
        randomize_apply_force = False

    class debug(rollRobotR_history_imitate_Cfg.debug):
        plot_navigation = True

    class navigation:
        num_nav_actions = 2
        num_nearest_obstacles = 4
        num_obstacles = 4
        num_nav_observations = 8 + num_nearest_obstacles * 4

        field_length = 24.0
        field_width = 12.0
        goal_tolerance = 0.75
        out_of_bounds_margin = 0.6

        path_type = "sine"  # "line" or "sine"
        path_resolution = 0.25
        path_center_y = field_width * 0.5
        path_amplitude = 1.0
        path_wavelength = 8.0
        lookahead_steps = 6

        observation_radius = 6.0
        safe_margin = 0.9
        robot_radius = 0.45

        obstacle_x_range = [4.0, 18.0]
        obstacle_radius_range = [0.45, 0.85]
        obstacle_path_bias = 0.75
        obstacle_path_offset = 1.2
        obstacle_margin = 0.9
        obstacle_min_spacing = 1.6

        max_forward_command = 0.5
        max_yaw_command = 1.0
        command_smoothing = 0.8
        speed_observation_scale = 0.8

        reward_progress = 6.0
        reward_track_recovery = 0.8
        penalty_track = 0.08
        reward_heading = 0.4
        reward_clearance = 0.05
        penalty_collision = 120.0
        penalty_out_of_bounds = 80.0
        reward_goal = 150.0
        step_penalty = 0.02

        recovery_trigger_gravity_z = -0.8
        recovery_release_gravity_z = -0.95
        recovery_settle_ang_vel = 0.5
        protective_pose = [0.0, -0.2, -0.8]

        locomotion_policy_path = "logs/locomotion.pt"
        recovery_policy_path = "logs/recovery.pt"
        nav_policy_path = "logs/nav/td3_ship_best_actor.pt"
