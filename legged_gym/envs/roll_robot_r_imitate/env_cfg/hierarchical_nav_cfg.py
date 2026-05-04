from legged_gym.envs.roll_robot_r_imitate.env_cfg.history_obs_imitate_cfg import (
    rollRobotR_history_imitate_Cfg,
)


class rollRobotR_hierarchical_nav_Cfg(rollRobotR_history_imitate_Cfg):
    class env(rollRobotR_history_imitate_Cfg.env):
        num_envs = 64
        num_observations = 8 + 4 * 4
        episode_length_s = 180
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

        min_forward_command = 0.0
        max_forward_command = 0.38
        max_yaw_command = 0.65
        command_smoothing = 0.8
        linear_command_smoothing = 0.65
        yaw_command_smoothing = 0.4
        speed_observation_scale = 0.8
        cruise_forward_command = 0.12
        cruise_clearance = 1.0

        path_following_enabled = True
        path_follow_heading_gain = 1.2
        path_follow_track_gain = 0.35
        path_follow_policy_clearance = 0.8
        path_follow_full_clearance = 2.2

        safety_shield_enabled = True
        safety_front_distance = 3.5
        safety_lateral_margin = 0.35
        safety_radius_buffer = 0.2
        safety_stop_clearance = -0.05
        safety_slow_clearance = 0.8
        safety_min_speed_scale = 0.18
        turn_speed_reduction = 0.35
        min_turn_speed_scale = 0.55

        use_terrain_mesh_obstacles = False
        terrain_obstacle_seed = None
        terrain_obstacle_height = 0.8
        terrain_obstacle_segments = 24
        terrain_nav_start_margin = 2.0
        terrain_nav_end_margin = 2.0
        terrain_nav_side_margin = 2.0
        terrain_obstacle_layout = None

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

        terminate_on_body_contact = False
        terminate_on_timeout = True
        terminate_on_trap = False
        terminate_on_collision = True
        terminate_on_goal = True
        terminate_on_out_of_bounds = True

        locomotion_policy_path = "logs/locomotion.pt"
        recovery_policy_path = "logs/recovery.pt"
        nav_policy_path = "logs/nav/td3_ship_best_actor.pt"
