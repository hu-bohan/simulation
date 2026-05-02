import math

import numpy as np
import torch
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul, torch_rand_float

import legged_gym.envs.roll_robot_r_imitate.observations as observations
from legged_gym.envs.base.observation_buffer_3d import ObservationBuffer3D
from legged_gym.envs.roll_robot_r.roll_robot_r import rollRobotR
from legged_gym.utils.legged_math import wrap_to_pi


class rollRobotR_hierarchical_nav(rollRobotR):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _prepare_reward_function(self):
        self.reward_functions = []
        self.reward_names = [
            "nav_total",
            "nav_progress",
            "nav_track_recovery",
            "nav_track_penalty",
            "nav_heading",
            "nav_clearance",
            "nav_goal",
            "nav_collision",
            "nav_out_of_bounds",
        ]
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_names
        }

    def _init_buffers(self):
        super()._init_buffers()

        self.student_obs_buf = torch.zeros(
            self.num_envs,
            self.cfg.env.single_history_obs_dim * self.cfg.env.include_history_step,
            device=self.device,
            dtype=torch.float,
        )
        self.student_obs_history_buf = ObservationBuffer3D(
            self.num_envs,
            self.cfg.env.single_history_obs_dim,
            self.cfg.env.include_history_step,
            self.device,
        )

        self.nav_obs_buf = torch.zeros(
            self.num_envs,
            self.cfg.navigation.num_nav_observations,
            device=self.device,
            dtype=torch.float,
        )
        self.nav_command_buffer = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.nav_last_actions = torch.zeros(
            self.num_envs, self.cfg.navigation.num_nav_actions, device=self.device, dtype=torch.float
        )
        self.recovery_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.obstacle_positions = torch.zeros(
            self.num_envs, self.cfg.navigation.num_obstacles, 2, device=self.device, dtype=torch.float
        )
        self.obstacle_radii = torch.zeros(
            self.num_envs, self.cfg.navigation.num_obstacles, device=self.device, dtype=torch.float
        )

        self.path_points_local = self._build_desired_path()
        self.goal_local = self.path_points_local[-1].clone()

        self.nav_local_pos = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.nav_heading = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_cross_track_error = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_heading_error = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_goal_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_lookahead_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_prev_cross_track_error = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_prev_lookahead_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_min_clearance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_front_clearance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_path_follow_blend = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_speed_scale = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.nav_collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nav_reached_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nav_out_of_bounds_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.term_body_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.term_timeout_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.term_trap_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.term_collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.term_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.term_out_of_bounds_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _build_desired_path(self):
        path_cfg = self.cfg.navigation
        x = torch.arange(
            0.0,
            path_cfg.field_length + path_cfg.path_resolution,
            path_cfg.path_resolution,
            device=self.device,
            dtype=torch.float,
        )

        if path_cfg.path_type == "sine":
            y = path_cfg.path_center_y + path_cfg.path_amplitude * torch.sin(
                2.0 * math.pi * x / path_cfg.path_wavelength
            )
        else:
            y = torch.full_like(x, path_cfg.path_center_y)

        return torch.stack([x, y], dim=1)

    def _path_y_at_x(self, x_value):
        path_cfg = self.cfg.navigation
        x_value = float(np.clip(x_value, 0.0, path_cfg.field_length))
        if path_cfg.path_type == "sine":
            return path_cfg.path_center_y + path_cfg.path_amplitude * math.sin(
                2.0 * math.pi * x_value / path_cfg.path_wavelength
            )
        return path_cfg.path_center_y

    def _path_heading_at_x(self, x_tensor):
        path_cfg = self.cfg.navigation
        if path_cfg.path_type == "sine":
            slope = (
                path_cfg.path_amplitude
                * (2.0 * math.pi / path_cfg.path_wavelength)
                * torch.cos(2.0 * math.pi * x_tensor / path_cfg.path_wavelength)
            )
            return torch.atan(slope)
        return torch.zeros_like(x_tensor)

    def _get_robot_local_pos(self):
        local_pos = self.root_states[:, :2] - self.env_origins[:, :2]
        local_pos[:, 1] += self.cfg.navigation.field_width * 0.5
        return local_pos

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return

        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        local_x = torch_rand_float(0.0, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        local_y = torch_rand_float(
            self.cfg.navigation.path_center_y - 0.25,
            self.cfg.navigation.path_center_y + 0.25,
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.root_states[env_ids, 0] = self.env_origins[env_ids, 0] + local_x
        self.root_states[env_ids, 1] = (
            self.env_origins[env_ids, 1] + local_y - self.cfg.navigation.field_width * 0.5
        )

        heading = self._path_heading_at_x(local_x) + torch_rand_float(
            -0.15, 0.15, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        zero = torch.zeros(len(env_ids), device=self.device)
        q1 = quat_from_euler_xyz(zero, zero, heading)
        q0 = self.base_init_state[3:7].unsqueeze(0).repeat(len(env_ids), 1)
        q2 = quat_mul(q0, q1)
        self.root_states[env_ids, 3:7] = q2

        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, :3] = self.nav_command_buffer[env_ids]
        if self.commands.shape[1] > 3:
            self.commands[env_ids, 3] = 0.0

    def _sample_obstacles_for_env(self):
        nav_cfg = self.cfg.navigation
        obstacles = []
        start_xy = np.array([0.0, nav_cfg.path_center_y])
        goal_xy = self.goal_local.detach().cpu().numpy()

        for obstacle_idx in range(nav_cfg.num_obstacles):
            path_biased = np.random.rand() < nav_cfg.obstacle_path_bias
            for _ in range(200):
                radius = np.random.uniform(*nav_cfg.obstacle_radius_range)
                x_pos = np.random.uniform(*nav_cfg.obstacle_x_range)

                if path_biased:
                    y_pos = self._path_y_at_x(x_pos) + np.random.uniform(
                        -nav_cfg.obstacle_path_offset, nav_cfg.obstacle_path_offset
                    )
                else:
                    y_pos = np.random.uniform(
                        nav_cfg.obstacle_margin + radius,
                        nav_cfg.field_width - nav_cfg.obstacle_margin - radius,
                    )

                candidate = np.array([x_pos, y_pos], dtype=np.float32)

                if np.linalg.norm(candidate - start_xy) < 2.0 + radius:
                    continue
                if np.linalg.norm(candidate - goal_xy) < 1.8 + radius:
                    continue
                if y_pos < nav_cfg.obstacle_margin + radius:
                    continue
                if y_pos > nav_cfg.field_width - nav_cfg.obstacle_margin - radius:
                    continue

                valid = True
                for existing_pos, existing_radius in obstacles:
                    if np.linalg.norm(candidate - existing_pos) < radius + existing_radius + nav_cfg.obstacle_min_spacing:
                        valid = False
                        break
                if valid:
                    obstacles.append((candidate, radius))
                    break

        while len(obstacles) < nav_cfg.num_obstacles:
            obstacles.append((np.array([-100.0, -100.0], dtype=np.float32), 0.0))

        positions = np.stack([item[0] for item in obstacles], axis=0)
        radii = np.array([item[1] for item in obstacles], dtype=np.float32)
        return positions, radii

    def _reset_navigation_task(self, env_ids):
        if len(env_ids) == 0:
            return

        env_ids_cpu = env_ids.detach().cpu().tolist()
        for env_id in env_ids_cpu:
            positions, radii = self._sample_obstacles_for_env()
            self.obstacle_positions[env_id] = torch.as_tensor(positions, device=self.device, dtype=torch.float)
            self.obstacle_radii[env_id] = torch.as_tensor(radii, device=self.device, dtype=torch.float)

        self.nav_command_buffer[env_ids] = 0.0
        self.nav_last_actions[env_ids] = 0.0
        self.nav_front_clearance[env_ids] = self.cfg.navigation.safety_slow_clearance + 1.0
        self.nav_path_follow_blend[env_ids] = 1.0
        self.nav_speed_scale[env_ids] = 1.0
        self.commands[env_ids, :3] = 0.0
        self.recovery_mode[env_ids] = False
        self._update_navigation_state()
        self.nav_prev_cross_track_error[env_ids] = self.nav_cross_track_error[env_ids]
        self.nav_prev_lookahead_distance[env_ids] = self.nav_lookahead_distance[env_ids]

    def _update_navigation_state(self):
        nav_cfg = self.cfg.navigation
        clip_range = self.cfg.normalization.clip_observations

        local_pos = self._get_robot_local_pos()
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1)

        deltas = self.path_points_local.unsqueeze(0) - local_pos.unsqueeze(1)
        dist_sq = torch.sum(deltas * deltas, dim=-1)
        closest_idx = torch.argmin(dist_sq, dim=1)
        lookahead_idx = torch.clamp(closest_idx + nav_cfg.lookahead_steps, max=self.path_points_local.shape[0] - 1)
        prev_idx = torch.clamp(closest_idx - 1, min=0)
        next_idx = torch.clamp(closest_idx + 1, max=self.path_points_local.shape[0] - 1)

        tangent = self.path_points_local[next_idx] - self.path_points_local[prev_idx]
        tangent_norm = torch.norm(tangent, dim=1, keepdim=True).clamp_min(1e-6)
        tangent_unit = tangent / tangent_norm

        closest_point = self.path_points_local[closest_idx]
        lookahead_point = self.path_points_local[lookahead_idx]
        offset = local_pos - closest_point
        cross_track = tangent_unit[:, 0] * offset[:, 1] - tangent_unit[:, 1] * offset[:, 0]
        path_heading = torch.atan2(tangent_unit[:, 1], tangent_unit[:, 0])
        heading_error = wrap_to_pi(path_heading - heading)
        goal_distance = torch.norm(local_pos - self.goal_local.unsqueeze(0), dim=1)
        goal_forward_distance = torch.clamp(self.goal_local[0] - local_pos[:, 0], min=0.0)
        lookahead_distance = torch.norm(local_pos - lookahead_point, dim=1)

        obstacle_delta = self.obstacle_positions - local_pos.unsqueeze(1)
        obstacle_distance = torch.norm(obstacle_delta, dim=2)
        clearance = obstacle_distance - self.obstacle_radii - nav_cfg.robot_radius
        order = torch.argsort(clearance, dim=1)

        nearest_count = min(nav_cfg.num_nearest_obstacles, nav_cfg.num_obstacles)
        gather_idx = order[:, :nearest_count]
        nearest_delta = torch.gather(
            obstacle_delta, 1, gather_idx.unsqueeze(-1).expand(-1, -1, obstacle_delta.shape[-1])
        )
        nearest_clearance = torch.gather(clearance, 1, gather_idx)
        nearest_radii = torch.gather(self.obstacle_radii, 1, gather_idx)

        cos_h = torch.cos(heading).unsqueeze(1)
        sin_h = torch.sin(heading).unsqueeze(1)
        body_x = (cos_h * nearest_delta[:, :, 0] + sin_h * nearest_delta[:, :, 1]) / nav_cfg.observation_radius
        body_y = (-sin_h * nearest_delta[:, :, 0] + cos_h * nearest_delta[:, :, 1]) / nav_cfg.observation_radius
        clearance_norm = nearest_clearance / nav_cfg.observation_radius
        radius_norm = nearest_radii / 5.0

        obstacle_features = torch.stack(
            (
                torch.clamp(body_x, -1.0, 1.0),
                torch.clamp(body_y, -1.0, 1.0),
                torch.clamp(clearance_norm, -1.0, 1.0),
                torch.clamp(radius_norm, 0.0, 1.0),
            ),
            dim=2,
        ).flatten(1)

        nav_obs_parts = [
            torch.clamp(local_pos[:, 0] / nav_cfg.field_length, 0.0, 1.2).unsqueeze(1),
            torch.clamp(local_pos[:, 1] / nav_cfg.field_width, -0.2, 1.2).unsqueeze(1),
            torch.clamp(speed / nav_cfg.speed_observation_scale, 0.0, 2.0).unsqueeze(1),
            torch.sin(heading).unsqueeze(1),
            torch.cos(heading).unsqueeze(1),
            torch.clamp(cross_track / 20.0, -1.0, 1.0).unsqueeze(1),
            torch.clamp(heading_error / math.pi, -1.0, 1.0).unsqueeze(1),
            torch.clamp(goal_forward_distance / nav_cfg.field_length, 0.0, 1.5).unsqueeze(1),
            obstacle_features,
        ]

        nav_obs = torch.cat(nav_obs_parts, dim=1)
        self.nav_obs_buf[:] = torch.clamp(nav_obs, -clip_range, clip_range)

        min_clearance = torch.min(clearance, dim=1).values
        self.nav_local_pos[:] = local_pos
        self.nav_heading[:] = heading
        self.nav_cross_track_error[:] = cross_track
        self.nav_heading_error[:] = heading_error
        self.nav_goal_distance[:] = goal_distance
        self.nav_lookahead_distance[:] = lookahead_distance
        self.nav_min_clearance[:] = min_clearance
        self.nav_collision_buf[:] = min_clearance < 0.0
        self.nav_reached_goal_buf[:] = (goal_distance < nav_cfg.goal_tolerance) | (
            local_pos[:, 0] >= self.goal_local[0]
        )
        self.nav_out_of_bounds_buf[:] = (
            (local_pos[:, 0] < -0.5)
            | (local_pos[:, 0] > nav_cfg.field_length + 0.5)
            | (local_pos[:, 1] < nav_cfg.out_of_bounds_margin)
            | (local_pos[:, 1] > nav_cfg.field_width - nav_cfg.out_of_bounds_margin)
        )

    def compute_observations(self):
        add_noise_cfg = self.cfg.noise.add_noise
        obs_real_time = torch.cat(
            (
                observations.obs_dof_pos_normalized(self, add_noise_cfg)
                if self.cfg.control.use_dof_limit_normalize
                else observations.obs_dof_pos(self, add_noise_cfg),
                observations.obs_dof_vel(self, add_noise_cfg),
                observations.obs_projected_gravity(self, add_noise_cfg),
                observations.obs_base_ang_vel(self, add_noise_cfg),
            ),
            dim=-1,
        )

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.cfg.domain_rand.randomize_control_latency:
            self.sensor_obs_delayed_buf.insert(obs_real_time)
            self.sensor_obs_delayed_buf.reset(reset_env_ids, obs_real_time[reset_env_ids])
            n_steps_ago = torch.floor(self.control_latency / self.dt).int()
            obs_delayed = self.sensor_obs_delayed_buf.get_obs_by_t(n_steps_ago)
            student_obs_current = torch.cat(
                (
                    obs_delayed,
                    observations.obs_last_actions(self),
                    observations.obs_commands(self),
                ),
                dim=-1,
            )
        else:
            student_obs_current = torch.cat(
                (
                    obs_real_time,
                    observations.obs_last_actions(self),
                    observations.obs_commands(self),
                ),
                dim=-1,
            )

        self.student_obs_history_buf.reset(reset_env_ids, student_obs_current[reset_env_ids])
        self.student_obs_history_buf.insert(student_obs_current)

        student_obs_buf = self.student_obs_history_buf.get_latest_obs(self.cfg.env.include_history_step)
        student_obs_buf = student_obs_buf.flatten(1, 2)
        self.student_obs_buf[:] = torch.clamp(
            student_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )

        self._update_navigation_state()
        self.obs_buf[:] = self.nav_obs_buf

    def check_termination(self):
        self._update_navigation_state()
        nav_cfg = self.cfg.navigation

        if self.termination_contact_indices.numel() > 0:
            body_contact = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
                dim=1,
            )
        else:
            body_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        long_time_trap = self.trap_static_time > self.cfg.commands.trap_time
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        self.term_body_contact_buf[:] = body_contact
        self.term_timeout_buf[:] = self.time_out_buf
        self.term_trap_buf[:] = long_time_trap
        self.term_collision_buf[:] = self.nav_collision_buf
        self.term_goal_buf[:] = self.nav_reached_goal_buf
        self.term_out_of_bounds_buf[:] = self.nav_out_of_bounds_buf

        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if nav_cfg.terminate_on_body_contact:
            self.reset_buf |= body_contact
        if nav_cfg.terminate_on_timeout:
            self.reset_buf |= self.time_out_buf
        if nav_cfg.terminate_on_trap:
            self.reset_buf |= long_time_trap
        if nav_cfg.terminate_on_collision:
            self.reset_buf |= self.nav_collision_buf
        if nav_cfg.terminate_on_goal:
            self.reset_buf |= self.nav_reached_goal_buf
        if nav_cfg.terminate_on_out_of_bounds:
            self.reset_buf |= self.nav_out_of_bounds_buf

    def compute_reward(self):
        nav_cfg = self.cfg.navigation
        progress_reward = nav_cfg.reward_progress * (
            self.nav_prev_lookahead_distance - self.nav_lookahead_distance
        )
        track_recovery_reward = nav_cfg.reward_track_recovery * (
            self.nav_prev_cross_track_error.abs() - self.nav_cross_track_error.abs()
        )
        track_penalty = -nav_cfg.penalty_track * self.nav_cross_track_error.abs()
        heading_reward = nav_cfg.reward_heading * torch.cos(self.nav_heading_error)

        clearance_reward = torch.where(
            self.nav_min_clearance < nav_cfg.safe_margin,
            -1.5 * torch.square(nav_cfg.safe_margin - self.nav_min_clearance),
            nav_cfg.reward_clearance * torch.clamp(self.nav_min_clearance - nav_cfg.safe_margin, max=4.0),
        )

        reward = (
            progress_reward
            + track_recovery_reward
            + track_penalty
            + heading_reward
            + clearance_reward
            - nav_cfg.step_penalty
        )

        reward = reward - self.nav_collision_buf.float() * nav_cfg.penalty_collision
        reward = reward - self.nav_out_of_bounds_buf.float() * nav_cfg.penalty_out_of_bounds
        reward = reward + self.nav_reached_goal_buf.float() * nav_cfg.reward_goal

        self.rew_buf[:] = reward
        self.episode_sums["nav_total"] += reward
        self.episode_sums["nav_progress"] += progress_reward
        self.episode_sums["nav_track_recovery"] += track_recovery_reward
        self.episode_sums["nav_track_penalty"] += track_penalty
        self.episode_sums["nav_heading"] += heading_reward
        self.episode_sums["nav_clearance"] += clearance_reward
        self.episode_sums["nav_goal"] += self.nav_reached_goal_buf.float() * nav_cfg.reward_goal
        self.episode_sums["nav_collision"] += -self.nav_collision_buf.float() * nav_cfg.penalty_collision
        self.episode_sums["nav_out_of_bounds"] += (
            -self.nav_out_of_bounds_buf.float() * nav_cfg.penalty_out_of_bounds
        )

        self.nav_prev_cross_track_error[:] = self.nav_cross_track_error
        self.nav_prev_lookahead_distance[:] = self.nav_lookahead_distance

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_navigation_task(env_ids)

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        self.compute_observations()
        return self.get_nav_observations(), self.get_student_obs()

    def _get_body_obstacle_geometry(self):
        obstacle_delta = self.obstacle_positions - self.nav_local_pos.unsqueeze(1)
        obstacle_distance = torch.norm(obstacle_delta, dim=2)
        clearance = obstacle_distance - self.obstacle_radii - self.cfg.navigation.robot_radius

        cos_h = torch.cos(self.nav_heading).unsqueeze(1)
        sin_h = torch.sin(self.nav_heading).unsqueeze(1)
        body_x = cos_h * obstacle_delta[:, :, 0] + sin_h * obstacle_delta[:, :, 1]
        body_y = -sin_h * obstacle_delta[:, :, 0] + cos_h * obstacle_delta[:, :, 1]
        return body_x, body_y, clearance

    def _compute_front_obstacle_clearance(self):
        nav_cfg = self.cfg.navigation
        body_x, body_y, clearance = self._get_body_obstacle_geometry()
        lateral_limit = self.obstacle_radii + nav_cfg.robot_radius + nav_cfg.safety_lateral_margin
        front_mask = (
            (body_x > 0.0)
            & (body_x < nav_cfg.safety_front_distance)
            & (torch.abs(body_y) < lateral_limit)
        )

        shield_clearance = clearance - nav_cfg.safety_radius_buffer
        far_clearance_value = max(nav_cfg.safety_slow_clearance + 1.0, nav_cfg.path_follow_full_clearance)
        far_clearance = torch.full_like(shield_clearance, far_clearance_value)
        front_clearance = torch.where(front_mask, shield_clearance, far_clearance).min(dim=1).values
        return front_clearance

    def _blend_path_following_yaw(self, policy_yaw, front_clearance):
        nav_cfg = self.cfg.navigation
        if not nav_cfg.path_following_enabled:
            return policy_yaw, torch.zeros_like(policy_yaw)

        clearance_range = max(nav_cfg.path_follow_full_clearance - nav_cfg.path_follow_policy_clearance, 1e-6)
        path_blend = torch.clamp(
            (front_clearance - nav_cfg.path_follow_policy_clearance) / clearance_range,
            0.0,
            1.0,
        )
        path_yaw = (
            nav_cfg.path_follow_heading_gain * self.nav_heading_error
            - nav_cfg.path_follow_track_gain * self.nav_cross_track_error
        )
        path_yaw = torch.clamp(path_yaw, -nav_cfg.max_yaw_command, nav_cfg.max_yaw_command)
        target_yaw = (1.0 - path_blend) * policy_yaw + path_blend * path_yaw
        return target_yaw, path_blend

    def _compute_forward_speed_scale(self, target_yaw, front_clearance):
        nav_cfg = self.cfg.navigation
        if nav_cfg.safety_shield_enabled:
            clearance_range = max(nav_cfg.safety_slow_clearance - nav_cfg.safety_stop_clearance, 1e-6)
            obstacle_speed_scale = torch.clamp(
                (front_clearance - nav_cfg.safety_stop_clearance) / clearance_range,
                0.0,
                1.0,
            )
            crawl_floor = torch.where(
                front_clearance > nav_cfg.safety_stop_clearance,
                torch.full_like(obstacle_speed_scale, nav_cfg.safety_min_speed_scale),
                torch.zeros_like(obstacle_speed_scale),
            )
            obstacle_speed_scale = torch.maximum(obstacle_speed_scale, crawl_floor)
        else:
            obstacle_speed_scale = torch.ones_like(target_yaw)

        yaw_fraction = torch.clamp(torch.abs(target_yaw) / max(nav_cfg.max_yaw_command, 1e-6), 0.0, 1.0)
        turn_speed_scale = torch.clamp(
            1.0 - nav_cfg.turn_speed_reduction * yaw_fraction,
            min=nav_cfg.min_turn_speed_scale,
            max=1.0,
        )
        return torch.minimum(obstacle_speed_scale, turn_speed_scale)

    def apply_navigation_actions(self, nav_actions):
        nav_actions = torch.clamp(nav_actions, -1.0, 1.0).to(self.device)
        nav_cfg = self.cfg.navigation
        thrust_norm = (nav_actions[:, 1] + 1.0) * 0.5
        target_commands = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        target_commands[:, 0] = nav_cfg.min_forward_command + thrust_norm * (
            nav_cfg.max_forward_command - nav_cfg.min_forward_command
        )
        target_commands[:, 0] = torch.where(
            thrust_norm < 0.05, torch.zeros_like(target_commands[:, 0]), target_commands[:, 0]
        )
        policy_yaw = nav_actions[:, 0] * nav_cfg.max_yaw_command
        front_clearance = self._compute_front_obstacle_clearance()
        target_yaw, path_blend = self._blend_path_following_yaw(policy_yaw, front_clearance)
        speed_scale = self._compute_forward_speed_scale(target_yaw, front_clearance)

        target_commands[:, 0] *= speed_scale
        cruise_allowed = (front_clearance > nav_cfg.cruise_clearance) & (~self.nav_reached_goal_buf)
        cruise_command = torch.full_like(target_commands[:, 0], nav_cfg.cruise_forward_command)
        target_commands[:, 0] = torch.where(
            cruise_allowed,
            torch.maximum(target_commands[:, 0], cruise_command),
            target_commands[:, 0],
        )
        target_commands[:, 2] = target_yaw

        linear_smoothing = getattr(nav_cfg, "linear_command_smoothing", nav_cfg.command_smoothing)
        yaw_smoothing = getattr(nav_cfg, "yaw_command_smoothing", nav_cfg.command_smoothing)
        self.nav_command_buffer[:, :2] = (
            linear_smoothing * self.nav_command_buffer[:, :2]
            + (1.0 - linear_smoothing) * target_commands[:, :2]
        )
        self.nav_command_buffer[:, 2] = (
            yaw_smoothing * self.nav_command_buffer[:, 2]
            + (1.0 - yaw_smoothing) * target_commands[:, 2]
        )
        self.nav_command_buffer[self.nav_reached_goal_buf] = 0.0
        self.nav_last_actions[:] = nav_actions
        self.nav_front_clearance[:] = front_clearance
        self.nav_path_follow_blend[:] = path_blend
        self.nav_speed_scale[:] = speed_scale
        self.commands[:, :3] = self.nav_command_buffer

    def get_student_obs(self):
        return self.student_obs_buf

    def get_nav_observations(self):
        return self.nav_obs_buf

    def get_low_level_masks(self):
        gravity_z = self.projected_gravity[:, 2]
        ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)

        upright = gravity_z < self.cfg.navigation.recovery_release_gravity_z
        tipped = gravity_z > self.cfg.navigation.recovery_trigger_gravity_z
        settled = ang_vel_norm < self.cfg.navigation.recovery_settle_ang_vel

        self.recovery_mode[upright] = False
        self.recovery_mode[tipped & settled] = True
        protective_mask = tipped & (~settled) & (~self.recovery_mode)
        return protective_mask, self.recovery_mode.clone()

    def get_protective_actions(self):
        pose = torch.tensor(self.cfg.navigation.protective_pose, device=self.device, dtype=torch.float)
        return pose.repeat(self.num_envs, 6)

    def get_navigation_status(self, env_id=0):
        return {
            "local_x": float(self.nav_local_pos[env_id, 0].item()),
            "local_y": float(self.nav_local_pos[env_id, 1].item()),
            "goal_distance": float(self.nav_goal_distance[env_id].item()),
            "track_error": float(self.nav_cross_track_error[env_id].item()),
            "min_clearance": float(self.nav_min_clearance[env_id].item()),
            "front_clearance": float(self.nav_front_clearance[env_id].item()),
            "path_blend": float(self.nav_path_follow_blend[env_id].item()),
            "speed_scale": float(self.nav_speed_scale[env_id].item()),
            "collision": bool(self.nav_collision_buf[env_id].item()),
            "goal_reached": bool(self.nav_reached_goal_buf[env_id].item()),
            "out_of_bounds": bool(self.nav_out_of_bounds_buf[env_id].item()),
        }

    def get_termination_status(self, env_id=0):
        return {
            "body_contact": bool(self.term_body_contact_buf[env_id].item()),
            "timeout": bool(self.term_timeout_buf[env_id].item()),
            "trap": bool(self.term_trap_buf[env_id].item()),
            "collision": bool(self.term_collision_buf[env_id].item()),
            "goal_reached": bool(self.term_goal_buf[env_id].item()),
            "out_of_bounds": bool(self.term_out_of_bounds_buf[env_id].item()),
        }

    def draw_debug_vis(self):
        super().draw_debug_vis()

        if not getattr(self.cfg.debug, "plot_navigation", False):
            return

        env_id = 0
        origin = self.env_origins[env_id].cpu().numpy()
        shift_y = self.cfg.navigation.field_width * 0.5

        goal_geometry = gymutil.WireframeSphereGeometry(
            self.cfg.navigation.goal_tolerance, 8, 8, None, color=(0, 1, 0)
        )
        goal_world = gymapi.Transform(
            gymapi.Vec3(
                origin[0] + float(self.goal_local[0].item()),
                origin[1] + float(self.goal_local[1].item()) - shift_y,
                origin[2] + 0.2,
            ),
            r=None,
        )
        gymutil.draw_lines(goal_geometry, self.gym, self.viewer, self.envs[env_id], goal_world)

        path_geometry = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0.7, 1))
        for point in self.path_points_local[::4]:
            point_world = gymapi.Transform(
                gymapi.Vec3(
                    origin[0] + float(point[0].item()),
                    origin[1] + float(point[1].item()) - shift_y,
                    origin[2] + 0.05,
                ),
                r=None,
            )
            gymutil.draw_lines(path_geometry, self.gym, self.viewer, self.envs[env_id], point_world)

        for obs_pos, obs_radius in zip(self.obstacle_positions[env_id], self.obstacle_radii[env_id]):
            if obs_radius <= 0.0:
                continue
            obstacle_geometry = gymutil.WireframeSphereGeometry(float(obs_radius.item()), 8, 8, None, color=(1, 0, 0))
            obstacle_world = gymapi.Transform(
                gymapi.Vec3(
                    origin[0] + float(obs_pos[0].item()),
                    origin[1] + float(obs_pos[1].item()) - shift_y,
                    origin[2] + 0.15,
                ),
                r=None,
            )
            gymutil.draw_lines(obstacle_geometry, self.gym, self.viewer, self.envs[env_id], obstacle_world)
