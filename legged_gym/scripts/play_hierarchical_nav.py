import os
from datetime import datetime

import cv2
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils.nav_policy_loader import load_navigation_policy
from legged_gym.utils.task_registry import get_args, task_registry


CREATE_VIDEO = True
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 50
VIDEO_FRAME_STRIDE = 1
VIDEO_TRACK_ENV = 0
VIDEO_CAMERA_OFFSET = np.array([-2.8, -1.6, 1.4], dtype=np.float32)
VIDEO_TARGET_OFFSET = np.array([0.4, 0.0, 0.35], dtype=np.float32)


def _load_jit_policy(policy_path, device, label):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"{label} policy not found: {policy_path}")

    policy = torch.jit.load(policy_path, map_location=device).to(device)
    policy.eval()
    return policy


def _compose_low_level_actions(env, locomotion_policy, recovery_policy):
    student_obs = env.get_student_obs().detach()
    locomotion_actions = locomotion_policy(student_obs)
    protective_mask, recovery_mask = env.get_low_level_masks()

    actions = locomotion_actions
    if protective_mask.any() or recovery_mask.any():
        actions = actions.clone()

    if protective_mask.any():
        protective_actions = env.get_protective_actions()
        actions[protective_mask] = protective_actions[protective_mask]

    if recovery_mask.any():
        recovery_actions = recovery_policy(student_obs)
        actions[recovery_mask] = recovery_actions[recovery_mask]

    return actions


def _create_video_recorder(env, experiment_name):
    video_env = VIDEO_TRACK_ENV
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = VIDEO_WIDTH
    camera_properties.height = VIDEO_HEIGHT
    camera_handle = env.gym.create_camera_sensor(env.envs[video_env], camera_properties)
    if camera_handle == -1:
        raise RuntimeError(
            "Failed to create camera sensor. Offscreen rendering needs a graphics device even in headless mode."
        )

    robot_pos = env.root_states[video_env, 0:3].detach().cpu().numpy()
    camera_pos = robot_pos + VIDEO_CAMERA_OFFSET
    target_pos = robot_pos + VIDEO_TARGET_OFFSET

    env.gym.set_camera_location(
        camera_handle,
        env.envs[video_env],
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
    )
    if env.device != "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)

    first_frame = env.gym.get_camera_image(env.sim, env.envs[video_env], camera_handle, gymapi.IMAGE_COLOR)
    if first_frame is None or len(first_frame) == 0:
        raise RuntimeError("Camera sensor returned an empty frame during recorder initialization.")
    first_frame = np.reshape(first_frame, (VIDEO_HEIGHT, VIDEO_WIDTH, 4))
    frame_size = (first_frame.shape[1], first_frame.shape[0])

    output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "video", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{datetime.now().strftime('%m%d_%H%M%S')}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, frame_size)
    return camera_handle, writer, output_path


def _write_video_frame(env, camera_handle):
    robot_pos = env.root_states[VIDEO_TRACK_ENV, 0:3].detach().cpu().numpy()
    camera_pos = robot_pos + VIDEO_CAMERA_OFFSET
    target_pos = robot_pos + VIDEO_TARGET_OFFSET

    env.gym.set_camera_location(
        camera_handle,
        env.envs[VIDEO_TRACK_ENV],
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
    )
    if env.device != "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    frame = env.gym.get_camera_image(env.sim, env.envs[VIDEO_TRACK_ENV], camera_handle, gymapi.IMAGE_COLOR)
    if frame is None or len(frame) == 0:
        raise RuntimeError("Camera sensor returned an empty frame while writing video.")
    frame = np.reshape(frame, (VIDEO_HEIGHT, VIDEO_WIDTH, 4))
    return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


def play(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.is_train = True
    env_cfg.env.enable_camera_sensors = CREATE_VIDEO
    env_cfg.env.episode_length_s = 60
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    video_writer = None
    video_path = None
    camera_handle = None
    if CREATE_VIDEO:
        camera_handle, video_writer, video_path = _create_video_recorder(
            env, "roll_robot_r_hierarchical_nav"
        )

    locomotion_policy = _load_jit_policy(env_cfg.navigation.locomotion_policy_path, env.device, "Locomotion")
    recovery_policy = _load_jit_policy(env_cfg.navigation.recovery_policy_path, env.device, "Recovery")
    nav_policy, _ = load_navigation_policy(env_cfg.navigation.nav_policy_path, env.device)

    nav_obs = env.get_nav_observations()
    max_steps = int(env.max_episode_length.item()) if hasattr(env.max_episode_length, "item") else int(env.max_episode_length)

    try:
        for step in range(max_steps):
            with torch.inference_mode():
                nav_actions = nav_policy.act(nav_obs)
                env.apply_navigation_actions(nav_actions)
                low_level_actions = _compose_low_level_actions(env, locomotion_policy, recovery_policy)
                nav_obs, _, nav_rewards, nav_dones, _, _ = env.step(low_level_actions)

            if (
                video_writer is not None
                and camera_handle is not None
                and step % VIDEO_FRAME_STRIDE == 0
            ):
                frame_bgr = _write_video_frame(env, camera_handle)
                video_writer.write(frame_bgr)

            if step % 50 == 0:
                status = env.get_navigation_status(0)
                command = env.nav_command_buffer[0]
                print(
                    f"step={step:04d} "
                    f"x={status['local_x']:.2f} "
                    f"y={status['local_y']:.2f} "
                    f"goal_dist={status['goal_distance']:.2f} "
                    f"track_err={status['track_error']:.2f} "
                    f"clearance={status['min_clearance']:.2f} "
                    f"cmd=({command[0].item():.2f}, {command[2].item():.2f})"
                )

            if bool(nav_dones[0].item()):
                status = env.get_navigation_status(0)
                print(
                    "episode reset | "
                    f"goal={status['goal_reached']} "
                    f"collision={status['collision']} "
                    f"out_of_bounds={status['out_of_bounds']} "
                    f"reward={nav_rewards[0].item():.2f}"
                )
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"video saved to: {video_path}")


if __name__ == "__main__":
    args = get_args()
    args.task = "roll_robot_r_hierarchical_nav"
    args.headless = True
    play(args)
