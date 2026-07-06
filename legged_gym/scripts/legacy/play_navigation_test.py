from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymtorch
from legged_gym.envs import *
from legged_gym.utils.task_registry import  get_args, task_registry
from legged_gym.utils.helpers import export_policy_as_jit
from legged_gym.utils.logger import Logger
from legged_gym.utils.camera_control import CameraController
from isaacgym import gymapi

import numpy as np
import torch
import threading
import time
import math
import cv2
import datetime
import matplotlib
import matplotlib.pyplot as plt
import copy

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.is_train = True
    env_cfg.env.episode_length_s = 100000
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    track_env = 0
    pos= np.array(env.root_states[track_env,0:3].cpu())
    
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    if CREATE_VIDEO:
        # 创建camera sensor
        video_env = 0 # 传感器创建后仿真中不能修改
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        camera_handle1 = env.gym.create_camera_sensor(env.envs[video_env], camera_properties)
        env.gym.render_all_camera_sensors(env.sim)
        first_image = env.gym.get_camera_image(env.sim, env.envs[video_env], camera_handle1, gymapi.IMAGE_COLOR)
        first_image1=np.reshape(first_image,(camera_properties.height,camera_properties.width,4))
        height, width, layer = first_image1.shape

        # 定义视频编码器和输出视频对象
        current_time=datetime.datetime.now()
        formatted_datetime = current_time.strftime('%m%d_%H%M%S')
        output_video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'video', train_cfg.runner.experiment_name)
        os.makedirs(output_video_dir,exist_ok=True)
        output_video_filename = '{}.mp4'.format(formatted_datetime)
        output_video_fullpath = os.path.join(output_video_dir, output_video_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码（H.264）
        video = cv2.VideoWriter(output_video_fullpath, fourcc, 50, (width, height))  # 50 FPS

    if TRACK_CAMERA and (not args.headless):
        camera_controller=CameraController()
        window_thread = threading.Thread(target=camera_controller.create_window)
        window_thread.daemon = True #守护线程，主线程退出时，自动关闭窗口
        window_thread.start()
    student_obs = env.get_student_obs()

    while True:
        # student_obs = env.get_student_obs()
        actions = torch.zeros((env.num_envs, 3),device=env.device)
        actions[:,0] = 0.5
        env.nav_step(actions.detach())
        if RECORD_FRAMES:
            #sim 0.005s x decimation=4 0.02s 50hz
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        if TRACK_CAMERA:
            dist,yaw,pitch,pattern = camera_controller.get_values()
            pos= np.array(env.root_states[track_env,0:3].cpu())
            camera_direction = np.array([math.cos(pitch)*math.cos(yaw),
                                        math.cos(pitch)*math.sin(yaw),
                                        math.sin(pitch)])
            env.set_camera(pos + camera_direction * dist, # camera pos
                        pos) #target pos

            if CREATE_VIDEO:
                camera_pos=pos + camera_direction * dist
                target=pos
                env.gym.set_camera_location(camera_handle1, env.envs[video_env], 
                                            gymapi.Vec3(camera_pos[0],camera_pos[1],camera_pos[2]),
                                            gymapi.Vec3(target[0],target[1],target[2]))
                env.gym.render_all_camera_sensors(env.sim)
                frame = env.gym.get_camera_image(env.sim, env.envs[video_env], camera_handle1, gymapi.IMAGE_COLOR)
                frame1=np.reshape(frame,(camera_properties.height,camera_properties.width,4))
                frame2=cv2.cvtColor(frame1, cv2.COLOR_RGBA2BGR)
                video.write(frame2)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    TRACK_CAMERA = True
    CREATE_VIDEO = False
    args = get_args()
    args.task = "roll_robot_r_history_imitate" # 在这里切换task, env_cfg, alg_cfg
    args.headless=False
    play(args)
