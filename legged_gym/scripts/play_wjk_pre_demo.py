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
from datetime import datetime#new
from rsl_rl.utils import check_nan #new

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
    # override some parameters for testing
    env_cfg.env.num_envs = 1  #TODO：这里要改一下，要改成上层网络训练的环境数？
    env_cfg.env.is_train = False
    env_cfg.env.episode_length_s = 100000 
    env_cfg.terrain.curriculum = False
    # env_cfg.num_rows = 5  # number of terrain rows (levels) 
    # env_cfg.num_cols = 4 
    env_cfg.commands.curriculum = False
    # env_cfg.asset.self_collisions = []
    env_cfg.asset.terminate_after_contacts_on = []
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner,_ = task_registry.make_alg_runner(env=env, name=args.task, args=args)#new
    ppo_runner.__init__(env,train_cfg)#new

    init_at_random_ep_len=True#new
    num_learning_iterations=train_cfg.runner.max_iterations#new
    if init_at_random_ep_len:#new
            env.episode_length_buf = torch.randint_like(
                env.episode_length_buf, high=int(env.max_episode_length)
            )


    env.env_origins[:,0] = 4  #这里可能需要适当修改？
    env.env_origins[:,1] = 4
    env.env_origins[:,2] = 8
    env.reset_idx(torch.tensor([0], dtype=torch.int32, device='cuda:0'))#TODO：这里只重置了第一个环境，后续需要改成初始重置所有环境
    obs = env.get_observations()
    
    
    ppo_runner.alg.train_mode()#new

    ppo_runner.logger.init_logging_writer()#new


    locomotion_agent = torch.jit.load('logs/locomotion.pt').to(env.device)
    recovery_agent = torch.jit.load('logs/recovery.pt').to(env.device)

    track_env = 0
    pos= np.array(env.root_states[track_env,0:3].cpu())
    
    stop_state_log = 500 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
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

    power_average_2s=np.zeros(int(2/env.dt)) #2/0.02=100

    l_low=env.default_dof_pos-env.dof_pos_limits[:,0]#关节余量
    l_up=env.dof_pos_limits[:,1]-env.default_dof_pos

    pattern = 0
    student_obs = env.get_student_obs()#这里的student_obs和上面的obs的区别是？作用分别是？
    nav_obs = env.get_nav_observations()#new,TODO:get_nav_observation，nav_obs应该怎么设置（相对目标的位置，角速度和线速度，降维后的2D激光），如何模拟相机获得的点云？

    start_recovery = False

    start_it = ppo_runner.current_learning_iteration#new
    total_it = start_it + num_learning_iterations#new
    for it in range(start_it,total_it):#new
        start = time.time()#new

        with torch.inference_mode():#new
            for _ in range(self.cfg["num_steps_per_env"]):#new

                nav_actions = ppo_runner.alg.act(nav_obs)#new
                #在这里添加：获取上层obs,上层运行action()函数获取三维动作指令（因为下层obs需要三维动作指令）
                student_obs = env.get_student_obs()#new
                # if env.projected_gravity[0,2] > -0.8 and torch.norm(env.ang_vel[0,:],dim=1) >0.2:
                if env.projected_gravity[0,2] < -0.95:
                    start_recovery = False
                    print("pos_good")
                print(start_recovery)
                if (env.projected_gravity[0,2] > -0.8 and torch.norm(env.base_ang_vel[0,:]) >0.5) and not start_recovery:
                    hip_ball = 0
                    calf_ball = -0.2
                    shank_ball = -0.8
                    leg_action = torch.tensor([[hip_ball,calf_ball,shank_ball]],device='cuda:0').detach()
                    actions = leg_action.repeat(1, 6)
                    print(0)
                elif (env.projected_gravity[0,2] > -0.8 and torch.norm(env.base_ang_vel[0,:]) < 0.5) or start_recovery:
                    start_recovery = True
                    actions = recovery_agent(student_obs.detach())
                    print(1)
                else:
                    actions = locomotion_agent(student_obs.detach())
                    print(2)
                #obs, _, rews, dones, infos, _ = env.step(actions.detach())
                obs, _, nav_rewards, dones, extras, _ = env.nav_step(nav_actions.detach())#new#TODO：,然后把它改成nav的rewards(修改compute_reward即可)，dones也需要修改，在原本的基础上添加到达终点后重置的逻辑
                nav_obs = env.get_nav_observations()#new#TODO:实现这个函数。或者把这个函数放进上面的nav_step，让他可以计算并多输出一个nav_obs。同时修改base_task的init，让他初始化一个nav_obs的history_buffer（有用吗？怎么感觉没地方用）
                if ppo_runner.cfg.get("check_for_nan", True):#new
                    check_nan(nav_obs, nav_rewards, dones)#new

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

                #更新下层网络的参数然后进行学习
                nav_obs,nav_rewards,dones = (nav_obs.to(ppo_runner.device), nav_rewards.to(ppo_runner.device), dones.to(ppo_runner.device))#new
                ppo_runner.alg.process_env_step(nav_obs, nav_rewards, dones, extras)#new
                intrinsic_rewards = ppo_runner.alg.intrinsic_rewards if ppo_runner.cfg["algorithm"]["rnd_cfg"] else None#new
                ppo_runner.logger.process_env_step(nav_rewards, dones, extras, intrinsic_rewards)#new
            
            stop = time.time()#new
            collect_time = stop - start#new
            start = stop#改到这里了#new

            ppo_runner.alg.compute_returns(nav_obs)#new

        loss_dict = ppo_runner.alg.update()#new

        stop = time.time()#new
        learn_time = stop - start#new
        ppo_runner.current_learning_iteration = it

        ppo_runner.logger.log(#new
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=ppo_runner.alg.learning_rate,
            action_std=ppo_runner.alg.get_policy().output_std,
            rnd_weight=ppo_runner.alg.rnd.weight if ppo_runner.cfg["algorithm"]["rnd_cfg"] else None,
        )

        # Save model#new
        if ppo_runner.logger.writer is not None and it % ppo_runner.cfg["save_interval"] == 0:#new
            ppo_runner.save(os.path.join(ppo_runner.logger.log_dir, f"model_{it}.pt"))  # type: ignore 
        
    # Save the final model after training and stop the logging writer
    if ppo_runner.logger.writer is not None:#new
        ppo_runner.save(os.path.join(ppo_runner.logger.log_dir, f"model_{ppo_runner.current_learning_iteration}.pt"))  # type: ignore
        ppo_runner.logger.stop_logging_writer()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    TRACK_CAMERA = False
    CREATE_VIDEO = False
    args = get_args()
    args.task = "roll_robot_r_history_imitate" # 在这里切换task, env_cfg, alg_cfg
    args.headless=False
    play(args)
