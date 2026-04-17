import glob
import math
import numpy as np
import os

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym import LEGGED_GYM_ROOT_DIR

MOTION_FILES = glob.glob(os.path.join(LEGGED_GYM_ROOT_DIR,'datasets/roll_robot_r/*'))

PROPRIOCEPTION_DIM = 18+18+18+3+3+3+3 # 本体感知 66
#18(q_pos)+18(q_vel)+18(action)
#3(base linear vel)
#3(base angular vel)
#3(projected gravity)
#3(command)
PRIVILEGED_DIM = 3+1+6+1+18+13+6 #超参数
TERRAIN_DIM = 9*9

class rollRobotRCfg(LeggedRobotCfg):
    class env:
        is_train=True
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        num_actions = 18 #6x3
        num_observations = PROPRIOCEPTION_DIM #神经网络输入向量的长度
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        episode_length_s = 20 # episode length in seconds
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm

        reference_state_initialization = False # initialize state from reference data
        amp_motion_files = MOTION_FILES # reference init用

    class debug:
        # draw_debug_vis
        plot_heights = False
        plot_normals = False

    class init_state:  
        pos = [0.0, 0.0, 0.3] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'hipdriver1': 0.,
            'hipdriver2': 0.,
            'hipdriver3': 0.,
            'hipdriver4': 0.,
            'hipdriver5': 0.,
            'hipdriver6': 0.,

            'thighdriver1': 0.,  
            'thighdriver2': 0.,  
            'thighdriver3': 0.,
            'thighdriver4': 0.,
            'thighdriver5': 0.,
            'thighdriver6': 0.,

            'shankdriver1': -0.2,
            'shankdriver2': -0.2,
            'shankdriver3': -0.2, 
            'shankdriver4': -0.2, 
            'shankdriver5': -0.2, 
            'shankdriver6': -0.2, 
        }
        default_joint_pos= [value for key, value in default_joint_angles.items()]
        stand_joint_pos = [0,0,-0.2]*6 # 只影响reward_dof_pos计算 可以和default不一样

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.1
            dof_pos = 1.0
            dof_vel = 0.1
            height_measurements = 1.0

        clip_observations = 10.
        clip_actions = 1.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        noise_curriculum = False
        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 1.0
            gravity = 0.05
            height_measurements = 0.1

    class terrain:
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        horizontal_scale = 0.05  # [m]   # 0.05
        vertical_scale = 0.005  # [m]   
        slope_treshold = 0.8  # slopes above this threshold will be corrected to vertical surfaces  

        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        terrain_names=["flat","smooth slope down", "smooth slope up", "rough slope down", "rough slope up"]

        # terrain_proportions = [1.0]
        # terrain_names=["flat"]

        init_terrain_at_max_level = False
        max_init_terrain_level = 9  # 随机等级的最大值
        init_terrain_all_level = 9  # 开始时的总体等级
        border_size = 10 

        measure_heights = True # 采样地形高度 计算地形法线
        add_height_observation = False # 高程图是否加入observation
        measured_points_x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4],dtype=float)/10.0
        measured_points_y = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4],dtype=float)/10.0

        curriculum = True
        init_difficulty = 0   # 初始化地形等级0～1 
        terrain_length = 8. #地形长宽(米)
        terrain_width = 8.   
        num_rows = 10  # number of terrain rows (levels) 
        num_cols = 20  # number of terrain cols (types) 

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain

        # 地形的摩擦参数
        static_friction = 0.
        dynamic_friction = 0.
        rolling_friction = 0.1
        torsion_friction = 0.1
        restitution = 0.0

        class level_property:
            # max difficulty时的数值
            platform_size = 2 # 地形中间空白区域
            slope_angle = math.tan(22/180*math.pi)  # 坡度tan(a)=0.4
            stairs_height_up = [0., 0.10] # 上楼梯
            stairs_height_down = [0., 0.17] # 下楼梯
            obstacles_height = [0., 0.1]  # h 最大落差2h(-h~h)
            gap_size = 1.0
            pit_depth = 1.0

    class commands:
        curriculum = False
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        trap_vel=0.1 #小于这个速度认为是卡住了
        trap_time=5 #卡住超过这个时间就reset

        # stand command
        generate_stand_cmd = False # 控制resample时后有没有stand指令
        cmd_stand_lin = 0.1 # 小于该速度认为应该stand
        cmd_stand_ang = 0.2
        stand_prob = 0.2 # stand指令的百分比

        class ranges:
            max_curriculum = 0.5 # lin_vel [m/s]
            max_ang_vel_yaw = 1.0 # [rad/s]


    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
            'hipdriver': 30,
            'thighdriver': 30,
            'shankdriver': 30,
        }  # [N*m/rad]
        damping = {
            'hipdriver': 1.5,
            'thighdriver': 1.5,
            'shankdriver': 1.5,
        }     # [N*m*s/rad]
        torque_limits = {
            'hipdriver': 12.,
            'thighdriver': 12.,
            'shankdriver': 12.,
        }     # [N*m]
        action_scale = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_dof_limit_normalize = True
    
    class domain_rand:
        push_robots = False #TODO: 这个是不是不用加
        max_push_vel_xyz = 1
        push_interval_s = 7.2

        randomize_kpkd_factor = True
        kp_factor_range = (0.9, 1.1)
        kd_factor_range = (0.9, 1.1)

        randomize_apply_force = False # 持续的力 #TODO: 这个是不是不用加
        force_range = [-10,10]
        torque_range = [-2,2]
        
        randomize_base_mass = True  # 载荷
        added_mass_range = [0., 10.]

        randomize_friction = True # 摩擦
        friction_range = [0.5, 3.0] # 和地面摩擦平均以后才是实际摩擦
        randomize_restitution = True
        restitution_range = [0, 0.2] # 碰撞后的回弹速度,材料的软硬
        
        randomize_control_latency = False
        control_latency_range = [0.00, 0.081]  # 取整原因0.08必须多一点

        randomize_dof_armature = True
        armature_range = [0.002, 0.02]

        randomize_limit = False # 注意limit有curriculum 每1000步增加
        # FIXME: pos vel limit未实现
        # smallest_limit_percent = [[-0.5,0.5],#hip
        #                           [-0.4,0.8],#thigh
        #                           [-0.8,0.8]]#shank
        # smallest_vel_percent = 0.3 # 15.71 -> 5.0

        smallest_torque_percent = [0.66,1.0] # [12,12] -> [8,12]
        # smallest_torque_percent = [0.85,1.0] # [12,12] -> [10,12]

        randomize_motor_offset = False
        motor_offset_range = [-0.02, 0.02]

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/roll_robot_r/urdf/roll_robot_r.urdf'
        actor_name = "roll_robot_r"
        foot_name = "foot"
        hip_name = "hip"
        thigh_name = "thigh"
        shank_name = "shank"
        base_name = "base_link"
        penalize_contacts_on = ["shank"] #计算reward_collision的范围(外部碰撞)
        penalize_self_collision = ["hip", "thigh"] #自碰撞
        terminate_after_contacts_on = ["base_link"] 
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False
        disable_gravity = False
        collapse_fixed_joints = True # foot have dont_collapse=true
        fix_base_link = False # fixe the base of the robot
        flip_visual_attachments = False 
        vhacd_enabled = False #启用凸分解
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        density = 0.001 # Default density parameter used for calculating mass and inertia tensor when no mass and inertia data are provided, in $kg/m^3$.

        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        # 直接在asset上加不对, urdf惯量是相对于质心坐标系 不是旋转轴
        # armature = 2e-3 #转子惯量 6e-5 * 6^2 = 2e-3
        armature = 0. # 在dof_prop中配置
        thickness = 0.01

    class rewards:
        soft_dof_pos_percent = 0.95 #软限位95%,硬限位仿真器会限制
        soft_dof_vel_percent = 0.33 #25 8
        soft_torque_percent = 0.5 #12Nm 6Nm

        base_height_target = 0.250  #215+25mm
        base_height_threshold = 0.22  #90%
        orientation_threshold = 90
        max_contact_force = 300. # forces above this value are penalized
        
        only_positive_rewards = False
        tracking_lin_vel_std = 0.125**2 # 0.125(0.5m/s), 0.25(1m/s)
        tracking_ang_vel_std = 0.25**2 # 0.125(0.5m/s), 0.25(1m/s)
        # duty 5-1=0.8333, 4-2=0.6666(slow), trot=0.5, 2-4=0.3333(fast), 1-5=0.1666
        gait_type_duty = 0.5

        reward_curriculum=True
        curriculum_reward_list = [
            "dof_vel_limits",
            "torque_limits",
            "dof_pos_limits",
            "power_consumption",
            "no_clip",
            "lin_vel_z",
        ]

        # 在stand时候 切换到这些reward
        stand_reward_list = [
            "stand_dof_pos",
            "stand_dof_vel",
            "stand_feet_on_ground",
            "tracking_lin_vel",
            "tracking_ang_vel",
        ]

        reward_global_coef = 0.1
        class scales():
            tracking_lin_vel = 5.0
            tracking_ang_vel = 5.0
            feet_air_time = 5.0
            gait_duty = 0

            lin_vel_z = -20.0
            ang_vel_xy = -0.05
            orientation = -1.0

            dof_position = -0.01
            torques = -1e-4
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.1
            # cot = -0.02
            # power_peak = -1.0
            power_consumption = -0.0005

            base_height = -0.5 
            collision = -2.0
            self_collision = -10.0
            feet_contact_forces_z = -0.004

            dof_pos_limits = -10.0
            dof_vel_limits = -1.75
            torque_limits = -0.65
            no_clip = -0.15 # 18+18=36

            stand_feet_on_ground = 2.0 # 站立时 六条腿触地

            # 没有指令时保持静止
            # stand_dof_pos =3.0 # exp
            # stand_dof_vel =0.0 
            # stand_dof_vel =3.0 

            stand_dof_pos =1.0 # square
            stand_dof_vel =0.15

            termination =-100.0 #意外摔倒的惩罚
            trap_static = -1 #卡住不动
