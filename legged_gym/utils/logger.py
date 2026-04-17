import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
from legged_gym import LEGGED_GYM_ROOT_DIR
import os.path as osp
import os
ERROR_FLAG = -1

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.steps = 0

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)
        self.steps += 1

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()
        self.num_episodes = 0
        self.plot_process = None
        self.steps = 0

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()
        # self._plot()

    def plot_states_and_save(self,exp_name):
        self.saving=True
        self.exp_name=exp_name
        self.plot_states()

    def _plot(self):
        # 定义分组配置
        plot_groups = [
            ([self.plot_base_vel_x, self.plot_base_vel_y, self.plot_base_vel_yaw], (1, 3)),
            ([self.plot_dof_pos, self.plot_dof_vel, self.plot_dof_torque, self.plot_torque_vel_curves], (2, 2)),
            ([self.plot_base_ang_vel, self.plot_base_vel_z, self.plot_contact_forces_z, 
            self.plot_gait_time, self.plot_base_net_force, self.plot_value], None)  # None表示自动布局
        ]

        # 为每组创建单独的Figure
        for group_idx, (plots, layout) in enumerate(plot_groups):
            num_plots = len(plots)
            
            # 确定布局
            if layout is None:
                # 自动计算布局
                cols = int(np.ceil(np.sqrt(num_plots)))
                rows = int(np.ceil(num_plots / cols))
            else:
                rows, cols = layout
                assert(rows*cols==num_plots)
            
            # 自动计算Figure大小
            base_width = 4.5  # 每个子图的基础宽度
            base_height = 3  # 每个子图的基础高度
            fig_width = base_width * cols
            fig_height = base_height * rows
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # 创建子图
            for i, plot_func in enumerate(plots):
                ax = fig.add_subplot(rows, cols, i+1)
                result = plot_func(ax)
                if result == ERROR_FLAG:  # 如果没有对应数据要画就隐藏
                    ax.axis('off')
            
            fig.tight_layout()
            
            if hasattr(self,"saving"):
                file_name=f"group_{group_idx}.svg"
                dir_name = osp.join(LEGGED_GYM_ROOT_DIR,"logs","figure",self.exp_name)
                os.makedirs(dir_name,exist_ok=True)
                fig.savefig(osp.join(dir_name,file_name))

        # 单独窗口的图表
        self.plot_gait_binary()
        self.plot_gating_weight()
        self.plot_symmetry_error()

        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()

    def plot_dof_pos(self, axe):
        if not (("dof_pos" in self.state_log)and
                ("dof_pos_target" in self.state_log)): return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, self.state_log["dof_pos"], label='measured')
        axe.plot(time, self.state_log["dof_pos_target"], label='target')
        axe.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        axe.legend()

    def plot_dof_vel(self, axe):
        if not ("dof_vel" in self.state_log): return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, self.state_log["dof_vel"], label='measured')
        if ("dof_vel_target" in self.state_log):
            axe.plot(time, self.state_log["dof_vel_target"], label='target')
        axe.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        axe.legend()

    def plot_base_vel_x(self, axe):
        if not ("base_vel_x" in self.state_log): return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, self.state_log["base_vel_x"], label='measured')
        if ("command_x" in self.state_log):
            axe.plot(time, self.state_log["command_x"], label='commanded')
        axe.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        axe.legend()

    def plot_base_vel_y(self, axe):
        if not ("base_vel_y" in self.state_log): return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, self.state_log["base_vel_y"], label='measured')
        if ("command_y" in self.state_log):
            axe.plot(time, self.state_log["command_y"], label='commanded')
        axe.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        axe.legend()

    def plot_base_vel_yaw(self, axe):
        if not ("base_vel_yaw" in self.state_log): return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, self.state_log["base_vel_yaw"], label='measured')
        if ("command_yaw" in self.state_log):
            axe.plot(time, self.state_log["command_yaw"], label='commanded')
        axe.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        axe.legend()
    
    def plot_base_ang_vel(self, axe):
        if not (("base_vel_roll" in self.state_log)or
                ("base_vel_pitch" in self.state_log)or
                ("base_vel_yaw" in self.state_log)): return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        if "base_vel_roll" in self.state_log:
            axe.plot(time, self.state_log["base_vel_roll"], label='roll')
        if "base_vel_pitch" in self.state_log:
            axe.plot(time, self.state_log["base_vel_pitch"], label='pitch')
        if "base_vel_yaw" in self.state_log:
            axe.plot(time, self.state_log["base_vel_yaw"], label='yaw')
        axe.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base angular velocity')
        axe.legend()

    def plot_base_vel_z(self, axe):
        if not "base_vel_z" in self.state_log: return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, self.state_log["base_vel_z"], label='measured')
        axe.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        axe.legend()

    def plot_contact_forces_z(self, axe):
        if not "contact_forces_z" in self.state_log: return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        forces = np.array(self.state_log["contact_forces_z"])
        for i in range(forces.shape[1]):
            axe.plot(time, forces[:, i], label=f'leg {i+1}') # leg 1~6
        axe.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        axe.legend()

    def plot_base_net_force(self, axe):
        if not "base_net_force" in self.state_log: return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        forces = np.array(self.state_log["base_net_force"])
        axe.plot(time, forces, label=f'base net force')
        axe.set(xlabel='time [s]', ylabel='Forces [N]', title='Contact forces of base')
        axe.legend()

    def plot_dof_torque(self, axe):
        if not "dof_torque" in self.state_log: return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, self.state_log["dof_torque"], label='measured')
        axe.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        axe.legend()
    
    def plot_torque_vel_curves(self, axe):
        if not (("dof_vel" in self.state_log) and ("dof_torque" in self.state_log)): return ERROR_FLAG
        axe.plot(self.state_log["dof_vel"], self.state_log["dof_torque"], 'x', label='measured')
        axe.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        axe.legend()

    def plot_value(self, axe):
        if not ("value_estimated" in self.state_log): return ERROR_FLAG
        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, self.state_log["value_estimated"], label='value')
        axe.set(xlabel='time [s]', ylabel='value estimated', title='value estimated')
        axe.legend()

    def plot_gait_binary(self):
        if not "contacts_filtered" in self.state_log: return

        fig, axe = plt.subplots(figsize=(10, 4))
        gait_all=np.stack(self.state_log["contacts_filtered"])
        time_length, leg_number = gait_all.shape
        fps=50  # 一帧0.02秒
        real_time = time_length/fps

        # 添加水平线 y=1, y=2,... y=5
        # for y in range(1, 6):  # 这将画y=1到y=5的线
        #     axe.axhline(y=y, color='gray', linestyle='-', linewidth=1.)

        for leg_idx in range(leg_number):
            gait = gait_all[:, leg_idx]
            
            # 找到连续触地的时间段
            touch_down_segments = []
            start = None
            for t in range(time_length):
                if gait[t] and start is None:  # 开始触地
                    start = t
                elif not gait[t] and start is not None:  # 结束触地
                    touch_down_segments.append([start, t - 1])
                    start = None
            if start is not None:  # 如果最后一个时间段是触地的
                touch_down_segments.append([start, time_length - 1])

            # 绘制矩形
            touch_down_segments1=np.array(touch_down_segments)/fps
            for i in range(touch_down_segments1.shape[0]):
                start=touch_down_segments1[i][0]
                end=touch_down_segments1[i][1]
                # x,y,w,h
                axe.add_patch(matplotlib.patches.Rectangle((start, leg_idx), end - start, 1, color='black'))
        
        axe.set(xlabel='Time(s)', ylabel='Leg Index', title='Torque')
        axe.set(xlim=[0,real_time],ylim=[0,leg_number])
        fig.show()

        if hasattr(self,"saving"):
            file_name=f"gait_binary.svg"
            dir_name = osp.join(LEGGED_GYM_ROOT_DIR,"logs","figure",self.exp_name)
            os.makedirs(dir_name,exist_ok=True)
            fig.savefig(osp.join(dir_name,file_name))

    def plot_gait_gray(self):
        if not "contacts_filtered" in self.state_log: return

        from scipy.ndimage import gaussian_filter1d
        gait_all=np.stack(self.state_log["contacts_filtered"])
        time_length, leg_number = gait_all.shape
        fps=50  # 一帧0.02秒
        real_time = time_length/fps
        gait_all_float = gait_all.transpose(0,1).float().cpu().numpy()

        # 对每条腿的信号进行高斯滤波
        sigma = 3  # 高斯滤波的标准差，控制平滑程度
        gait_data_filtered = np.zeros_like(gait_all_float)
        for leg_idx in range(leg_number):
            gait_data_filtered[leg_idx,:] = gaussian_filter1d(gait_all_float[leg_idx,:], sigma=sigma)

        # 绘制灰度图
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(gait_data_filtered, aspect='auto', cmap='gray', origin='lower',
                extent=[0, real_time, 0, leg_number])

        ax.set(xlabel='Time(s)', ylabel='Leg Index', title='Torque')
        ax.set(xlim=[0,real_time],ylim=[0,leg_number])
        fig.colorbar(label='Touchdown Intensity')
        fig.show()

    def plot_gait_time(self, axe):
        if not "last_air_time" in self.state_log: return ERROR_FLAG

        air_time_buffer1=np.array(self.state_log["last_air_time"])
        contact_time_buffer1=np.array(self.state_log["last_contact_time"])
        clipped_step_time_buffer1=np.array(self.state_log["clipped_step_time"])
        real_step_time_buffer1=np.array(self.state_log["real_step_time"])

        time = np.linspace(0, self.steps * self.dt, self.steps)
        axe.plot(time, air_time_buffer1, label="air_time")
        axe.plot(time, contact_time_buffer1, label="contact_time")
        # axe.plot(time, clipped_step_time_buffer1, label="clipped_step_time") # do not need
        axe.plot(time, real_step_time_buffer1, label="real_step_time")
        axe.set(xlabel='time [s]', ylabel='gait time [s]', title='gait time')
        axe.legend()

    def plot_gating_weight(self):
        if not "gating_weight" in self.state_log: return

        gating_weight=np.stack(self.state_log["gating_weight"])
        num_expert = gating_weight.shape[1]
        time = np.linspace(0, self.steps * self.dt, self.steps)
        if not hasattr(self,"expert_names"):
            self.expert_names=[f"expert:{i}" for i in range(num_expert)]

        fig, axe = plt.subplots(figsize=(10, 4))
        for i in range(num_expert):
            axe.plot(time, gating_weight[:,i], label=f"{self.expert_names[i]}")
        axe.set(xlabel='time [s]', ylabel='gating weight', title='gating weight')
        axe.legend()
        fig.show()

        if hasattr(self,"saving"):
            file_name=f"gating_weight.svg"
            dir_name = osp.join(LEGGED_GYM_ROOT_DIR,"logs","figure",self.exp_name)
            os.makedirs(dir_name,exist_ok=True)
            fig.savefig(osp.join(dir_name,file_name))

    def plot_symmetry_error(self):
        if not ("symmetry_error_rotation" in self.state_log): return ERROR_FLAG
        if not ("symmetry_error_rotation_mirror" in self.state_log): return ERROR_FLAG
        
        fig, axe = plt.subplots(figsize=(10, 4))

        time = np.linspace(0, self.steps * self.dt, self.steps)
        rotation = self.state_log["symmetry_error_rotation"]
        rotation_mirror = self.state_log["symmetry_error_rotation_mirror"]
        axe.plot(time, rotation, label='rotation')
        axe.plot(time, rotation_mirror, label='rotation+mirror')

        # Plot average
        avg_rotation = np.mean(rotation)
        avg_rotation_mirror = np.mean(rotation_mirror)
        axe.axhline(avg_rotation, color='C0', linestyle='--', 
                label=f'rotation avg: {avg_rotation:.5f}')
        axe.axhline(avg_rotation_mirror, color='C1', linestyle='--', 
                label=f'rotation+mirror avg: {avg_rotation_mirror:.5f}')
        
        # Add text annotations for averages
        axe.text(time[-1], avg_rotation, f'{avg_rotation:.5f}', 
                color='C0', ha='left', va='center')
        axe.text(time[-1], avg_rotation_mirror, f'{avg_rotation_mirror:.5f}', 
                color='C1', ha='left', va='center')
        
        axe.set(xlabel='time [s]', ylabel='symmetry_error', title='symmetry_error')
        axe.legend()
        fig.show()

        if hasattr(self,"saving"):
            file_name=f"symmetry_error.svg"
            dir_name = osp.join(LEGGED_GYM_ROOT_DIR,"logs","figure",self.exp_name)
            os.makedirs(dir_name,exist_ok=True)
            fig.savefig(osp.join(dir_name,file_name))

if __name__=="__main__":
    state_log = {
        "dof_pos": np.random.rand(500),
        "dof_pos_target": np.random.rand(500),
        "dof_vel": np.random.rand(500),
        "dof_vel_target": np.random.rand(500),
        "base_vel_x": np.random.rand(500),
        "command_x": np.random.rand(500),
        "base_vel_y": np.random.rand(500),
        "command_y": np.random.rand(500),
        "base_vel_yaw": np.random.rand(500),
        "command_yaw": np.random.rand(500),
        "base_vel_z": np.random.rand(500),
        "contact_forces_z": np.random.rand(500, 6),
        "dof_torque": np.random.rand(500),
        "contacts_filtered":np.random.rand(500, 6),
        "last_air_time": np.random.rand(500),
        "last_contact_time": np.random.rand(500),
        "clipped_step_time": np.random.rand(500),
        "real_step_time": np.random.rand(500),
    }

    dt = 0.02
    plotter = Logger(dt)
    plotter.state_log = state_log
    plotter.steps = 500
    plotter._plot()