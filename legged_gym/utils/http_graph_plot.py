import requests
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from threading import Thread
import re

class graphPlotter:
    def __init__(self,
                 variable_to_plot,#要plot的变量名
                 point_num,#x范围
                 y_range,#y范围
                 ):
        self.all_point_num=point_num #如果要控制x轴显示的长度,修改这个值

        self.refresh_freq=25 #绘图器窗口刷新频率
        
        self.data_freq=50 #get请求频率(数据频率)
        self.data_interval=1/self.data_freq
        self.time_length=self.all_point_num*self.data_interval #x轴长度 #500x0.01=0.05
        self.y_range=y_range


        #command
        self.joint_position_command=list(np.zeros(18))
        #observation
        self.imu_rpy_angle=list(np.zeros(3))
        self.imu_rotate_velocity=list(np.zeros(3))
        self.robot_angular_velocity=list(np.zeros(3))
        self.robot_projected_gravity=list(np.zeros(3))
        self.encoder_joint_position=list(np.zeros(18))
        self.encoder_joint_velocity=list(np.zeros(18))
        self.urdf_joint_position=list(np.zeros(18))
        self.urdf_joint_velocity=list(np.zeros(18))
        self.net_actions=list(np.zeros(18))
        self.joystick_command=list(np.zeros(3))
        self.joint_torque=list(np.zeros(18))
        self.contact_forces_z=list(np.zeros(6))

        self.variable=variable_to_plot
        self.var_number=len(self.variable)

        #解析为变量名和变量序号
        self.variable_name = []
        self.variable_index = []
        for var in self.variable:
            match = re.match(r'(.+?)\[(\d+)\]', var)  # 使用正则表达式匹配变量名和索引
            if match:
                name, index = match.groups()
                self.variable_name.append(name)
                self.variable_index.append(int(index))

        self.init_graph()
        self.boot_time=time.time()
        self.update_thread=Thread(target=self.update_value,
                                  daemon=True)
        self.update_thread.start()

    def send_GET(self):
        url = 'http://localhost:3861'
        try:
            response = requests.get(url, params={'variable_name': self.variable_name})
        except Exception as e:
            if type(e) is requests.exceptions.ConnectionError:
                return
            else:
                raise e
        response_dict = response.json()
        
        for var_name, var_value in response_dict.items():
            if hasattr(self, var_name):
                setattr(self, var_name, var_value)
                # print("{} updated value:{}".format(var_name,var_value))

    def init_graph(self):
        self.fig, self.axe = plt.subplots(figsize=(10, 5))
        self.lines:list[Line2D] = [None for _ in range(self.var_number)]
        self.xdata = np.zeros(self.all_point_num)
        self.y_datas:list[np.ndarray]= [None for _ in range(self.var_number)]

        for i in range(self.var_number):
            self.y_datas[i]=np.zeros(self.all_point_num)
            line,=self.axe.plot(self.xdata, self.y_datas[i], 
                                label=self.variable_name[i]+"[{}]".format(self.variable_index[i]))
            self.lines[i]=line

        self.axe.set_xlim(0, self.time_length)  # 设置x轴范围
        self.axe.set_ylim(self.y_range[0], self.y_range[1])  # 设置y轴范围
        self.axe.legend()

    def update_value(self):
        #以100Hz(data freq)的频率发送GET请求
        while True:
            current_loop_start_time = time.time()
            get_thread=Thread(target=self.send_GET)
            get_thread.start()

            self.xdata = np.roll(self.xdata,-1)
            current_time=current_loop_start_time-self.boot_time
            self.xdata[-1] =  current_time
            for i in range(self.var_number):
                self.y_datas[i]=np.roll(self.y_datas[i], -1)
                self.y_datas[i][-1]=getattr(self, self.variable_name[i])[self.variable_index[i]]
                self.lines[i].set_data(self.xdata,self.y_datas[i])

            self.axe.set_xlim(current_time-self.time_length, current_time)  # 设置x轴范围

            #频率控制
            current_loop_end_time=time.time()
            actual_duration =current_loop_end_time-current_loop_start_time
            if actual_duration < self.data_interval:
                time.sleep(self.data_interval - actual_duration)
                freq=1/self.data_interval
            else:
                freq=1/actual_duration
            print("plotter频率:{:.2f}".format(freq))

def update(frame,g_plotter:graphPlotter):
    return g_plotter.axe

if __name__=="__main__":
    variable1=[
        'joint_torque[0]',
        'joint_torque[1]',
        'joint_torque[2]',
        'joint_torque[3]',
        'joint_torque[4]',
        'joint_torque[5]',
        'joint_torque[6]',
        'joint_torque[7]',
        'joint_torque[8]',
        'joint_torque[9]',
        'joint_torque[10]',
        'joint_torque[11]',
        'joint_torque[12]',
        'joint_torque[13]',
        'joint_torque[14]',
        'joint_torque[15]',
        'joint_torque[16]',
        'joint_torque[17]',
    ]
    variable2=[
        'encoder_joint_velocity[0]',
        'encoder_joint_velocity[1]',
        'encoder_joint_velocity[2]',
        'encoder_joint_velocity[3]',
        'encoder_joint_velocity[4]',
        'encoder_joint_velocity[5]',
        'encoder_joint_velocity[6]',
        'encoder_joint_velocity[7]',
        'encoder_joint_velocity[8]',
        'encoder_joint_velocity[9]',
        'encoder_joint_velocity[10]',
        'encoder_joint_velocity[11]',
        'encoder_joint_velocity[12]',
        'encoder_joint_velocity[13]',
        'encoder_joint_velocity[14]',
        'encoder_joint_velocity[15]',
        'encoder_joint_velocity[16]',
        'encoder_joint_velocity[17]',
    ]

    variable3=[
        'robot_angular_velocity[0]',
        'imu_rpy_angle[0]',
        'robot_angular_velocity[1]',
        'imu_rpy_angle[1]',
        'robot_angular_velocity[2]',
        'imu_rpy_angle[2]',
    ]

    variable4=[
        'robot_projected_gravity[0]',
        'imu_rotate_velocity[0]',
        'robot_projected_gravity[1]',
        'imu_rotate_velocity[1]',
        'robot_projected_gravity[2]',
        'imu_rotate_velocity[2]',
    ]

    g_plotter1=graphPlotter(variable_to_plot=variable3,
                            point_num=300,
                            y_range=[-2,2])
    ani1=FuncAnimation(fig=g_plotter1.fig,
                        func=update,
                        fargs=[g_plotter1],
                        frames=range(100000),
                        interval=1/g_plotter1.refresh_freq*1000,
                        blit=False)
    
    g_plotter2=graphPlotter(variable_to_plot=variable4,
                            point_num=300,
                            y_range=[-1,1])
    ani2=FuncAnimation(fig=g_plotter2.fig,
                        func=update,
                        fargs=[g_plotter2],
                        frames=range(100000),
                        interval=1/g_plotter2.refresh_freq*1000,
                        blit=False)
    plt.show()


    # 'imu_rpy_angle[0]',
    # 'imu_rpy_angle[1]',
    # 'imu_rpy_angle[2]',
    # 'imu_rotate_velocity[0]',
    # 'imu_rotate_velocity[1]',
    # 'imu_rotate_velocity[2]',
    # 'robot_angular_velocity[0]',
    # 'robot_angular_velocity[1]',
    # 'robot_angular_velocity[2]',
    # 'robot_projected_gravity[0]',
    # 'robot_projected_gravity[1]',
    # 'robot_projected_gravity[2]',
    # 'urdf_joint_position[2]',
    # 'urdf_joint_velocity[2]'
    # 'joint_position_command[2]',