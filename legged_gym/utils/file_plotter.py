import csv
import matplotlib.pyplot as plt
import numpy as np

path="/home/xuxin/allCode/RL_roll_recovery/logs/observation"
# file="observation@2024-04-15_02-33-05.csv"
file="observation@2024-04-24_21-11-17.csv"

file_path=path+"/"+file

# 读取CSV文件
timestamps = []
#18+18+18+18+3
joint_pos=[[] for _ in range(18)]
joint_vel=[[] for _ in range(18)]
joint_torque=[[] for _ in range(18)]
net_action=[[] for _ in range(18)]
command=[[] for _ in range(3)]

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        timestamp = "{:.2f}".format(float(row[0]))  # 格式化时间戳为最多两位小数
        timestamps.append(timestamp)

        for i in range(1, 19):
            joint_pos[i-1].append(float(row[i]))
        for i in range(19, 37):
            joint_vel[i-19].append(float(row[i]))
        for i in range(37, 55):
            joint_torque[i-37].append(float(row[i]))
        for i in range(55, 73):
            net_action[i-55].append(float(row[i]))
        for i in range(73, 76):
            command[i-73].append(float(row[i]))

x_range=[0,len(timestamps)]
# x_range=[100,400]
timestamps=np.array(timestamps)[x_range[0]:x_range[1]]
joint_pos=np.array(joint_pos)[:,x_range[0]:x_range[1]]
joint_vel=np.array(joint_vel)[:,x_range[0]:x_range[1]]
joint_torque=np.array(joint_torque)[:,x_range[0]:x_range[1]]
net_action=np.array(net_action)[:,x_range[0]:x_range[1]]
command=np.array(command)[:,x_range[0]:x_range[1]]

def call_back(event):
    axtemp=event.inaxes
    x_min, x_max = axtemp.get_xlim()
    y_min, y_max = axtemp.get_ylim()
    x_change = (x_max - x_min) / 10
    y_change = (y_max - y_min) / 10
    if event.button == 'up':
        axtemp.set(xlim=(x_min + x_change, x_max - x_change))
        axtemp.set(ylim=(y_min + y_change, y_max - y_change))
        # print('up')
    elif event.button == 'down':
        axtemp.set(xlim=(x_min - x_change, x_max + x_change))
        axtemp.set(ylim=(y_min - y_change, y_max + y_change))
        # print('down')
    event.canvas.draw_idle() # 绘图动作实时反映在图像上

tick_spacing = 100
x_ticks = range(0, len(timestamps), tick_spacing)

# 绘制不同类型的传感器数据曲线
################ joint position ################
fig1 = plt.figure(figsize=(12, 6))
ax=fig1.subplots()
for i in range(18):
    ax.plot(timestamps,joint_pos[i], label=f'joint {i+1}')
ax.set_xticks(x_ticks)
ax.set_xlabel('time')
ax.set_ylabel('joint position')
ax.legend(loc='upper right')
fig1.show()
fig1.canvas.mpl_connect('scroll_event', call_back)

################ joint velocity ################
fig2 = plt.figure(figsize=(12, 6))
ax=fig2.subplots()
for i in range(18):
    ax.plot(timestamps,joint_vel[i], label=f'joint {i+1}')
ax.set_xticks(x_ticks)
ax.set_xlabel('time')
ax.set_ylabel('joint velocity')
ax.legend(loc='upper right')
fig2.show()
fig2.canvas.mpl_connect('scroll_event', call_back)

################ joint torque ################
fig3 = plt.figure(figsize=(12, 6))
ax=fig3.subplots()
for i in range(18):
    ax.plot(timestamps,joint_torque[i], label=f'joint {i+1}')
ax.set_xticks(x_ticks)
ax.set_xlabel('time')
ax.set_ylabel('joint torque')
ax.legend(loc='upper right')
fig3.show()
fig3.canvas.mpl_connect('scroll_event', call_back)

################ net action ################
fig4 = plt.figure(figsize=(12, 6))
ax=fig4.subplots()
for i in range(18):
    ax.plot(timestamps,net_action[i], label=f'joint {i+1}')
ax.set_xticks(x_ticks)
ax.set_xlabel('time')
ax.set_ylabel('net action')
ax.legend(loc='upper right')
fig4.show()
fig4.canvas.mpl_connect('scroll_event', call_back)

################ command ################
# fig5 = plt.figure(figsize=(12, 6))
# ax=fig5.subplots()
# for i in range(3):
#     ax.plot(timestamps,command[i], label=f'command {i+1}')
# ax.set_xticks(x_ticks)
# ax.set_xlabel('time')
# ax.set_ylabel('command')
# ax.legend(loc='upper right')
# fig5.show()
# fig5.canvas.mpl_connect('scroll_event', call_back)


plt.show()
