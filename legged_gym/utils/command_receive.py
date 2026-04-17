import tkinter as tk

class CommandReceiver:
    def __init__(self):
        self.scrollbar1=None
        self.scrollbar2=None
        self.scrollbar3=None
        self.scrollbar4=None

    def create_window(self):
        # 创建一个函数来创建窗口和滚动条
        self.window = tk.Tk()
        self.window.title("指令输入")

        lin_vel_x = [-1.0, 1.0] # min max [m/s]
        # lin_vel_x = [-0.3, 0.3] # min max [m/s]
        lin_vel_y = [-1.0, 1.0]   # min max [m/s]
        # lin_vel_y = [-0.3, 0.3]   # min max [m/s]
        ang_vel_yaw = [-1, 1]    # min max [rad/s]
        heading = [-180, 180]

        self.scrollbar1 = tk.Scale(self.window, label="x", from_=lin_vel_x[0], to=lin_vel_x[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.scrollbar2 = tk.Scale(self.window, label="y", from_=lin_vel_y[0], to=lin_vel_y[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.scrollbar3 = tk.Scale(self.window, label="r", from_=ang_vel_yaw[0], to=ang_vel_yaw[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.scrollbar4 = tk.Scale(self.window, label="h", from_=heading[0], to=heading[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.scrollbar1.pack()
        self.scrollbar2.pack()
        self.scrollbar3.pack()
        self.scrollbar4.pack()

        # 设置窗口大小
        self.window.geometry("400x500")
        # 设置字体大小
        self.scrollbar1.config(font=("Consolas", 24))
        self.scrollbar2.config(font=("Consolas", 24))
        self.scrollbar3.config(font=("Consolas", 24))
        self.scrollbar4.config(font=("Consolas", 24))

        # 居中显示窗口
        self._center_window()

        # 启动Tkinter主循环
        self.window.mainloop()


    # 创建一个函数来获取滚动条的值
    def get_values(self):
        if self.scrollbar1 is not None:
            return (self.scrollbar1.get(),
                    self.scrollbar2.get(),
                    self.scrollbar3.get(),
                    self.scrollbar4.get()/180*3.14)
        else:
            return (0.0,0.0,0.0,0.0)

    # 函数用于将窗口居中显示在屏幕上
    def _center_window(self):
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.window.geometry("+{}+{}".format(x, y))
