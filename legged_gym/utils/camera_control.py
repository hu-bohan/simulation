import tkinter as tk

class CameraController:
    def __init__(self):
        self.scrollbar1=None
        self.scrollbar2=None
        self.scrollbar3=None

    def create_window(self):
        # 创建一个函数来创建窗口和滚动条
        self.window = tk.Tk()
        self.window.title("相机调整")

        dist = [0, 10.0] # min max [m/s]
        yaw = [0, 360]   # min max [m/s]
        pitch = [-90, 90]    # min max [rad/s]

        self.scrollbar1 = tk.Scale(self.window, label="dist", from_=dist[0], to=dist[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.scrollbar2 = tk.Scale(self.window, label="yaw", from_=yaw[0], to=yaw[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.scrollbar3 = tk.Scale(self.window, label="pitch", from_=pitch[0], to=pitch[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.textbox = tk.Entry(self.window, font=("Consolas", 24))
        
        self.scrollbar1.pack()
        self.scrollbar2.pack()
        self.scrollbar3.pack()
        self.textbox.pack()

        self.scrollbar1.set(5)
        self.scrollbar2.set(270)
        self.scrollbar3.set(30)
        self.textbox.insert(0, "0")

        # 设置窗口大小
        self.window.geometry("400x350")
        # 设置字体大小
        self.scrollbar1.config(font=("Consolas", 24))
        self.scrollbar2.config(font=("Consolas", 24))
        self.scrollbar3.config(font=("Consolas", 24))
        # 居中显示窗口
        self._center_window()

        # 启动Tkinter主循环
        self.window.mainloop()

    # 创建一个函数来获取滚动条的值
    def get_values(self):
        if self.scrollbar1 is not None:
            return(self.scrollbar1.get(),
                    self.scrollbar2.get()/180*3.14,
                    self.scrollbar3.get()/180*3.14,
                    int(self.textbox.get()))
        else:
            return(0.0,0.0,0.0,0)

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



