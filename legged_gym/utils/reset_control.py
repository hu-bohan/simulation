import tkinter as tk
import math

class ResetController:
    def __init__(self):
        self.scrollbar1=None
        self.scrollbar2=None
        self.scrollbar3=None
        self.scrollbar4=None
        self.button_reset=None
        self.button_start=None
        self.reset_btn_clicked=False
        self.start_btn_clicked=False

    def create_window(self):
        # 创建一个函数来创建窗口和滚动条
        self.window = tk.Tk()
        self.window.title("reset位置")

        yaw_range=[-180,180]
        pitch_range=[0,180]
        roll_range=[-90,90]

        self.scrollbar1 = tk.Scale(self.window, label="yaw", from_=yaw_range[0], to=yaw_range[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.scrollbar2 = tk.Scale(self.window, label="pitch", from_=pitch_range[0], to=pitch_range[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.scrollbar2.set(180)
        self.scrollbar3 = tk.Scale(self.window, label="roll", from_=roll_range[0], to=roll_range[1],
                            orient="horizontal",length=300, width=30, sliderlength=40,resolution=0.01)
        self.button_reset = tk.Button(text="reset", command=self.reset_button_click)
        self.button_start = tk.Button(text="start", command=self.start_button_click)

        self.scrollbar1.pack()
        self.scrollbar2.pack()
        self.scrollbar3.pack()
        self.button_start.pack()
        self.button_reset.pack()

        # 设置窗口大小
        self.window.geometry("400x500")
        # 设置字体大小
        self.scrollbar1.config(font=("Consolas", 24))
        self.scrollbar2.config(font=("Consolas", 24))
        self.scrollbar3.config(font=("Consolas", 24))
        self.button_reset.config(font=("Consolas", 24))
        self.button_start.config(font=("Consolas", 24))

        # 居中显示窗口
        self._center_window()

        # 启动Tkinter主循环
        self.window.mainloop()

    def reset_button_click(self):
        self.reset_btn_clicked = True

    def start_button_click(self):
        self.start_btn_clicked = True

    # 创建一个函数来获取滚动条的值
    def get_values(self):
        if self.scrollbar1 is not None:
            return(self.scrollbar1.get()/180*math.pi,
                   self.scrollbar2.get()/180*math.pi,
                   self.scrollbar3.get()/180*math.pi)
        else:
            return(0.0,0.0,0.0)

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


if __name__=="__main__":
    r=ResetController()
    r.create_window()