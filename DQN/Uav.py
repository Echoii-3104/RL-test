import numpy as np  # NumPy是Python的一种开源的数值计算扩展
import random as rd
import time
import sys
if sys.version_info.major == 2:  # 查看版本号
    import Tkinter as tk  # 重命名Tkinter，且后面调用时只能使用重命名后的名字----注：Tkinter 是使用 python 进行窗口视窗设计的模块
else:
    import tkinter as tk

class UAV(tk.Tk, object):
    def __init__(self):
        super(UAV, self).__init__()
        self.x = 0.0
        self.y = 0.0


