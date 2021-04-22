import numpy as np  # NumPy是Python的一种开源的数值计算扩展
import random as rd
import time
import sys
if sys.version_info.major == 2:  # 查看版本号
    import Tkinter as tk  # 重命名Tkinter，且后面调用时只能使用重命名后的名字----注：Tkinter 是使用 python 进行窗口视窗设计的模块
else:
    import tkinter as tk

class IoTD(tk.Tk, object):
    def __init__(self):
        super(IoTD, self).__init__()
        self.x = 0.0
        self.y = 0.0
        self.send_power = 2.82e-3  # 用户发射功率为2.82mw
        self.data = 1e5  # 1e5 100kbits 传输数据包大小
        self.F = 1e6  # 1e6 1MHz 要消耗的CPU时钟周期
        self.t_delay = 2  # 最大延时容忍 2s
        self.t_energy_harvesting = 0  # 能量收割时间
        self.t_local = 0  # 本地计算时间
        self.rou = 0  # 是否卸载决策变量
        self.a = 0  # 悬停位置选择变量
        self.f_local = 0  # 本地计算频率
        self.p_energy_harvesting = 0  # 能量收割分配功率
        self.f_offloading = 0  # 卸载租用计算频率


