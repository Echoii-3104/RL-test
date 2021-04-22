"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

"""
import numpy as np  # NumPy是Python的一种开源的数值计算扩展
import random as rd
import math as mt
import time
import sys
np.set_printoptions(suppress=True)
import array as ay
from Ue import IoTD
from Uav import UAV
if sys.version_info.major == 2:  # 查看版本号
    import Tkinter as tk  # 重命名Tkinter，且后面调用时只能使用重命名后的名字----注：Tkinter 是使用 python 进行窗口视窗设计的模块
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


# 定义了一个Maze环境，该环境下包括如下方法：
# 初始化方法，继承了tkinter父类
# 建立maze环境的方法，步骤如下：1：定义画布canvas；2：创建画布网格；
class Maze:  # 属于类tkinter的子类，可以继承类tkinter的所有方法
    def __init__(self):  # def=define 定义一个函数  init # python中类的专有函数：构造函数，生成对象时调用
        # super(Maze, self).__init__()  # super(Maze, self).init() 调用super函数，解决构造方法中的初始值无法继承的问题
        self.UEs = 20   # 定义地面节点数量
        self.UAVs = 1   # 定义UAV数量
        self.UAV_hovering_pos = 4  # UAV悬停位置数量
        self.X_range = 1000   # X轴范围为5km
        self.Y_range = 1000   # Y轴范围为5km
        # 定义UEs的位置(x,y)
        self.UE_locations = np.zeros((self.UEs, 2), dtype=float)
        # 定义UAV的位置
        self.UAV_locations = np.zeros((self.UAVs, 2), dtype=float)
        # print(self.UE_locations)
        # 随机生成初始位置坐标，且在整个运行过程中，所有位置坐标不可变动
        for i in range(len(self.UE_locations)):
            self.UE_locations[i, 0] = rd.random() * self.X_range
            self.UE_locations[i, 1] = rd.random() * self.Y_range
        # print(self.UE_locations)

        for i in range(len(self.UAV_locations)):
            self.UAV_locations[i, 0] = rd.random() * self.X_range
            self.UAV_locations[i, 1] = rd.random() * self.Y_range

        # 定义UAV的Hovering位置坐标
        self.UAV_hovering_locations = np.zeros((self.UAV_hovering_pos, 2), dtype=float)
        # 随机生成UAV的hovering位置坐标
        for i in range(len(self.UAV_hovering_locations)):
            self.UAV_hovering_locations[i, 0] = rd.random() * self.X_range
            self.UAV_hovering_locations[i, 1] = rd.random() * self.Y_range
        # print(self.UAV_hovering_locations)

        # 定义通信与计算配置参数
        self.UAV_height = 20  # 无人机飞行高度50m
        self.C = 30  # the most connections for UAV
        self.a = 4.88
        self.b = 0.49
        self.c = 3e8  # the speed of light 300000km/s
        self.ita_LoS = 10 ** 0.01
        self.ita_NLoS = 10 ** 2.1
        self.fc = 2e3  # 载波频率 / MHz
        self.B = 10e6
        self.g0 = 1e-3  # 1m信道增益 -30dB
        self.npp = 1e-9  # 噪声功率 -60dBm
        self.ki = 1e-27  # effective switched capacitance
        self.ita_0 = 0.8  # 能量收集效率energy conversation efficiency

        self.fl_max = 6e5  # 用户最大计算频率1e7 - 10MHz
        self.fo_max = 5e7  # 无人机最大计算频率1e8 - 100MHz
        self.p_max = 0.1  # 无人机最大充电功率 1W

        self.omiga = 1e-10  # 梯度下降步长
        self.R_max = 10  # 总算法最大迭代次数
        self.K_max = 100  # 用户调度算法最大迭代次数
        self.e1 = 1e-10  # 总算法最小迭代差异容忍
        self.e2 = 1e-12  # 用户调度算法最小迭代差异容忍

        # 实例化UEs个IoTD对象, 要不断摸索python中强大的方法
        self.iotds = np.array([IoTD() for i in range(self.UEs)])  # 该数组为1行，10列
        # print(self.iotds)
        # iotds变量初始化
        for i in range(self.UEs):
            self.iotds[i].x = self.UE_locations[i, 0]
            self.iotds[i].y = self.UE_locations[i, 1]
            # print(self.iotds[i].x)

        # self.iotds2 = np.array([IoTD() for i in range(self.UEs)]).reshape([2,5])  # 调整数组格式为2行，5列
        # print(self.iotds2)

        # 实例化1个UAV对象
        self.uavs = np.array([UAV() for i in range(self.UAVs)])  # 该数组为1行，10列
        # print(self.uavs)
        for i in range(self.UAVs):
            self.uavs[i].x = self.UAV_locations[i, 0]
            self.uavs[i].y = self.UAV_locations[i, 1]
            # print(self.uavs[i].y)

        # 定义动作空间大小
        self.action_number = (2 * self.UAV_hovering_pos) ** self.UEs
        # 定义动作空间数组
        # 每个UE两个动作，一个是row，一个是a
        self.single_action = np.zeros(2 * self.UEs, dtype=int)
        # 定义每个状态占据的空间,
        self.n_features = 1
        # 定义行为空间大小，即采样的大小
        self.n_actions = 2 * self.UAV_hovering_pos  # 行为空间大小为2M

        self.sample_actions = 30

    # 重置环境
    # 在并行优化的DQN中，这不是一个序列优化问题，因此令：state=action
    def reset(self):
        # 随机生成位置
        # iotds变量初始化
        #         self.t_energy_harvesting = 0  # 能量收割时间
        #         self.t_local = 0  # 本地计算时间
        #         self.rou = 0  # 是否卸载决策变量
        #         self.a = 0  # 悬停位置选择变量
        #         self.f_local = 0  # 本地计算频率
        #         self.p_energy_harvesting = 0  # 能量收割分配功率
        #         self.f_offloading = 0  # 卸载租用计算频率
        for i in range(self.UEs):
            self.iotds[i].x = self.UE_locations[i, 0]
            self.iotds[i].y = self.UE_locations[i, 1]
            self.iotds[i].t_energy_harvesting = 0
            self.iotds[i].t_local = 0
            self.iotds[i].rou = 0  # 全部选择本地计算，0，1
            self.iotds[i].a = 0  # 稍后选择最近位置进行连接，1，2，3，。。。，M
            self.iotds[i].f_local = self.fl_max  # 本地计算用最大计算频率
            self.iotds[i].p_energy_harvesting = self.p_max / self.UEs  # 能量收集功率初始化时平均分配
            self.iotds[i].f_offloading = 0  # 初始情况下，所有UE进行本地计算，所以租用的卸载频率为0

        # 随便选择一个位置进行连接，只要满足连接数最大约束即可
        A = np.zeros(self.UAV_hovering_pos, dtype=int)
        for i in range(self.UEs):
            while True:
                ind = rd.randint(1, 3)
                # print(ind)
                if A[ind-1] < self.C:
                    self.iotds[i].a = ind
                    A[ind-1] = A[ind-1] + 1
                    break
                else:
                    continue

        # # 选择最近的位置进行连接传能
        #
        # for i in range(self.UEs):
        #     # print(self.iotds[i].a)
        #     # print(self.UEs)
        #     # calculate the distance from UE to Hovering Positions
        #     min_distance = 100.0 ** 3
        #     for j in range(self.UAV_hovering_pos):
        #         # 三维空间的直线距离
        #         distance = ((self.iotds[i].x - self.UAV_hovering_locations[j, 0]) ** 2 + (
        #                     self.iotds[i].y - self.UAV_hovering_locations[j, 1]) ** 2 + self.UAV_height ** 2) ** 0.5
        #         # 先判断是否满足约束sum(a)<=C
        #         print("j = "+str(j)+" A[j]= "+str(A[j]) + "    " + str(distance) + "    " + str(self.iotds[i].a))
        #         if distance < min_distance:
        #             min_distance = distance
        #             self.iotds[i].a = j + 1  # index从0开始，因此j+1表示hovering位置是从1开始
        #             print("self.UEs = "+str(self.UEs)+"  i = "+str(i)+"   self.iotds[i].a = "+str(self.iotds[i].a))
        #
        #     if self.iotds[i].a >= 0:
        #         # 下面的索引要用self.iotds[i].a-1，表示数组索引是从0开始
        #         A[self.iotds[i].a - 1] = A[self.iotds[i].a - 1] + 1  # 注意：这里当index是-1时，代表数组中最后一个元素
        #         # Peh[self.iotds[i].a-1] = Peh[self.iotds[i].a-1] + self.iotds[i].send_power
        #
        # # 调整每个位置上UAV的个数
        for j in range(self.UAV_hovering_pos):
            print("A[j] = "+str(A[j]))

        for i in range(self.UAVs):
            self.uavs[i].x = self.UAV_locations[i, 0]
            self.uavs[i].y = self.UAV_locations[i, 1]

        # 计算初始状态：初始状态----所有UEs选择最近的位置进行卸载和充电，所有UEs都选择在本地计算
        # 在初始化的时候，都设定初始值，只要初值可行即可，不需要调用优化p和f函数
        action_rou_a_p_f = []
        ini_action_rou_a = []
        for i in range(self.UEs):
            action_rou_a_p_f.append(self.iotds[i].rou)
            action_rou_a_p_f.append(self.iotds[i].a)
            action_rou_a_p_f.append(self.iotds[i].p_energy_harvesting)
            action_rou_a_p_f.append(self.iotds[i].f_offloading)
            ini_action_rou_a.append(self.iotds[i].rou)
            ini_action_rou_a.append(self.iotds[i].a)

        # print("in_reset_function_ini_action_rou_a="+str(ini_action_rou_a))
        # print("action_rou_a_p_f="+str(action_rou_a_p_f))
        init_state, feasible, reward = self.calculateState(action_rou_a_p_f)
        # print("ini_state="+str(init_state))
        # print("ini_feasible=" + str(feasible))



        # print("---------------in env_Qiang self.iotds-------------")
        aaaa = []
        for i in range(self.UEs):
            aaaa.append(self.iotds[i].rou)
            aaaa.append(self.iotds[i].a)
        print("aaaa = "+str(aaaa))

        return init_state, self.iotds

    def transmissionTime(self, iotd):
        # selected position coordinates are:
        # self.UAV_hovering_locations[iotd.a-1,0]  self.UAV_hovering_locations[iotd.a-1,1]
        # 三维空间的直线距离
        distance3D = ((iotd.x - self.UAV_hovering_locations[iotd.a-1, 0])**2 + (iotd.y - self.UAV_hovering_locations[iotd.a-1, 1])**2 + self.UAV_height**2)**0.5
        # 平面直线距离
        distance2D = ((iotd.x - self.UAV_hovering_locations[iotd.a-1, 0])**2 + (iotd.y - self.UAV_hovering_locations[iotd.a-1, 1])**2)**0.5
        # 增益
        g = self.g0 / distance3D
        P_LoS = (1 + self.a * mt.exp(-self.b*(mt.atan(self.UAV_height/distance2D) - self.a)))**(-1)  # the probability of LoS connection
        P_NLoS = 1 - P_LoS  # the probability of NLoS connection
        PL_LoS = 20 * mt.log(4 * mt.pi * self.fc * distance3D / self.c) + P_LoS * self.ita_LoS + P_NLoS * self.ita_NLoS  # the average pathloss
        Pu = iotd.send_power / (10 ** (PL_LoS / 10))  # the average received power
        SINR = Pu * g / (self.npp ** 2)
        rate = self.B * mt.log2(float(1+SINR))
        trantime = iotd.data / rate
        return trantime

    def offloadingTime(self, iotd):
        # print("iotd.rou="+str(iotd.rou)+"  iotd.f_offloading="+str(iotd.f_offloading))
        offtime = 100 ** 6
        if iotd.rou == 1:
            offtime = iotd.F / iotd.f_offloading
        else:
            offtime = 100 ** 6
        return offtime

    # for complicated state
    def calculateState(self, action_rou_a_p_f):
        # 计算状态就是1：判断当前取值组合是否可行；2：计算目标函数值；3：返回状态即取值组合+目标函数值
        # 首先解析action数组：action_rou_a_p_f = [rou1, a1, p_energy_harvesting1, f_offloading1, rou2, a2,
        # p_energy_harvesting2, f_offloading2,..., rouN, aN, p_energy_harvestingN, f_offloadingN], N = self.UEs
        rou_a = []
        # 根据action内容给所有iotds赋值
        for i in range(self.UEs):
            # 每隔2个元素处理一次
            self.iotds[i].rou = action_rou_a_p_f[i*4]
            self.iotds[i].a = action_rou_a_p_f[i*4+1]
            self.iotds[i].p_energy_harvesting = action_rou_a_p_f[i*4+2]
            self.iotds[i].f_offloading = action_rou_a_p_f[i*4+3]
            rou_a.append(self.iotds[i].rou)
            rou_a.append(self.iotds[i].a)
        # print("in_calculatestate_function_test_rou_a = "+str(rou_a))

        # 根据新的行为计算奖励
        # 计算所有约束条件是否满足
        A = np.zeros(self.UAV_hovering_pos, dtype=int)
        # Peh = np.zeros(self.UAV_hovering_pos, dtype=float)
        # 定义状态为5个部分：
        # 约束为self.UEs + 3 * self.UAV_hovering_pos个，目标函数值为1个
        # state = np.zeros(self.UEs + 3 * self.UAV_hovering_pos + 1, dtype=float)
        # 根据传进来的action_rou_a_p_f中的a计算连接数是否超出预期
        for i in range(self.UEs):
            for j in range(self.UAV_hovering_pos):
                if self.iotds[i].a == j+1:
                    A[j] = A[j] + 1
        feasible = 1
        # print("calculate_state_feasible="+str(feasible))
        # 判断是否满足sum(pi_ch)<puav_max，计算状态state
        state_UE = np.zeros(self.UEs, dtype=float)
        for i in range(self.UEs):
            C1 = self.iotds[i].rou * (self.iotds[i].a / self.iotds[i].a) * \
                 (self.transmissionTime(self.iotds[i]) + self.offloadingTime(self.iotds[i])) \
                 - self.iotds[i].t_delay
            state_UE[i] = C1
            if C1 > 0:
                feasible = -1
                print("C1 = "+str(C1+self.iotds[i].t_delay)+"  the i is: "+str(i)+ "  the delay is: "+str(self.iotds[i].t_delay))
                # return [], feasible

        state_pos_f = np.zeros(self.UAV_hovering_pos, dtype=float)
        state_pos_p = np.zeros(self.UAV_hovering_pos, dtype=float)
        state_pos_a = np.zeros(self.UAV_hovering_pos, dtype=float)
        for j in range(self.UAV_hovering_pos):
            C2 = 0.0
            C3 = 0.0
            for i in range(self.UEs):
                if self.iotds[i].a == j + 1:
                    C2 = C2 + self.iotds[i].rou * (self.iotds[i].a / self.iotds[i].a) * self.iotds[i].f_offloading
                    C3 = C3 + self.iotds[i].p_energy_harvesting * (self.iotds[i].a / self.iotds[i].a)
            C2 = C2 - self.fo_max
            if C2 > 0:
                feasible = -1
                print("C2 = " + str(C2+self.fo_max) + "  the j is: " + str(j) + "  the fo_max is: " + str(self.fo_max))
                # return [], feasible
            C3 = C3 - self.p_max
            if C3 > 0.001:
                feasible = -1
                print("C3 = " + str(C3+self.p_max) + "  the j is: " + str(j) + "  the p_max is: " + str(self.p_max))
                # return [], feasible
            state_pos_f[j] = C2
            state_pos_p[j] = C3
            state_pos_a[j] = A[j] - self.C
            if state_pos_a[j] > 0:
                feasible = -1
                print("Aj = " + str(A[j]) + "  the j is: " + str(j) + "  the p_max is: " + str(self.C))
                # return [], feasible

        for j in range(self.UAV_hovering_pos):
            if A[j] > self.C:
                feasible = -1

        # 计算目标函数值
        objvalue = 0.0
        for i in range(self.UEs):
            # calculate Til
            til = self.iotds[i].F / self.iotds[i].f_local
            # calculate Tieh
            tieh1 = (1 - self.iotds[i].rou) * self.ki * self.iotds[i].F * (self.iotds[i].f_local ** 2)
            tieh2 = self.iotds[i].rou * self.iotds[i].send_power * self.transmissionTime(self.iotds[i])
            distance3D = ((self.iotds[i].x - self.UAV_hovering_locations[self.iotds[i].a - 1, 0]) ** 2 + (
                    self.iotds[i].y - self.UAV_hovering_locations[
                self.iotds[i].a - 1, 1]) ** 2 + self.UAV_height ** 2) ** 0.5
            tieh3 = self.ita_0 * self.g0 / (distance3D * self.iotds[i].p_energy_harvesting)
            tieh = (tieh1 + tieh2) / tieh3
            # calculate Titr
            titr = self.transmissionTime(self.iotds[i])
            # calculate Tio
            if self.iotds[i].rou == 1:
                tio = self.iotds[i].F / self.iotds[i].f_offloading
            else:
                tio = 100 ** 6
            obj_sub = (1 - self.iotds[i].rou) * til + (self.iotds[i].a / self.iotds[i].a) * (
                        tieh + self.iotds[i].rou * (titr + tio))
            objvalue = objvalue + obj_sub
        # 在所有约束条件满足情况下，计算目标函数值----初始状态值
        # 约束为self.UEs + 3 * self.UAV_hovering_pos个，目标函数值为1个
        # state = np.zeros(self.UEs + 3 * self.UAV_hovering_pos + 1, dtype=float)
        # 状态调整为动作+目标函数值，约束条件不要放入状态中，因为在这种并行训练模式下，所有约束都是可行解
        state = []
        # 状态就用目标函数值
        if feasible == 1:
            state.append((100-objvalue)/100)
            # 奖励的计算依据包含在状态的最后一个元素中
            reward = state[len(state)-1]
        else:
            state.append(0)
            reward = 0

        # print("action_rou_a_p_f="+str(action_rou_a_p_f))

        return state, feasible, reward

    # optimize the p_energy_harvesting and f_offloading
    def getOptimalPehAndFo(self, action_rou_a):
        action_rou_a_p_f = []
        # 为了和原版程序一致，定义二维数组
        objai = np.zeros([self.UEs, self.UAV_hovering_pos])
        objpi = np.zeros([self.UEs, self.UAV_hovering_pos])
        objfi = np.zeros([self.UEs, self.UAV_hovering_pos])
        for i in range(self.UEs):
            for j in range(self.UAV_hovering_pos):
                if action_rou_a[i*2 + 1] == j+1:
                    objai[i, j] = 1


        AA = np.zeros([self.UEs, self.UAV_hovering_pos], dtype=float)  # self.UEs=N, self.UAV_hovering_pos=M
        for j in range(self.UAV_hovering_pos):
            a_A_j = 0
            a_F_j = 0
            for i in range(self.UEs):
                # 三维空间的直线距离
                distance3D = ((self.iotds[i].x - self.UAV_hovering_locations[action_rou_a[i*2+1] - 1, 0]) ** 2 + (
                            self.iotds[i].y - self.UAV_hovering_locations[action_rou_a[i*2+1] - 1, 1]) ** 2
                              + self.UAV_height ** 2) ** 0.5
                # 增益
                g = self.g0 / distance3D
                AA[i, j] = ((1-action_rou_a[i*2])*self.ki*self.iotds[i].F*(self.iotds[i].f_local**2)
                            + action_rou_a[i*2]*self.iotds[i].send_power*self.transmissionTime(self.iotds[i])) / (self.ita_0 * g)
                # self.iotds[i].a的取值区间是1~self.UAV_hovering_pos
                # self.iotds[i].rou的取值区间是0,1，因此不需要与自己相除
                a_A_j = a_A_j + objai[i, j] * mt.sqrt(AA[i, j])  # 论文后面会改正，先按照这个来计算
                a_F_j = a_F_j + objai[i, j] * action_rou_a[i*2] * mt.sqrt(self.iotds[i].F)

            # 开始真正进行优化计算
            for i in range(self.UEs):
                # 如果选择了第j个位置进行连接
                if objai[i, j] == 1:  # j的取值从0开始，action_rou_a[i*2+1]的a的取值从1开始，所以j=0,1,2对应着a=1,2,3
                    f_min = self.iotds[i].F / (self.iotds[i].t_delay - self.transmissionTime(self.iotds[i]))
                # 如果没有选择第j个位置进行连接
                else:
                    f_min = 0

                if objai[i, j] == 0:
                    objpi[i, j] = 0
                    objfi[i, j] = 0
                    continue
                else:
                    if a_A_j == 0:
                        objpi[i, j] = self.p_max
                    else:
                        objpi[i, j] = self.p_max * mt.sqrt(AA[i, j])/a_A_j


                    if a_F_j != 0:
                        f_opt = self.fo_max * mt.sqrt(self.iotds[i].F)/a_F_j
                    else:
                        f_opt = 0
                        # self.iotds[i].f_offloading = f_opt

                    if a_F_j != 0 and f_min <= f_opt:
                        objfi[i, j] = f_opt
                    else:
                        objfi[i, j] = self.fo_max

                    self.iotds[i].t_energy_harvesting = AA[i, j]/self.iotds[i].p_energy_harvesting

                if action_rou_a[i*2] == 0:
                    self.iotds[i].f_offloading = 0

        for i in range(self.UEs):
            self.iotds[i].p_energy_harvesting = 0
            self.iotds[i].f_offloading = 0
            for j in range(self.UAV_hovering_pos):
                self.iotds[i].p_energy_harvesting = self.iotds[i].p_energy_harvesting + objpi[i, j]
                self.iotds[i].f_offloading = self.iotds[i].f_offloading + objfi[i, j]

        # 将优化后的结果装入数组action_rou_a_p_f
        for i in range(self.UEs):
            action_rou_a_p_f.append(action_rou_a[i*2])
            action_rou_a_p_f.append(action_rou_a[i*2+1])
            action_rou_a_p_f.append(self.iotds[i].p_energy_harvesting)
            action_rou_a_p_f.append(self.iotds[i].f_offloading)

        return action_rou_a_p_f

    # 当DQN选择好行为action后，执行该行为，返回下一个状态，计算奖励值
    def step(self, single_ue_action_rou_a, index):
        # 首先解析action数组：action_rou_a_p_f = [rou1, a1, p_energy_harvesting1, f_offloading1, rou2, a2,
        # p_energy_harvesting2, f_offloading2,..., rouN, aN, p_energy_harvestingN, f_offloadingN], N = self.UEs
        self.iotds[index].rou = single_ue_action_rou_a[0]
        self.iotds[index].a = single_ue_action_rou_a[1]
        actions_rou_a = []
        # 根据所有iotds得到action_rou_a_p_f
        for i in range(self.UEs):
            actions_rou_a.append(self.iotds[i].rou)
            actions_rou_a.append(self.iotds[i].a)
        # get the optimal p and f
        # print("In step function, the action is: "+str(actions_rou_a))
        # 测试是否改变了rou_a
        rou_a = []
        # 根据action内容给所有iotds赋值
        for i in range(self.UEs):
            # 每隔2个元素处理一次
            rou_a.append(self.iotds[i].rou)
            rou_a.append(self.iotds[i].a)
        # print("before_in_step_function_test_rou_a = "+str(rou_a))

        # print("in_step_actions_rou_a = "+str(actions_rou_a))
        actions_rou_a_p_f = self.getOptimalPehAndFo(actions_rou_a)
        # print("in_step_actions_rou_a_p_f = " + str(actions_rou_a_p_f))
        # 测试是否改变了rou_a
        rou_a = []
        # 根据action内容给所有iotds赋值
        for i in range(self.UEs):
            # 每隔2个元素处理一次
            rou_a.append(self.iotds[i].rou)
            rou_a.append(self.iotds[i].a)
        # print("after_in_step_function_test_rou_a = "+str(rou_a))
        # print("actions_rou_a_p_f="+str(actions_rou_a_p_f))


        # 根据新的行为计算奖励
        # state_next = []
        state_next, feasible, reward = self.calculateState(actions_rou_a_p_f)
        print("state_next = "+str(state_next)+"  feasible = "+str(feasible)+"  reward = "+str(reward))
        # if feasible == -1:
        #     state_next
        #     reward = 100000
        # else:
        #     reward = 100000-state_next[len(state_next)-1]
        # # 这里不存在中间训练时死棋的现象，直接返回下一个状态和奖励就可以
        # state = []
        # state.append(state_next)
        return state_next, reward, feasible

    # def render(self):
    #     # time.sleep(0.01)
    #     self.update()  # update()方法用于更新字典中的键/值对，可以修改存在的键对应的值，也可以添加新的键/值对到字典中


# 测试主函数，用来验证环境Env是否编写正确
if __name__ == "__main__":
    # maze game
    env = Maze()  # 从main函数开始运行，实例化一个Maze类
    env.reset()
    # env.sampleActions()
