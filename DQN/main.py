from Env import Maze
from DQN-brain import DeepQNetwork
import numpy as np

# DQN与环境交互最重要的部分
def run_maze(env):
    # 用来控制什么时候学习
    step = 0     #记录目前的步骤
    observation, UE_iotds = env.reset()  # 调用Maze类中的重置方法，使得红色方块位于初始位置，重新开始

    best_objective = -1000000.0
    best_strategy = []

    for episode in range(500):
        index = episode
        # 对UEs一个一个取，进行训练
        # 从0开始，0，1，2，3，4，。。。，self.UEs-1
        index = index % env.UEs
        print("***********Training the " + str(index) + "th UEs*************")
        while True:
            # index = 3
            # 先假设优化第一个UE
            print("-----------------round ("+str(step)+") for the " + str(index) + "th UE------------------")
            print("The state of "+str(step)+" is "+str(observation))
            # 测试是否改变了rou_a
            rou_a = []
            A = np.zeros(env.UAV_hovering_pos, dtype=int)
            # 根据action内容给所有iotds赋值
            for i in range(env.UEs):
                # 每隔2个元素处理一次
                rou_a.append(UE_iotds[i].rou)
                rou_a.append(UE_iotds[i].a)
                for j in range(env.UAV_hovering_pos):
                    if UE_iotds[i].a == j+1:
                        A[j] = A[j] + 1
            # print("before_choose_action_in_main_step_function_test_rou_a = " + str(rou_a))
            # for j in range(env.UAV_hovering_pos):
            #     print(A[j])

            action = RL.choose_action(np.array(observation), A)
            # print("choosed_action=" + str(action))
            # 测试是否改变了rou_a
            rou_a = []
            # 根据action内容给所有iotds赋值
            for i in range(env.UEs):
                # 每隔2个元素处理一次
                rou_a.append(env.iotds[i].rou)
                rou_a.append(env.iotds[i].a)
            # print("after_choose_action_in_main_step_function_test_rou_a = " + str(rou_a))
            # 解析该action
            action_rou_a = []
            # action = 0, 1, 2
            if action < env.UAV_hovering_pos:
                action_rou_a.append(0)  # rou = 0
                action_rou_a.append(action+1)  # a = action + 1
            else:
                action_rou_a.append(1)  # rou = 1
                action_rou_a.append(action+1-env.UAV_hovering_pos)

            # print("choosed_in_main_function_action_rou_a="+str(action_rou_a))
            # index = 0
            observation_, reward, feasible = env.step(action_rou_a, index)

            # print("next_state="+str(observation_)+"  reward="+str(reward))
            # DQN 存储记忆，将当前的状态，行为，奖励以及下一个状态存下来
            all_action_rou_a = []
            for i in range(env.UEs):
                all_action_rou_a.append(env.iotds[i].rou)
                all_action_rou_a.append(env.iotds[i].a)
                # print("all_action_rou_a="+str(all_action_rou_a))

            tmp = []
            # print("feasible="+str(feasible))
            if feasible == 1:
                # print("best_objective="+str(best_objective)+" reward="+str(reward))
                if best_objective < reward:
                    best_objective = reward
                    tmp.extend(observation)
                    tmp.extend(all_action_rou_a)
                    tmp.append(reward)
                    tmp.extend(observation_)
                    # print("tmp="+str(tmp))
                    best_strategy = tmp

            RL.store_transition(observation, all_action_rou_a, reward, observation_)

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (step > 200) and (step % 5 == 0):  #积累200步后每五步学习一次
                print("RL.learn")
                cost_his = RL.learn()

            # swap observation # 将下一个 state_ 变为 下次循环的 state
            observation = observation_

            # break while loop when end of this episode  # 如果终止, 就跳出循环
            # 总步数
            step += 1
            if step % 1000 == 0:
                # step = 0
                break
            # print("The step is "+str(step))
            print("    ")
        # break

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(cost_his)), cost_his)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()


    print("best_strategy = " + str(best_strategy))


# if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
# 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
# 先运行下面的代码块
if __name__ == "__main__":
    # maze game
    env = Maze()  # 从main函数开始运行，实例化一个Maze类


    # 实例化一个DeepQNetwork类，经典配置
    # RL = DeepQNetwork(env.n_actions, env.n_features, env.sample_actions, env.UEs,
    #                   learning_rate=0.01,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
    #                   memory_size=2000,  # 记忆上限
    #                   # output_graph=True  # 是否输出 tensorboard 文件
    #                   )
    RL = DeepQNetwork(env.n_actions, env.n_features, env.sample_actions, env.UEs,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.7,  # 改变贪婪搜索概率
                      replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                      memory_size=2000,  # 记忆上限
                      # output_graph=True  # 是否输出 tensorboard 文件
                      )
    # after(delay_ms, callback=None, *args)，在延时100ms后，调用函数run_maze，此函数在本文件中被定义好
    run_maze(env)

    # 以下是一般大多数tkinter程序员都共有的步骤，它的代码做了以下这些事情：
    # 1.从痛苦inter模块中加载一个组件类。
    # 2.创建该组件类的实例为标签类
    # 3.在父组件中打包新标签。
    # 4.调用主循环，显示窗口，同时开始tkinter的事件循环。
    # mainloop方法最后执行，将标签显示在屏幕，进入等待状态（注：若组件未打包，则不会在窗口中显示），准备响应用户发起的GUI事件。
    # 在mainloop函数中，tkinter内部会监控这些事件，如键盘活动，鼠标单击等。事实上，tkinter的mainloop函数与下面的Python伪代码实质是一样的：
    # def mainloop():
    # 	while the main window has not been closed:
    # 		if an event has occurred:
    # 			run the associated event handler function

    # RL.plot_cost()  # 观看神经网络的误差曲线
