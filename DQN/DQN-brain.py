"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,  # 需要输出多少个action的值，就是控制的动作 如左右
            n_features,  # 要接受多少个观测状态----
            sample_actions,  # 采样的行为个数
            n_ues,           # UE的数量
            learning_rate=0.01,    # 学习率
            reward_decay=0.9,  # 奖励的衰减率 就是Sara里面的gamma
            e_greedy=0.9,  # 代表90%选择这个行为
            replace_target_iter=300,  # 隔多少步后 把估计神经网络的值全部复制给目标神经网络
            memory_size=500,  # 每个存储数据的大小
            batch_size=32,  # 随机梯度下降的值
            e_greedy_increment=None,  # 不断的缩小随机的范围
            output_graph=False,  # 是否输出图表
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.sample_actions = sample_actions
        self.n_ues = n_ues
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step  # 一共学习了多少步
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_] 初始化记忆库 RL.store_transition(observation, action, reward, observation_)
        # 总结：采用np.array()创建时需要几个维度就要用几个[ ]括起来，这种创建方式要给定数据；采用np.ones()或np.zeros()创建分别产生全1或全0的数据，
        # 用a.shape会输出你创建时的输入，创建时输入了几个维度输出就会用几个[ ]括起来，shape的返回值是一个元组，里面每个数字表示每一维的长度
        # 这里生成一个二维数组，包括self.memory_size行，n_features * 2 + 2列；
        # 存放经验回放数据，包括
        self.memory = np.zeros((self.memory_size, n_features * 3 + 2 * self.n_ues))  # 存储st, st+1, a, reward

        # consist of [target_net, evaluate_net] # 建立神经网络
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        # ------------------ 建立评估神经网络 -----------------
        # self.s为行数不限，列数为n_features的张量（数组）
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input  #输入状态值
        # self.q_target为行数不限，列数为n_actions的张量（数组）
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss  # 用来接收 q_target 的值, 这个之后会通过计算得到
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            # c_names非常重要，是用来储存神经网络参数的，在后面的学习中会把整个参数全部复制给目标神经网络
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            # 第一层，一个非常简单的神经元，y = relu（w1*x + b1） 输入进去的是状态值 输入给第二层神经网络的值
            with tf.variable_scope('l1'):
                # w1为n_features行，n_l1列的变量
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                # b1为1行，n_l1列的变量
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # tf.matmul(self.s, w1)得到的是不限行，n_l1列的变量
                # tf.matmul(self.s, w1) + b1得到的是1行，n_l1列的变量
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            # 第二层， 还是一个简单的神经元，y = w2*x + b2 输入是上一层的值，输出的是估计值，也就是Q估计
            with tf.variable_scope('l2'):
                # w2是n_l1行，n_actions列的变量
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                # b2是1行，n_actions列的变量
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                # tf.matmul(l1, w2)是1行，n_actions列的变量
                # q_eval是1行，n_actions列的变量，所以q_eval有n_actions个输出值，每个值代表一个action对应的Q估计
                self.q_eval = tf.matmul(l1, w2) + b2

        # 建立损失函数这里就是q_target - q_eval 来计算error q_target是目标神经网络的输出值，这里输入进来做期望
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        # 训练优化器
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        # 建立目标神经网络，目标神经网络的结构必须要跟估计神经网络一模一样，因为会做整个参数的赋值
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input # 输入 这里只有一个输入 因为他不需要做训练
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 储存记忆，从上往下储存，满了就循环
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = []
        transition.extend(s)
        transition.extend(a)
        transition.append(r)
        transition.extend(s_)
        # transition = np.hstack((s, [a, r], s_))
        print("Qiang_transition="+str(transition))


        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    # 选择动作，90%使用DQN选择，10%使用随机办法
    def choose_action(self, observation, A):
        # to have batch dimension when feed into tf placeholder
        # 因为observation加入时是一维的数值
        # np.newaxis 为 numpy.ndarray（多维数组）增加一个轴,多加入了一个行轴
        # print("observation-before np.newaxis="+str(observation))
        observation = observation[np.newaxis, :]
        # print("observation-after np.newaxis="+str(observation))
        # np.random.uniform生成均匀分布的随机数，默认0-1，大概率选择actions_value最大下的动作
        if np.random.uniform() < self.epsilon:
            # print(self.epsilon)
            # forward feed the observation and get q value for every actions
            # 向前输出 这里使用的是估计神经网络
            # 在一开始的时候，直接输入observation值，得到四个action值，选择最大那个，再返回去计算下一个状态和奖励值
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            print(actions_value)
            action = np.argmax(actions_value)
            # 解析action
            # if action < len(A):
            #     t =
            # else:
            #     action_rou_a.append(1)  # rou = 1
            #     action_rou_a.append(action+1-env.UAV_hovering_pos)
            # [[0.31960368 0.11395732 0.30785137 0.07434813]]
            # 2----因为序号为2的值最大，所以动作为2
            # print(action)
        else:
            # 小概率随机选择actions_value下的一个动作，np.random.randint用于生成一个指定范围内的整数
            # 该函数返回一个从[0, self.n_actions)之间的随机整数，不包括self.n_actions
            action = np.random.randint(0, self.n_actions)
            # print(self.n_actions)
            # print(action)
        return action

    def learn(self):
        # check to replace target parameters
        # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        # 从 memory 中随机抽取 batch_size 这么多记忆
        # 它是一个batch一个batch的训练的 不是说每一步就训练哪一步
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        # 这里分别计算一次q_next和q_eval即下一个目标Q值和估计Q值；
        # 计算目标Q值时，要self.s_，其值是batch_memory[:, -self.n_features:]
        # 计算估计Q值时，要self.s，其值是batch_memory[:, :self.n_features]
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })
        # print("batch_memory"+str(batch_memory[:, -self.n_features:]))
        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # batch_memory就是之前储存下来的记忆，根据不同的需要有所改变
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        # 训练 eval_net
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)  # 记录 cost 误差

        # print(self.cost_his)

        # increasing epsilon  # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return self.cost_his

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



