import tensorflow as tf
import gym
import random
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.pyplot as plt
import numpy as np


# 리플레이를 저장할 리스트
REPLAY_MEMORY = deque()
TMP_MEMORY = deque()

#하이퍼파라미터
learning_rate = 0.0001
discount_factor = 0.99
INITIAL_EPSILON = 1.00
episode_list = []
train_error_list = []
actions_list = []
max_episodes = 10000

total_count = 0
Max_reward = 0

# 테스트 에피소드 주기
TEST_PERIOD = 100

# src model에서 target model로 trainable variable copy 주기
COPY_PERIOD = 10000


episode = 0
mean_loss_value = 0
rAll = 0

class CNNDQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        # 네트워크 생성
        self.build_network()

    def build_network(self):
        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)  # X = Input Image
            self.Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
            self.x_image = tf.reshape(self.X, [-1, 80, 80, 5])

            self.W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 5, 32], stddev=0.1))
            self.b_conv1 = tf.Variable(tf.zeros(shape=[32]))
            self.h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1, 2, 2, 1], padding='VALID') + self.b_conv1)

            self.W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1))
            self.b_conv2 = tf.Variable(tf.zeros(shape=[64]))
            self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 4, 4, 1], padding='VALID') + self.b_conv2)

            self.W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
            self.b_conv3 = tf.Variable(tf.zeros( shape=[64]))
            self.h_conv3 = tf.nn.relu(
                tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding='VALID') + self.b_conv3)

            # 10 x 10 x 64
            self.h_fc1_pool = tf.reshape(self.h_conv3, [-1, 3136])

            self.W_fc1 = tf.Variable(tf.truncated_normal([3136, 512], stddev=0.1))
            self.b_fc1 = tf.Variable(tf.zeros(shape=[512]))
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_fc1_pool, self.W_fc1) + self.b_fc1)

            self.W_fc2 = tf.Variable(tf.truncated_normal([512, 64], stddev=0.1))
            self.b_fc2 = tf.Variable(tf.zeros(shape=[64]))
            self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

            self.W_fc3 = tf.Variable(tf.truncated_normal([64, self.output_size], stddev=0.1))
            self.b_fc3 = tf.Variable(tf.zeros( shape=[self.output_size]))
            self.Qpred = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3

        # 손실 함수 및 최적화 함수
        self.loss = tf.reduce_mean(tf.square(self.Y - self.Qpred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    # 예측한 Q값 구하기
    def predict(self, state):
        x = np.reshape(state, newshape=[1, self.input_size])
        pQ = self.session.run(self.Qpred, feed_dict={self.X: x})
        return pQ

    # e-greedy 를 사용하여 action값 구함
    def egreedy_action(self, epsilon, env, state):
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            Q_h = self.predict(state)
            action = np.argmax(Q_h)
        return action

def update_from_memory(mainDQN, targetDQN, batch_size):

    state_batch = np.ndarray(shape=[batch_size, mainDQN.input_size])
    y_batch = np.ndarray(shape=[batch_size, mainDQN.output_size])

    minibatch = random.sample(REPLAY_MEMORY, batch_size)
    i = 0

    for sample in minibatch:
        state, action, reward, new_state, done = sample         # unpacking

        one_hot_action = np.zeros(mainDQN.output_size)

        if done:
            one_hot_action[action] = reward
        else:
            one_hot_action[action] = reward + discount_factor * np.max(targetDQN.predict(new_state))

        state_batch[i] = np.reshape(state, newshape=[1, 80*80*5])
        y_batch[i] = one_hot_action
        i += 1

    loss_value, _ = mainDQN.session.run([mainDQN.loss, mainDQN.optimizer],
                                       feed_dict={mainDQN.X: state_batch, mainDQN.Y: y_batch})


    return loss_value

def update_from_memory2(mainDQN, targetDQN, TMPMOMORY, batch_size):

    state_batch = np.ndarray(shape=[batch_size, mainDQN.input_size])
    y_batch = np.ndarray(shape=[batch_size, mainDQN.output_size])

    minibatch = random.sample(TMPMOMORY, batch_size)
    i = 0
    for sample in minibatch:
        state, action, reward, new_state, done = sample         # unpacking

        one_hot_action = np.zeros(mainDQN.output_size) 

        if done:
            one_hot_action[action] = reward
        else:
            one_hot_action[action] = reward + discount_factor * np.max(targetDQN.predict(new_state))

        state_batch[i] = np.reshape(state, newshape=[1, 80*80*5])
        y_batch[i] = one_hot_action
        i += 1

    loss_value, _ = mainDQN.session.run([mainDQN.loss, mainDQN.optimizer],
                                       feed_dict={mainDQN.X: state_batch, mainDQN.Y: y_batch})

    return loss_value


def bot_play(DQN, env):

    state_memory = np.ndarray(shape=[5, 80 * 80])
    new_state_memory = np.ndarray(shape=[5, 80 * 80])


    state = env.reset()
    state_memory[0] = Preprocessing_Image(state)
    state_memory[1] = Preprocessing_Image(state)
    state_memory[2] = Preprocessing_Image(state)
    state_memory[3] = Preprocessing_Image(state)
    new_state, reward, done, info = env.step(1)  # FIRE
    state_memory[4] = Preprocessing_Image(new_state)


    # INPUT MEMORY에 4개를 채운 후...

    reward_sum = 0
    done = False
    step = 0
    action = 0
    realaction = 1
    before_action = -1
    no_op_count = 0
    lives = 5
    while not done:
        env.render()

        Qvalue = DQN.predict(state_memory)
        print(Qvalue)
        action = np.argmax(Qvalue)

        if (action == before_action):
            no_op_count += 1
        else:
            no_op_count = 0
            before_action = action

        if (no_op_count >= 30):
            action = 1
            before_action = 1
            no_op_count = 0

        print("Play Bot Action : {0}".format(action))

        new_state, reward, done, info = env.step(action)

        new_state_memory = np.copy(state_memory)
        new_state_memory[0] = state_memory[1]
        new_state_memory[1] = state_memory[2]
        new_state_memory[2] = state_memory[3]
        new_state_memory[3] = state_memory[4]
        new_state_memory[4] = Preprocessing_Image(new_state)

        reward_sum += reward
        step += 1
        state_memory = np.copy(new_state_memory)

    return reward_sum

def draw_error_values():
    fig = plt.figure(figsize=(20, 5))
    plt.ion()
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(episode_list[0:], train_error_list[0:], 'r', label='Train Error Values')
    ax2.plot(episode_list[0:], actions_list[0:], 'b', label='Number of Actions')
    ax1.set_ylabel('Train Error Values', color='r')
    ax2.set_ylabel('Number of Actions', color='b')
    ax1.set_xlabel('Episodes')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
    input("Press Enter to close the trained error figure...")
    plt.close(fig)

def get_copy_var_ops(src_scope_name='main', target_scope_name='target'):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)

    for src_var, target_var in zip(src_vars, target_vars):
        op_holder.append(target_var.assign(src_var.value()))

    return op_holder

def saveModel(session, src_scope_name='main', path='./breakout.ckpt'):
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)

    tf.train.Saver(src_vars).save(session, path)
    print("Model saved successfully!")

def Preprocessing_Image(rgb):
    # RGB를 Black & White 로 바꾸고
    # 이미지 사이즈를 105x80 으로 축소 후 80x80으로 잘라서 보냄
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.where(gray > 0 , 255, 0)
    tmp = gray[::2, ::2]
    output = tmp[25:, :]
    return np.reshape(output, newshape=[1, 80*80])

def restoreModel(session, path='./DeepMind_complete5'):

    tf.train.Saver().restore(sess=session, save_path=path)
    print("Model restored successfully.")




if __name__ == "__main__":

    BATCH_SIZE = 32
    env = gym.make('Breakout-v0')
    env.reset()
    input_size = 80 * 80 * 5
    output_size = env.action_space.n

    with tf.Session() as sess:
        mainDQN = CNNDQN(sess, input_size, output_size, name='main')
        targetDQN = CNNDQN(sess, input_size, output_size, name='target')

        # 변수 초기화
        init = tf.global_variables_initializer()
        state_memory = np.ndarray(shape=[5, 80*80])
        new_state_memory = np.ndarray(shape=[5, 80 * 80])

        sess.run(init)

        copy_ops = get_copy_var_ops(src_scope_name='main', target_scope_name='target')
        sess.run(copy_ops)
        epsilon = INITIAL_EPSILON

        saver = tf.train.Saver()
        before = 20
        #restoreModel(sess, "./DeepMind_complete4")
        total_count = 0
        start = 0

        for i in range(3):
            tmpreward = bot_play(mainDQN, env)


        for episode in range(start, max_episodes):
            rAll = 0
            done = False
            if(episode > 100):
                if(epsilon > 0.1):
                    epsilon *= 0.995
                else:
                    epsilon = 0.1
            state = env.reset()
            state_memory[0] = Preprocessing_Image(state)

            state_memory[1] = Preprocessing_Image(state)
            state_memory[2] = Preprocessing_Image(state)
            state_memory[3] = Preprocessing_Image(state)
            new_state, reward, done, info = env.step(1)  # FIRE
            state_memory[4] = Preprocessing_Image(new_state)

            #INPUT MEMORY에 5개를 채운 후...

            action = 0
            realaction = 1
            lives = 5
            no_op_count = 0;
            before_action = -1
            while not done:

                if(len(REPLAY_MEMORY) < 10000):
                    action = env.action_space.sample()
                    # 0 : noop, 1 : fire    2: left     3 : right     4 : leftfire    5: rightfire
                else:
                    action = mainDQN.egreedy_action(epsilon, env, state_memory)

                #print("Action : {0}".format(action))

                if(action == before_action):
                    no_op_count += 1
                else:
                    no_op_count = 0
                    before_action = action

                if(no_op_count >= 30) :
                    action = 1
                    before_action = -1
                    no_op_count = 0

                new_state, reward, done, info = env.step(action)

                new_state_memory = np.copy(state_memory)
                new_state_memory[0] = state_memory[1]
                new_state_memory[1] = state_memory[2]
                new_state_memory[2] = state_memory[3]
                new_state_memory[3] = state_memory[4]
                new_state_memory[4] = Preprocessing_Image(new_state)

                rAll += reward


                TMP_MEMORY.append((state_memory, action, 0, new_state_memory, done))

                if (reward >= 1):
                    if (len(TMP_MEMORY) < before):
                        tmpstate_memory, tmpaction, prevreward, tmpnew_state_memory, tmpdone = TMP_MEMORY[0]
                        TMP_MEMORY[0] = ((tmpstate_memory, tmpaction, 1, tmpnew_state_memory, tmpdone))
                    else:
                        tmpstate_memory, tmpaction, prevreward, tmpnew_state_memory, tmpdone = TMP_MEMORY[len(TMP_MEMORY) - before]
                        TMP_MEMORY[len(TMP_MEMORY) - before] = ((tmpstate_memory, tmpaction, 1, tmpnew_state_memory, tmpdone))


                # REPLAY_MEMORY 크기가 BATCH_SIZE 보다 크고 10000개 이상일 때 매회 10프레임마다 학습
                if len(REPLAY_MEMORY) > 10000 and total_count % (before/2):
                    mean_loss_value = update_from_memory(mainDQN, targetDQN, BATCH_SIZE)

                total_count += 1
                state_memory = np.copy(new_state_memory)

                if total_count % COPY_PERIOD == 0:
                    a = sess.run(copy_ops)
                    print("ToalCount : {0}, Target NN Copied".format(total_count))
                    print("ToalCount : {0}, Saved NN Copied".format(total_count))
                    saver.save(sess, "./DeepMind_complete5")
            
            if(rAll > 3):
                k = 0
                for k in range(len(TMP_MEMORY)):
                    REPLAY_MEMORY.append(TMP_MEMORY[k])

            if len(REPLAY_MEMORY) > 80000:
                REPLAY_MEMORY.popleft()

            if (rAll >= Max_reward):
                mean_loss_value = update_from_memory2(mainDQN, targetDQN, TMP_MEMORY, len(TMP_MEMORY))
                a = sess.run(copy_ops)
                Max_reward = rAll

            TMP_MEMORY.clear()

            print("ToalCount : {0}, NowEpisode : {1}, NowReward : {2},  Max_reward {3},  ".format(total_count, episode, rAll, Max_reward))
            if len(REPLAY_MEMORY) > BATCH_SIZE:
                episode_list.append(episode)
                train_error_list.append(mean_loss_value)
                actions_list.append(rAll)

            if episode > 0 and episode % TEST_PERIOD == 0:
                total_reward = 0
                for i in range(5):
                    tmpreward = bot_play(mainDQN, env)
                    total_reward += tmpreward
                    if (tmpreward > Max_reward):
                        Max_reward = tmpreward

                ave_reward = total_reward / 5
                print("TEST episode: {0}, Epsilon: {1}, Evaluation Average Reward: {2}, meanLoss : {3}".format(episode, epsilon,
                                                                                               ave_reward, mean_loss_value))
                if ave_reward >= 50:
                    break

        #saveModel(sess, src_scope_name='main', path='./breakout.ckpt')
        saver.save(sess, "./DeepMind_complete5")
        env.reset()
        env.close()
        draw_error_values()

        input("Press Enter to make the trained bot play...")
        bot_play(mainDQN, env)