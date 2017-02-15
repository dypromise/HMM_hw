#!usr/bin/python#coding=utf-8
import numpy as np
import random
import numpy.random as nr
import logspace as ls
import scipy.misc as sm
import copy

T_true = np.array([[0.8, 0.2, 0.0],
                   [0.1, 0.7, 0.2],
                   [0.1, 0.0, 0.9]])
P_head = np.array([0.1, 0.5, 0.9])
theta_true = np.array([1 - P_head, P_head])
pi_true = np.array([1, 0, 0], dtype='float')


def sampling(N):
    """

    :param N:
    :return: 采样序列以及其状态序列
    """
    sampling_list = []
    state_list = []
    current_state = 0

    for k in range(N):
        state_list.append(current_state)
        random_number = random.random()
        p_head_current = P_head[current_state]
        if (random_number <= p_head_current):
            sampling_list.append(1)
        else:
            sampling_list.append(0)

        random_number = random.random()
        trans_prob = T_true[current_state]
        if (random_number > trans_prob[0]):
            if (random_number <= trans_prob[:-1].sum()):
                current_state = 1
            else:
                current_state = 2
        else:
            current_state = 0
    return np.array(sampling_list, dtype='int'), np.array(state_list, dtype='int')


def MLE(X, Inference_method=1):
    """
    最大似然估计，EM框架；
    运行结束参数估计完毕
    :param X:
    :return:
    """
    if (Inference_method==1):
        func = get_gama_epsilon_list
    elif(Inference_method==2):
        func = get_gama_epsilon_list_Gibbssampling
    elif(Inference_method==3):
        func = get_gama_epsilon_list_Gibbssampling_multichain


    MAX_ITER = 500
    T = nr.random((3, 3))
    T = T / T.sum(1).reshape(-1, 1)#因为sum后是一个行向量，不论是sum(0)还是sum(1)
    pi = nr.random((3, 1))
    pi = pi / pi.sum(0)
    theta_1 = nr.random(3)  # 发射0概率
    theta_2 = 1 - theta_1  # 发射1概率
    theta = np.array([theta_1, theta_2])

    N = np.size(X)  # N序列长度
    # EM算法
    for iter in range(MAX_ITER):
        # E_step
        gama_list, epsilon_list, p_x = func(X, N, T, theta, pi)

        # M_step
        pi = gama_list[0] / (gama_list[0].sum())
        T = epsilon_list[1:].sum(0)
        T = T / T.sum(1).reshape(-1, 1)
        tmp = np.zeros((1, 3))
        for i in range(N):
            if (X[i] == 1):
                tmp = tmp + gama_list[i]
        theta_2 = tmp / gama_list.sum(0)
        theta_1 = 1 - theta_2
        theta = np.array([theta_1, theta_2])
        #print theta_2
        print iter
        print (((T-T_true)**2).sum().sum())
        print T


def get_gama_epsilon_list(X, N, T, theta, pi):
    """
    前向后向算法获得alpha，beta序列，然后计算gama，epsilon序列
    :param X:
    :param pi:
    :param T:
    :param theta:
    :return:计算MLE中E-step中的gama，epsilon
    """
    log_pi = np.log(pi)
    log_T = np.log(T)
    log_theta = np.log(theta)

    alpha_list = np.zeros((N, 3))
    beta_list = np.zeros((N, 3))
    gama_list = np.zeros((N, 3))
    epsilon_list = np.zeros((N, 3, 3))

    alpha_list[0] = ls.log_multiply(log_pi.reshape(1, -1), log_theta[X[0]])  # log_pi是列向量，必须横过来
    beta_list[N - 1] = np.zeros(3)  # 因为在log空间
    for i in range(1, N):
        alpha_list[i] = ls.log_multiply(ls.log_vec_mat_dot(alpha_list[i - 1], log_T), log_theta[X[i]])
        beta_list[N - 1 - i] = \
            ls.log_mat_vec_dot(log_T, ls.log_multiply(beta_list[N - i], log_theta[X[N - i]]).reshape(-1, 1)).transpose()
    p_x = ls.log_sum(alpha_list[-1])

    # 下面求gama，epsilon是在正常空间
    for i in range(N):
        gama_list[i] = np.exp(ls.log_multiply(alpha_list[i], beta_list[i]) - p_x)
        if (i == 0): continue
        epsilon_list[i] = np.exp(
            ls.log_a_T_b(alpha_list[i - 1].reshape(-1, 1), log_T, ls.log_multiply(log_theta[X[i]], beta_list[i])) - p_x)
    return gama_list, epsilon_list, p_x



def get_gama_epsilon_list_Gibbssampling(X, N, T, theta, pi, iterations=2000):
    """
    用Gibbs采样的方法推断（计算）gama以及epsilon
    :param X:
    :param N:
    :param T:
    :param theta:
    :param iterations:
    :return:
    """
    Y_mat = []
    Y = nr.randint(0,3,N)  # 初始化一个Y序列

    for i in range(iterations):
        p_yt = (theta[X[N - 1]] * T[Y[N - 1 - 1]]).reshape(3)
        p_yt /= p_yt.sum()
        randnum = nr.random()
        if (randnum < p_yt[0]):
            Y[N - 1] = 0
        elif (randnum < 1 - p_yt[-1]):
            Y[N - 1] = 1
        else:
            Y[N - 1] = 2
        for t in range(N - 2, 0, -1):
            p_yt = (theta[X[t]] * T[Y[t - 1]] * T[:, Y[t+1]].reshape(1, -1)).reshape(3)
            p_yt /= p_yt.sum()
            randnum = nr.random()
            if (randnum < p_yt[0]):
                Y[t] = 0
            elif (randnum < 1 - p_yt[-1]):
                Y[t] = 1
            else:
                Y[t] = 2
        p_yt = (theta[X[0]] * T[:, Y[1]]).reshape(3)
        p_yt /= p_yt.sum()
        if (randnum < p_yt[0]):
            Y[0] = 0
        elif (randnum < 1 - p_yt[-1]):
            Y[0] = 1
        else:
            Y[0] = 2
        if (i >= 1500):
            #注意，append（obj）需要深拷贝
            Y_mat.append(copy.deepcopy(Y))
    # 下面利用多个Y序列统计gama，epsilon
    Y_mat = np.array(Y_mat)
    #下面的count做了加常数平滑，因为采样次数少的情况下，有些位置会出现全零，归一化会出错。
    state_count = np.zeros((N, 3), dtype='float') + 0.001
    trans_count = np.zeros((N, 3, 3), dtype='float') + 0.001
    for i in range(np.shape(Y_mat)[0]):
        Y = Y_mat[i]
        state_count[0, Y[0]] += 1
        for j in range(1, N):
            state_count[j, Y[j]] += 1
            trans_count[j, Y[j - 1], Y[j]] += 1
    gama_list = state_count / (state_count.sum(1)).reshape(-1, 1)
    epsilon_list = trans_count / trans_count.sum(2).reshape((N, 3, 1))

    return gama_list, epsilon_list, 0
def get_gama_epsilon_list_Gibbssampling_multichain(X, N, T, theta, pi, iterations=200, chainNum  = 10):
    """
    用Gibbs采样的方法推断（计算）gama以及epsilon
    :param X:
    :param N:
    :param T:
    :param theta:
    :param iterations:
    :return:
    """
    Y_mat = []
    Ys = nr.randint(0,3,(chainNum,N))  # 初始化一个Y序列

    for i in range(iterations):
        for j in range(chainNum):

            p_yt = (theta[X[N - 1]] * T[Ys[j,N - 1 - 1]]).reshape(3)
            p_yt /= p_yt.sum()
            randnum = nr.random()
            if (randnum < p_yt[0]):
                Ys[j,N - 1] = 0
            elif (randnum < 1 - p_yt[-1]):
                Ys[j,N - 1] = 1
            else:
                Ys[j,N - 1] = 2
            for t in range(N - 2, 0, -1):
                p_yt = (theta[X[t]] * T[Ys[j,t - 1]] * T[:, Ys[j,t+1]].reshape(1, -1)).reshape(3)
                p_yt /= p_yt.sum()
                randnum = nr.random()
                if (randnum < p_yt[0]):
                    Ys[j,t] = 0
                elif (randnum < 1 - p_yt[-1]):
                    Ys[j,t] = 1
                else:
                    Ys[j,t] = 2
            p_yt = (theta[X[0]] * T[:, Ys[j,1]]).reshape(3)
            p_yt /= p_yt.sum()
            if (randnum < p_yt[0]):
                Ys[j,0] = 0
            elif (randnum < 1 - p_yt[-1]):
                Ys[j,0] = 1
            else:
                Ys[j,0] = 2
        if (i >= 100):
            #注意，append（obj）需要深拷贝
            Y = Ys.sum(0)/chainNum
            Y_mat.append(copy.deepcopy(Y))
        # 下面利用多个Y序列统计gama，epsilon
    Y_mat = np.array(Y_mat)
    #下面的count做了加常数平滑，因为采样次数少的情况下，有些位置会出现全零，归一化会出错。
    state_count = np.zeros((N, 3), dtype='float') + 0.001
    trans_count = np.zeros((N, 3, 3), dtype='float') + 0.001
    for i in range(np.shape(Y_mat)[0]):
        Y = Y_mat[i]
        state_count[0, Y[0]] += 1
        for j in range(1, N):
            state_count[j, Y[j]] += 1
            trans_count[j, Y[j - 1], Y[j]] += 1
    gama_list = state_count / (state_count.sum(1)).reshape(-1, 1)
    epsilon_list = trans_count / trans_count.sum(2).reshape((N, 3, 1))

    return gama_list, epsilon_list, 0


def Vitebi(X, pi, T, theta):
    """

    :param X:
    :param pi:
    :param T:
    :param theta:
    :return: 从N到1的状态序列
    """
    N = np.shape(X)[0]
    alpha_list = np.zeros((N, 3))
    max_parent_list = np.zeros((N, 3))
    alpha_list[0] = ls.log_multiply(pi.reshape(1, -1), theta[X[0]])  # pi是列向量，必须横过来
    for i in range(1, N):
        tmp, max_parent_list[i] = ls.log_vec_mat_max(alpha_list[i - 1], T)
        alpha_list[i] = ls.log_multiply(tmp, theta[X[i]])
    p_max = alpha_list[-1].max()
    max_state = list(alpha_list[-1]).index(p_max)
    state_list = np.zeros(N, dtype='int')
    state_list[N - 1] = max_state
    for i in range(N - 1, 0, -1):
        max_state = max_parent_list[i][max_state]
        state_list[i - 1] = max_state
    return state_list


def match_rate(state_list_true, state_list_max, N):
    count = 0
    for i in range(N):
        if (state_list_max[i] == state_list_true[i]):
            count += 1
    return float(count) / N


N = 1000
samples, state_list_true = sampling(N)
print samples
MLE(samples,2)
print "end"
"""
state_list_max = Vitebi(samples, pi_true, T_true, theta_true)
print state_list_max
print state_list_true
matchrate = match_rate(state_list_true,state_list_max,N)
print matchrate
"""
