#!usr/bin/python#coding=utf-8
import numpy as np
import scipy.misc as sm


def log_sum(a):
    # max = a.max()
    # return sm.logsumexp(a - max) + max
    return sm.logsumexp(a)


def log_multiply(a, b):
    """

    :param a: a与b必须是同一种向量
    :param b:
    :return: 逐点积
    """
    return a + b


def log_vec_vec_dot(a, b):
    """

    :param a:必须是行向量
    :param b: 必须是列向量
    :return: 内积
    """
    c = a + b.reshape(1, -1)
    return log_sum(c)


def log_vec_mat_dot(a, mat):
    """

    :param a: 必须是行向量
    :param mat:
    :return:
    """
    [M, N] = mat.shape
    c = a.reshape(-1, 1) + mat
    d = np.zeros(N)
    for i in range(N):
        d[i] = log_sum(c[:, i])
    return np.matrix(d)


def log_vec_mat_max(a, mat):
    """

    :param a: 必须是行向量
    :param mat:
    :return:
    """
    [M, N] = mat.shape
    c = np.zeros(N)
    indexmax = np.zeros(N)
    for i in range(N):
        tmp = list(a + mat[:, i].reshape(M))#二维数组没法直接变list
        max = np.max(tmp)
        indexmax[i] = tmp.index(max)
        c[i] = max
    return np.matrix(c), np.matrix(indexmax)


def log_mat_vec_dot(mat, a):
    """

    :param mat:
    :param a: 必须是列向量
    :return:
    """
    [M, N] = mat.shape
    c = mat + a.reshape(1,-1)
    d = np.zeros((M, 1))
    for i in range(M):
        d[i, :] = log_sum(c[i, :])
    return np.matrix(d)


def log_a_T_b(a, T, b):
    """

    :param a: 必须是列向量
    :param T:
    :param b: 必须是行向量
    :return:
    """

    return a + T + b
