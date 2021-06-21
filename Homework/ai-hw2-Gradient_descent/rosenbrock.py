import numpy as np
from sympy import *
import time

Your_code = 'TBD'
x, y = symbols('x y')
f = np.square(1 - x) + 100 * np.square(y - np.square(x))

fdx = diff(f, x)
fdy = diff(f, y)

def cal_f(x, y):
    # 计算Rosenbrock函数值
    res = np.square(1 - x) + 100 * np.square(y - np.square(x))
    return res

def cal_dx(xn, yn):
    # 计算x偏导数
    fx = fdx.subs([(x, xn), (y, yn)])
    # f1 = -2 * (1 - xn) + 400 * xn * (yn - xn ** 2)
    return fx

def cal_dy(xn, yn):
    # 计算y偏导数
    fy = fdy.subs([(x, xn), (y, yn)])
    # f1 = 200 * (yn - xn ** 2)
    return fy

# 小心这里的迭代次数和步长，如果太长你可能会因为跑得太快错过顶点
def train_grad(max_iter=100000, step=0.001):  # 梯度下降迭代主函数

    w = np.zeros((2,), dtype=np.float32)  # 初始化求解数值从([0, 0])开始
    loss = 20  # 设置函数初值为20
    iter_count = 0

    while loss > 0.001 and iter_count < max_iter:
        err = np.zeros((2,), dtype=np.float32)

        # 计算x偏导数
        err[0] = cal_dx(w[0], w[1])

        # 计算y偏导数
        err[1] = cal_dy(w[0], w[1])

        for j in range(2):
            w[j] -= step * err[j]  # 梯度下降迭代

        loss = cal_f(w[0], w[1]) - 0  # 最小值为0

        # print('err' + str(err))
        # print('w' + str(w))
        print("iter_count: ", iter_count, "the loss:", cal_f(w[0], w[1]))  # 每次迭代输出迭代序号和当前函数值
        # print('---------------')
        iter_count += 1

    return w


if __name__ == '__main__':  # main主程序
    btime = time.mktime(time.localtime())

    # 调用梯度下降迭代主函数
    w = train_grad(100000, 0.001)
    # 显示w的值
    print(w)

    etime = time.mktime(time.localtime())
    print(etime - btime)