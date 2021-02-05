'''
基于PyBrain实现标准BP和累计BP两种算法下的单层神经网络
标准BP类比于随机梯度下降，Ek = 1/2(sigma(yk_j^-yk_j)^2)
累计BP类比于梯度下降，E = 1/m(sigma(Ek))

'''
from chap4 import DataSet
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.close('all')

def preprocess(data):
    # 1.将非数映射数字
    for title in data.columns:
        if data[title].dtype == 'object':
            # 如果属性是离散值，data[title] 为 series
            encoder = LabelEncoder()
            data[title] = encoder.fit_transform(data[title])
    # 2.去均值和方差归一化
    ss = StandardScaler()
    X = data.drop('种类', axis=1)  # ndarray
    Y = data['种类']  # Series
    X = ss.fit_transform(X)
    x, y = np.array(X), np.array(Y).reshape(Y.shape[0],1)  # ndarray shape:(17,8)  (17,1)
    return x, y

def sigmoid(x):
    '''
    sigmoid函数
    :param x:
    :return:
    '''
    return 1/(1+np.exp(-x))
def d_sigmoid(x):
    '''
    sigmoid函数求导
    :param x:
    :return:
    '''
    return x*(1-x)

def accumulate_BP(x, y, dim=10, eta=0.8, max_iter=500):
    '''
    累计BP算法,E = 1/m(sigma(Ek))
    :param x: 训练数据
    :param y: 训练数据标签
    :param dim: 神经网络隐藏层维度
    :param eta: 学习率
    :param max_iter: 迭代次数
    :return:
    '''
    n_samples = x.shape[0]
    w1 = np.zeros((x.shape[1], dim))
    b1 = np.zeros((n_samples, dim))
    w2 = np.zeros((dim,1))
    b2 = np.zeros((n_samples,1))
    losslist = []
    for iter in range(max_iter):
        # 前向传播
        u1 = np.dot(x, w1)+b1
        out1 = sigmoid(u1)
        u2 = np.dot(out1, w2)+b2
        out2 = sigmoid(u2)
        loss = np.mean(np.square(y-out2))/2
        losslist.append(loss)
        print('iter:%d loss:%.4f'%(iter, loss))
        # 反向传播
        # 标准BP
        d_out2 = -(y-out2)
        d_u2 = d_out2*d_sigmoid(out2)
        d_w2 = np.dot(np.transpose(out1),d_u2)
        d_b2 = d_u2
        d_out1 = np.dot(d_u2, np.transpose(w2))
        d_u1 = d_out1*d_sigmoid(out1)
        d_w1 = np.dot(np.transpose(x),d_u1)
        d_b1 = d_u1
        # 更新
        w1 = w1 - eta*d_w1
        w2 = w2 - eta*d_w2
        b1 = b1 - eta*d_b1
        b2 = b2 - eta*d_b2
    # loss可视化
    plt.figure()
    plt.plot([i+1 for i in range(max_iter)], losslist)
    plt.legend(['accumulated BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    return w1, w2, b1, b2

def standared_BP(x, y, dim=10, eta=0.8, max_iter=500):
    '''
    标准BP类比于随机梯度下降，Ek = 1/2(sigma(yk_j^-yk_j)^2)
    :param x: 训练数据
    :param y: 训练数据标签
    :param dim: 隐藏层维度
    :param eta: 学习率
    :param max_iter: 迭代次数
    :return:
    '''
    n_samples = 1
    w1 = np.zeros((x.shape[1], dim))
    b1 = np.zeros((n_samples, dim))
    w2 = np.zeros((dim,1))
    b2 = np.zeros((n_samples, 1))
    losslist = []
    for iter in range(max_iter):
        loss_per_iter = []
        for m in range(x.shape[0]):
            xi, yi = x[m,:], y[m,:]
            xi, yi = xi.reshape(1, xi.shape[0]), yi.reshape(1, yi.shape[0])
            # 前向传播
            u1 = np.dot(xi, w1) + b1
            out1 = sigmoid(u1)
            u2 = np.dot(out1, w2) + b2
            out2 = sigmoid(u2)
            loss = np.square(yi - out2) / 2
            loss_per_iter.append(loss)
            print('iter:%d loss:%.4f' % (iter, loss))
            # 反向传播
            # 标准BP
            d_out2 = -(yi - out2)
            d_u2 = d_out2 * d_sigmoid(out2)
            d_w2 = np.dot(np.transpose(out1), d_u2)
            d_b2 = d_u2
            d_out1 = np.dot(d_u2, np.transpose(w2))
            d_u1 = d_out1 * d_sigmoid(out1)
            d_w1 = np.dot(np.transpose(xi), d_u1)
            d_b1 = d_u1
            # 更新
            w1 = w1 - eta * d_w1
            w2 = w2 - eta * d_w2
            b1 = b1 - eta * d_b1
            b2 = b2 - eta * d_b2
        losslist.append(np.mean(loss_per_iter))
    # loss可视化
    plt.figure()
    plt.plot([i + 1 for i in range(max_iter)], losslist)
    plt.legend(['accumulated BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    return w1, w2, b1, b2


if __name__ == '__main__':
    # 获取数据，转化为DataFrame
    data, title, labels_full = DataSet.watermelon()
    title.append('种类')  # ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率', '种类']
    data = pd.DataFrame(data, columns=title)
    x, y = preprocess(data)
    # print('x : ', x)
    # print('y : ', y)
    dim = 10
    # w1, w2, b1, b2 = accumulate_BP(x,y,dim)
    w1, w2, b1, b2 = standared_BP(x,y,dim)
    # 测试
    u1 = np.dot(x,w1)+b1
    out1 = sigmoid(u1)
    u2 = np.dot(out1, w2)+b2
    out2 = sigmoid(u2)
    y_pred = np.round(out2)
    res = pd.DataFrame(np.hstack((y,y_pred)),columns=['真值','预测'])
    # res.to_excel('accumulate_BP_result.xlsx', index = False)
    res.to_excel('standard_BP_result.xlsx', index = False)




