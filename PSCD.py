from mxnet.gluon import data as gdata
import sys
import time
from mxnet import nd
import numpy as np
from DDHIP import *
from FM_Sketch import FM_Sketch


mnist_train = gdata.vision.FashionMNIST(train=False)
mnist_test = gdata.vision.FashionMNIST(train=False)

batch_size = 50
transformer = gdata.vision.transforms.ToTensor()

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=False)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False)


num_inputs = 784
num_hiddens = 10
num_outputs = 10

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
params = [W1, W2]


def sigmoid(X):
    return 1 / (1 + nd.exp(-X))


def sigmoid_prime(y):
    return sigmoid(y) * (1 - sigmoid(y))


class net:
    def __init__(self, params, batch_size, lr):
        self.W1 = params[0]
        self.W2 = params[1]
        self.batch_size = batch_size
        self.lr = lr
        setup = DDHIP_Setup(l=num_inputs)
        self.mpk, self.msk = setup.setup()
        self.dropout = 0.5


    def forward(self, X):
        X = X.reshape((-1, num_inputs))
        mask = nd.random.uniform(0, 1, shape=X.shape) > self.dropout

        w1 = X * (mask / (1 - self.dropout))  #

        w2 = self.W1
        res = nd.zeros(shape=(w1.shape[0], w2.shape[1]))
        for i in range(w1.shape[0]):
            for j in range(w2.shape[1]):
                if mask[i, j]:  #
                    encrypt = DDHIP_Encrypt(w1[i, :], self.mpk, self.msk)
                    ct = encrypt.encrypt()
                    decrypt = DDHIP_Decrypt(w2[:, j], self.msk, ct)
                    res[i, j] = decrypt.decrypt()
                else:
                    res[i, j] = 0  #
        Z = res

        self.h = sigmoid(Z)

        o = nd.dot(self.h, self.W2)
        return self.softmax2(o)

    def evaluate_accuracy(self, data_iter, net):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
            n += y.size
        return acc_sum / n

    def softmax2(self, X):
        row_max = X.max(axis=1)
        row_max = row_max.reshape(-1, 1)
        X = X - row_max
        X_exp = X.exp()
        partition = X_exp.sum(axis=1, keepdims=True)
        return X_exp / partition

    def cross_entropy(self, y_hat, y):
        return -nd.pick(y_hat, y).log()

    def backword(self, dLdo, X):
        self.W2_grad = nd.dot(self.h.T, dLdo)
        dLdh = nd.dot(dLdo, self.W2.T)
        dLdz = dLdh * sigmoid_prime(self.h)
        X = X.reshape((-1, num_inputs))
        self.W1_grad = nd.dot(X.T, dLdz)

    def sgd(self, batch_size):
        self.W2 -= self.lr * (self.W2_grad / self.batch_size)
        self.W1 -= self.lr * (self.W1_grad / self.batch_size)

    def train(self, train_iter, test_iter, num_epochs, batch_size):
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in train_iter:
                start2 = time.time()
                y_hat = self.forward(X)
                l = self.cross_entropy(y_hat, y).sum()
                y_eye = nd.eye(10)[y]
                self.backword(y_hat - y_eye, X)
                self.sgd(batch_size)
                end2 = time.time()
                print('time cost', end2 - start2)
                y = y.astype('float32')
                train_l_sum += l.asscalar()
                train_acc_sum += (self.forward(X).argmax(axis=1) == y).sum().asscalar()
                n += y.size
                break
            test_acc = self.evaluate_accuracy(test_iter, self.forward)
            print(
                f'epoch {epoch + 1}, loss {round(train_l_sum / n, 4)}, train_accuracy {round(train_acc_sum / n, 4)}, test_accuracy {test_acc}, time {round(time.time() - start, 4)}')


num_epochs, lr = 20, 0.1
net = net(params, batch_size, lr)
net.train(train_iter, test_iter, num_epochs, batch_size)





###################################################################KDDCUP
# import time
# import numpy as np
# import mxnet
# import pandas as pd
# from mxnet import nd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from mxnet.gluon.data import DataLoader, ArrayDataset
# from DDHIP import *
# from FM_Sketch import FM_Sketch
#
# batch_size = 160
#
# # 读取数据集
# data = pd.read_csv(r'C:\Users\ThinkPad\Desktop\kddcup.data_10_percent.gz', header=None)
#
# # 对分类变量进行编码
#
# for i in range(1, 4):
#     le = LabelEncoder()
#     data[i] = le.fit_transform(data[i])
#
# # 将数据集分为特征和标签
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values
#
# # 将标签进行编码
# le = LabelEncoder()
# y = le.fit_transform(y)
#
# # 划分训练集和测试集，并使用stratify参数确保类别分布相似
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # 定义数据集和数据加载器
# train_dataset = ArrayDataset(nd.array(X_train), nd.array(y_train))
# train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = ArrayDataset(nd.array(X_test), nd.array(y_test))
# test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # 定义模型参数a=784, c=10, b=256  三层MLP
# num_inputs, num_hiddens, num_outputs = 41, 23, 23
# # 随机初始化w和b
# w_h = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
# w_o = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
# b_h = nd.zeros(num_hiddens)
# b_o = nd.zeros(num_outputs)
# params = [w_h, w_o, b_h, b_o]
#
#
# # 定义激活函数和导数
# def sigmoid(X):
#     return 1 / (1 + nd.exp(-X))
#
#
# def sigmoid_prime(y):
#     return sigmoid(y) * (1 - sigmoid(y))
#
#
# class Net():
#     def __init__(self, params, batch_size, lr):
#         self.W1 = params[0]
#         self.W2 = params[1]
#         self.b1 = params[2]
#         self.b2 = params[3]
#         self.batch_size = batch_size
#         setup = DDHIP_Setup(l=num_inputs)
#         self.mpk, self.msk = setup.setup()
#         self.lr = lr
#         self.dropout = 0.5
#
#     def softmax(self, X):
#         X_exp = X.exp()
#         partition = X_exp.sum(axis=1, keepdims=True)
#         return X_exp / partition
#
#     # 定义交叉熵，即损失函数
#     def loss(self, y_hat, y):
#         return -nd.pick(y_hat, y).log()
#
#
#     def forward(self, X):
#         X = X.reshape((-1, num_inputs))
#         mask = nd.random.uniform(0, 1, shape=X.shape) > self.dropout
#
#         w1 = X * (mask / (1 - self.dropout))
#
#         w2 = self.W1
#         res = nd.zeros(shape=(w1.shape[0], w2.shape[1]))
#         for i in range(w1.shape[0]):
#             for j in range(w2.shape[1]):
#                 if mask[i, j]:  #
#                     encrypt = DDHIP_Encrypt(w1[i, :], self.mpk, self.msk)
#                     ct = encrypt.encrypt()
#                     decrypt = DDHIP_Decrypt(w2[:, j], self.msk, ct)
#                     res[i, j] = decrypt.decrypt()
#                 else:
#                     res[i, j] = 0  #
#         Z = res
#
#         self.h = sigmoid(Z) + self.b1
#
#         o = nd.dot(self.h, self.W2)+ self.b2
#         return self.softmax(o)
#
#     # def forward(self, X):
#     #     M = nd.dot(X, self.W1) + self.b1
#     #     self.h = sigmoid(M)
#     #     Z = nd.dot(self.h, self.W2) + self.b2
#     #     O = self.softmax(Z)
#     #     return O
#
#     def backward(self, dLdz, X):
#         self.W2_grad = nd.dot(self.h.T, dLdz)
#         dLdh = nd.dot(dLdz, self.W2.T)
#         dLdM = dLdh * sigmoid_prime(self.h)
#         self.W1_grad = nd.dot(X.T, dLdM)
#         self.b2_grad = dLdz.mean(axis=0)
#         self.b1_grad = dLdM.mean(axis=0)
#
#     def sgd(self):
#         # 梯度裁剪
#         max_grad = 1
#         self.W2_grad = nd.clip(self.W2_grad, -max_grad, max_grad)
#         self.W1_grad = nd.clip(self.W1_grad, -max_grad, max_grad)
#         self.b2_grad = nd.clip(self.b2_grad, -max_grad, max_grad)
#         self.b1_grad = nd.clip(self.b1_grad, -max_grad, max_grad)
#
#         self.W2 -= self.lr * (self.W2_grad / self.batch_size)
#         self.W1 -= self.lr * (self.W1_grad / self.batch_size)
#         self.b2 -= self.lr * (self.b2_grad / self.batch_size)
#         self.b1 -= self.lr * (self.b1_grad / self.batch_size)
#
#     def evaluate_accuracy(self, data_iter):
#         acc_sum, n = 0.0, 0
#         for X, y in data_iter:
#             y = y.astype('float32')
#             acc_sum += (self.forward(X).argmax(axis=1) == y).sum().asscalar()
#             n += y.size
#         return acc_sum / n
#
#     def train(self, train_iter, test_iter, num_epochs):
#         for epoch in range(num_epochs):
#             train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
#             for i, (X, y) in enumerate(train_iter):
#                 start2 = time.time()
#                 y = y.astype('float32')
#                 y_hat = self.forward(X)
#                 l = self.loss(y_hat, y).sum()
#                 y_eye = nd.one_hot(y, num_outputs)
#                 self.backward(y_hat - y_eye, X)
#                 self.sgd()
#                 end2 = time.time()
#                 print('每一批次时间', end2 - start2)
#                 train_l_sum += l.asscalar()
#                 train_acc_sum += (self.forward(X).argmax(axis=1) == y).sum().asscalar()
#                 n += y.size
#             test_acc = self.evaluate_accuracy(test_iter)
#             print(
#                 f'epoch:{epoch} loss:{round(train_l_sum / n, 4)} train_accuracy:{round(train_acc_sum / n, 4)} test_accuracy:{test_acc} time:{round(time.time() - start, 4)}')
#
#
# num_epochs, lr = 20, 0.01
# net_model = Net(params, batch_size, lr)
# net_model.train(train_iter, test_iter, num_epochs)










# import mxnet as mx ########USPS
# from mxnet import nd
# from mxnet import gluon, autograd, np, npx
# from mxnet.gluon.data.vision import transforms
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from mxnet.gluon.data import DataLoader, ArrayDataset
# import time
# from sklearn.datasets import fetch_openml
# from FM_Sketch import FM_Sketch
# # 获取数据集
# usps = fetch_openml('usps', version=2)
# X = usps.data.astype(np.float32)
# y = usps.target.astype(np.float32)
#
# # 数据预处理（示例：标准化）
# X_mean = X.mean(axis=0)
# X_std = X.std(axis=0)
# X = (X - X_mean) / X_std
#
# # 小批量大小
# batch_size = 30
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # 定义数据集和数据加载器
# train_dataset = ArrayDataset(nd.array(X_train), nd.array(y_train))
# train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = ArrayDataset(nd.array(X_test), nd.array(y_test))
# test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # 定义模型参数a=784, c=10, b=256
# num_inputs, num_hiddens, num_outputs = 256, 200, 10
# # 随机初始化w和b
# w_h = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
# w_o = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
# b_h = nd.zeros(num_hiddens)
# b_o = nd.zeros(num_outputs)
# params = [w_h, w_o, b_h, b_o]
#
# # 定义激活函数和导数
# def sigmoid(X):
#     return 1 / (1 + nd.exp(-X))
#
#
# def sigmoid_prime(y):
#     return sigmoid(y) * (1 - sigmoid(y))
#
#
# class Net():
#     def __init__(self, params, batch_size, lr):
#         self.W1 = params[0]
#         self.W2 = params[1]
#         self.b1 = params[2]
#         self.b2 = params[3]
#         self.batch_size = batch_size
#         self.lr = lr
#
#     def softmax(self, X):
#         X_exp = X.exp()
#         partition = X_exp.sum(axis=1, keepdims=True)
#         return X_exp / partition
#
#     # 定义交叉熵，即损失函数
#     def loss(self, y_hat, y):
#         return -nd.pick(y_hat, y).log()
#
#     def forward(self, X):
#         M = nd.dot(X, self.W1) + self.b1
#         self.h = sigmoid(M)
#         Z = nd.dot(self.h, self.W2) + self.b2
#         O = self.softmax(Z)
#         return O
#
#     def backward(self, dLdz, X):
#         self.W2_grad = nd.dot(self.h.T, dLdz)
#         dLdh = nd.dot(dLdz, self.W2.T)
#         dLdM = dLdh * sigmoid_prime(self.h)
#         self.W1_grad = nd.dot(X.T, dLdM)
#         self.b2_grad = dLdz.mean(axis=0)
#         self.b1_grad = dLdM.mean(axis=0)
#
#     def sgd(self):
#         # 梯度裁剪
#         max_grad = 1
#         self.W2_grad = nd.clip(self.W2_grad, -max_grad, max_grad)
#         self.W1_grad = nd.clip(self.W1_grad, -max_grad, max_grad)
#         self.b2_grad = nd.clip(self.b2_grad, -max_grad, max_grad)
#         self.b1_grad = nd.clip(self.b1_grad, -max_grad, max_grad)
#
#         self.W2 -= self.lr * (self.W2_grad / self.batch_size)
#         self.W1 -= self.lr * (self.W1_grad / self.batch_size)
#         self.b2 -= self.lr * (self.b2_grad / self.batch_size)
#         self.b1 -= self.lr * (self.b1_grad / self.batch_size)
#
#     def evaluate_accuracy(self, data_iter):
#         acc_sum, n = 0.0, 0
#         for X, y in data_iter:
#             y = y.astype('float32')
#             acc_sum += (self.forward(X).argmax(axis=1) == y).sum().asscalar()
#             n += y.size
#         return acc_sum / n
#
#     def train(self, train_iter, test_iter, num_epochs):
#         for epoch in range(num_epochs):
#             train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
#             for i, (X, y) in enumerate(train_iter):
#                 y = y.astype('float32')
#                 y_hat = self.forward(X)
#                 l = self.loss(y_hat, y).sum()
#                 y_eye = nd.one_hot(y, num_outputs)
#                 self.backward(y_hat - y_eye, X)
#                 self.sgd()
#                 train_l_sum += l.asscalar()
#                 train_acc_sum += (self.forward(X).argmax(axis=1) == y).sum().asscalar()
#                 n += y.size
#             test_acc = self.evaluate_accuracy(test_iter)
#             print(
#                 f'epoch:{epoch} loss:{round(train_l_sum / n, 4)} train_accuracy:{round(train_acc_sum / n, 4)} test_accuracy:{test_acc} time:{round(time.time() - start, 4)}')
#
#
# num_epochs, lr = 20, 0.01
# net_model = Net(params, batch_size, lr)
# net_model.train(train_iter, test_iter, num_epochs)



