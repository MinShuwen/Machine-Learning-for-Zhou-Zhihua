# -*- coding:utf-8 -*-
'''
自己编写代码实现逻辑回归
'''
import matplotlib.pyplot as plt
import numpy as np

class Logistic():
    def __init__(self):
        self._history_w = []
        self._likelihood = []

    def load_input_data(self,data_file):
        with open(data_file) as f:
            input_x = []
            input_y = []
            for line in f:
                [id,x1,x2,y] = line.split()
                input_x.append([1.0,float(x1),float(x2)])
                input_y.append(int(y))
        self._input_x = np.array(input_x,dtype=np.float)
        self._input_y = np.array(input_y,dtype=np.float)

    def sigmoid(self,x,w):
        return 1.0/(1+np.exp(-np.inner(w,x)))

    def likelihood_function(self,w):
        # 极大似然估计
        tmp = np.inner(self._input_x,w)
        a = np.inner(tmp.T,self._input_y)
        b = np.sum(np.log(1+np.exp(tmp)))
        return b-a

    def batch_gardient_descent(self,iter_num,iter_rate):
        (data_num,features) = np.shape(self._input_x)
        w = np.ones(features)
        for i in range(iter_num):
            theta = self.sigmoid(self._input_x,w)
            delta = theta - self._input_y
            w = w - iter_rate*np.inner(self._input_x.T,delta)
            self._history_w.append(w)
            self._likelihood.append(self.likelihood_function(w))
        self._final_w = w
        return w

    def stochastic_gradient_descent(self,iter_num,iter_rate):
        (data_num,features) = np.shape(self._input_x)
        w = np.ones(features)
        for i in range(iter_num):
            for j in range(data_num):
                iter_rate = 4/(1.0+j+i)+0.01
                theta = self.sigmoid(self._input_x[j],w)
                delta = theta - self._input_y[j]
                w = w - iter_rate*delta*self._input_x[j]
                self._history_w.append(w)
                self._likelihood.append(self.likelihood_function(w))
        self._final_w = w
        return w


    def draw_result(self,title):
        total_data = np.shape(self._input_y)[0]
        self._nagative_x = []
        self._positive_x = []
        for i in range(total_data):
            if self._input_y[i]>0:
                self._positive_x.append(self._input_x[i])
            else:
                self._nagative_x.append(self._input_x[i])

        plt.figure(1)
        x1 = [x[1] for x in self._positive_x]
        x2 = [x[2] for x in self._positive_x]
        plt.scatter(x1,x2,label = 'positive',color = 'g',s=20,marker='o')
        x1 = [x[1] for x in self._nagative_x]
        x2 = [x[2] for x in self._nagative_x]
        plt.scatter(x1,x2,label = 'nagtive',color = 'r',s=20,marker='x')
        plt.xlabel('density')
        plt.ylabel('sugar')
        def f(x):
            return -(self._final_w[0]+self._final_w[1]*x)/self._final_w[2]
        x = np.linspace(0,1,endpoint=True)
        plt.plot(x,f(x),'b-',lw = 1)
        plt.title(title)
        plt.legend()
        plt.show()

    def draw_w_history(self, title):
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        x = np.arange(len(self._history_w))
        w0 = [w[0] for w in self._history_w]
        w1 = [w[1] for w in self._history_w]
        w2 = [w[2] for w in self._history_w]
        ax1.set_title(title + ' w trend')
        ax1.set_ylabel('w[0]')
        ax1.scatter(x, w0, label='w[0]', color='b', s=10, marker=".")
        ax2.set_ylabel('w[1]')
        ax2.scatter(x, w1, label='w[1]', color='g', s=10, marker=".")
        ax3.set_ylabel('w[2]')
        ax3.scatter(x, w2, label='w[2]', color='r', s=10, marker=".")
        plt.show()

    def draw_likelihood_function(self, title):
        plt.figure(1)
        x = np.arange(len(self._likelihood))
        plt.scatter(x, self._likelihood, label='Likelihood', color='g', s=10, marker=".")
        plt.xlabel('x')
        plt.ylabel('Likelihood function')
        plt.title(title + ' Likelihood trend')
        plt.legend()
        plt.show()

def main():
    log = Logistic()
    log.load_input_data("watermelon.csv")
    # log.batch_gardient_descent(iter_num=3000, iter_rate=0.001)
    # title = "Batch Gradient Descent"
    log.stochastic_gradient_descent(iter_num=100, iter_rate=0.001)
    title = "Stochastic Gradient Descent"
    log.draw_result(title)
    log.draw_w_history(title)
    log.draw_likelihood_function(title)

if __name__ == '__main__':
    main()