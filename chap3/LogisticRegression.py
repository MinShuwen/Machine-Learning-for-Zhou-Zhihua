# -*- coding=utf-8 -*-
'''
sklearn 库实现逻辑回归
'''

from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def loadDataSet(fileName):
    # load data
    dataMat = [] # 数组
    file = open(fileName)
    for line in file.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(len(curLine)):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
    dataArr = array(dataMat)
    return dataArr

# draw scatter diagram to show the raw data
def plot(dataset):
    # seperate the data from the target attributes
    X = dataset[:,1:3]
    y = dataset[:,3]
    f1 = plt.figure(1)
    plt.title('watermelon')
    plt.xlabel('density')
    plt.ylabel('sugar')
    plt.scatter(X[y==1.0,0],X[y==1.0,1],marker='o',color='k',label = 'good')
    plt.scatter(X[y==0.0,0],X[y==0.0,1],marker='o',color='g',label = 'bad')
    plt.legend(loc = 'upper right')
    return plt

'''
using sklearn lib for LR
'''
def lr_sklearn(dataset):
    # get train and test set
    X = dataset[:, 1:3]
    y = dataset[:, 3]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.5,random_state=0)

    # model training
    log_model = LogisticRegression()
    log_model.fit(X_train,y_train)

    # model testing
    y_pred = log_model.predict(X_test)

    # summarize the acc of fitting
    confusion_matrix = metrics.confusion_matrix(y_test,y_pred) # 混淆矩阵
    result = metrics.classification_report(y_test,y_pred) # 查全率、查准率、F1值
    return confusion_matrix, result



if __name__ == '__main__':
    dataset = loadDataSet('watermelon.csv')
    watermelon = plot(dataset)
    # watermelon.show()
    confusion_matrix, result = lr_sklearn(dataset)
    print(confusion_matrix)
    print(result)