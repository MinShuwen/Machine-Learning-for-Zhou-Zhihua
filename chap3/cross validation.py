'''
k折交叉验证法
数据集：iris：通过花朵的形状数据推测花卉的类型
blood transfusion service center dataset：通过献血行为的历史数据，推测某人是否会在某一时段献血
'''

# 可用于数据处理和csv文件输出的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# 一个Python里面的图形库
import seaborn as sns

'''
sklearn中自带iris数据集的使用：
https://scikit-learn.org/dev/modules/generated/sklearn.datasets.load_iris.html
'''
sns.set(style='white', color_codes=True)
iris = datasets.load_iris()
# iris = pd.read_csv('iris.csv')
X = iris.data # array (150,4)
y = iris.target # array (150,1)
#
# f1 = plt.figure(1)
# plt.scatter(data[:,0], data[:,1])
# plt.legend(loc = 'species')
# plt.show()

'''
k-折交叉验证可直接根据
sklearn.model_selection.cross_val_predict()得到精度、F1值等度量（该函数要求1＜k＜n-1）
'''
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# log-regression lib model
log_model1 = LogisticRegression()
log_model2= LogisticRegression()

m = np.shape(X)[0] # 150

# 10-folds CV
y_pred = cross_val_predict(log_model1,X,y,cv=10)
print(metrics.accuracy_score(y,y_pred)) # 0.9533333333333334

# Loocv
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
acc = 0
for train, test in loo.split(X):
    log_model2.fit(X[train],y[train])
    y_pr = log_model2.predict(X[test])
    if y_pr == y[test]:
        acc += 1
print(acc/m) # 0.9533333333333334
