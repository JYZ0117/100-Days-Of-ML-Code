#导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#导入数据集
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
#划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print(sc)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#使用K—NN算法对训练集进行训练
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
print(classifier)
classifier.fit(X_train, y_train)
#预测测试集
y_pred = classifier.predict(X_test)
print(y_pred)
#生成混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)