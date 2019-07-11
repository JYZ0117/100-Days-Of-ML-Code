#Step 1: Data Preprocessing 数据预处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\Lenovo\Desktop\python work\studentscores.csv')

X=dataset.iloc[ : ,   : 1 ].values
Y=dataset.iloc[ : , 1 ].values


#Step 2: Fitting Simple Linear Regression Model to the training set 将简单线性回归模型拟合到训练集上
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(X_train,Y_train)

#Step 3: Predecting the Result 预测结果
Y_pred=regressor.predict(X_test)

#Step 4: Visualization 可视化
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , Y_pred, color ='blue')
plt.show()

