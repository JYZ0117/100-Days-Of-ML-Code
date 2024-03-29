# step1:Importing the libraries 导入库
import numpy as np
import pandas as pd

# step2:Importing dataset 导入数据集
dastaset=pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values

# step3:Handling the missing data 处理丢失数据
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

# step4:Encoding categorial data 编辑代码分类数据
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

# step5: Splitting the datasets into training sets and Test sets 将数据及拆分成训练集和测试集
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

# step 6: Feature Scaling 衡量特征
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
