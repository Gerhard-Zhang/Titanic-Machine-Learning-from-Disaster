# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:57:45 2017

@author: Admin
"""

#导入panda方便数据读取和预处理
import pandas as pd

#分别对训练和测试数据从本地进行读取
train = pd.read_csv('F:\Titanic-Machine Learning from Disaster/train.csv')
test = pd.read_csv('F:\Titanic-Machine Learning from Disaster/test.csv')

#输出一下看看是什么样子，看看数据类型什么的或者看看有没有缺失
print train.info()
print test.info()

selected_features = ['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']

X_train = train[selected_features]
X_test = test[selected_features]

y_train = train['Survived']

#补全 Embarked
print '\nHere\n'
print X_train['Embarked'].value_counts()
print X_test['Embarked'].value_counts()

#使用频率最高的特征值来填充

X_train['Embarked'].fillna('S',inplace=True)
X_test['Embarked'].fillna('S',inplace=True)

#填充age
X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_train['Fare'].mean(),inplace=True)

print X_train.info()
print X_test.info()

#X_train.to_csv('F:\Titanic-Machine Learning from Disaster/X_train.csv')
#X_test.to_csv('F:\Titanic-Machine Learning from Disaster/X_test.csv')