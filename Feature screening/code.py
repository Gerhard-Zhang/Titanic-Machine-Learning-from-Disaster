# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:17:28 2017

@author: Gerhard
"""

#导入panda方便数据读取和预处理
import pandas as pd

#从互联网上读取 titanic 的数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

#print titanic.info()

titanic.head()
titanic.info()


#分离数据特征与预测目标
y = titanic['survived']
X = titanic.drop(['row.name','name','survived'])

'''
#分割数据，依然采样 25% 用于测试
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=33)
'''