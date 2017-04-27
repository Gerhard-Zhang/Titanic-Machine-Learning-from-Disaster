# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:51:45 2017

@author: Admin
"""

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_test = pd.read_csv('X_test.csv')
X_train = pd.read_csv('X_train.csv')

y_train = train['Survived']
#y_test = test['Survived']   莫名其妙通不过

print X_train.info()
print X_test.info()
print 'y_train:\n'
print y_train
print '打印完毕\n'

'''
使用决策树进行分类
'''
#使用scikit-learn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

#转换特州，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
print vec.feature_names_

#同样需要对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))

#从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc=DecisionTreeClassifier()
#用训练数据进行模型学习
dtc.fit(X_train,y_train)
#用训练好的决策树模型对测试数据进行预测
y_predict = dtc.predict(X_test)

print '\n预测结果 y_predict:\n'

for i in range(0,418):
    print y_predict[i];

#将这个结果存储在文件中
DecisionTree_submission=pd.DataFrame(
        {'PassengerId':test['PassengerId'],'Survived':y_predict} )
DecisionTree_submission.to_csv('DecisionTree_submission.csv')


#从 sklearn.ensemble 中导入 RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
#使用默认配置初始化
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict = rfc.predict(X_test)

rfc_submission=pd.DataFrame(
        {'PassengerId':test['PassengerId'],'Survived':rfc_y_predict} )
rfc_submission.to_csv('rfc_submission.csv')

'''此处结果是 No module named xgboost
#从流行工具包xgboost导入XGBClassifier 用于处理分类预测问题
from xgboost import XGBClassifier
xgbc = XGBClassifier()
#使用默认配置的XGBClassifier进行预测操作

xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission=pd.DataFrame(
        {'PassengerId':test['PassengerId'],'Survived':xgbc_y_predict} )
xgbc_submission.to_csv('xgbc_submission.csv')
'''#xgboost  xgbc 模块结束

#使用 k近邻分类器
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
knc_y_predict = knc.predict(X_test)
knc_submission=pd.DataFrame(
        {'PassengerId':test['PassengerId'],'Survived':knc_y_predict} )
knc_submission.to_csv('knc_submission.csv')

#使用支持向量机
from sklearn.svm import SVR
#使用线性核函数配置
linear_svr = SVR()
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)
linear_svr_submission = pd.DataFrame(
        {'PassengerId':test['PassengerId'],'Survived':linear_svr_y_predict} )
linear_svr_submission.to_csv('linear_svr_submission.csv')

'''
#从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report
#输出预测准确性
print '输出预测准确性:'
print dtc.score(X_test,y_test)

#输出更加详细的分类性能
print '输出更加详细的分类性能:'
print classification_report(y_predict, y_test, target_names=['died','Survived'])
'''
'''
#解析下来便是采用 DictVectorizer对特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
dict_vec.feature_names_

X_test = dict_vec.transform(X_test.to_dict(orient='record'))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

from xgboost import XGBClassifier
xgbc = XGBClassifier()

from sklearn.cross_validation import cross_val_score
cross_val_score(rfc, X_train, y_train, cv=5).mean()

'''