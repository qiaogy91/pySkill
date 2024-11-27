#!/usr/bin/env python
# @Project ：pySkill 
# @File    ：s01.py
# @Author  ：qiaogy
# @Email   ：qiaogy@example.com
# @Date    ：2024/11/18 14:06
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# （1）加载数据、数据划分
x, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# （2）交叉验证、获取最佳 K值
df = pd.DataFrame(columns=['score'])
for x in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=x)
    # 对每个k执行 “5折交叉验证”，返回每个Fold评分 [0.85,0.88,0.86,0.83,0.87]
    # scoring 如何评估模型的表现，accuracy 表示分类模型的准确率，适用于分类任务。计算预测正确的样本数与总样本数的比例
    v = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    df.loc[x, 'score'] = v.mean()

k = df.loc[:, 'score'].idxmax()

# （3）使用最佳值训练模型
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# (4)进行预测
y_ = model.predict(X_test)


# （4）评估当前模型
res = model.score(X_test, y_test)
print(res)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_)
print(accuracy)
