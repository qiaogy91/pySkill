#!/usr/bin/env python
# @Project ：pySkill 
# @File    ：s1.py
# @Author  ：qiaogy
# @Email   ：qiaogy@example.com
# @Date    ：2024/12/3 08:51
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('./data/BostonHousing.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 归一化

# X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

# 训练
grid = GridSearchCV(
    estimator=KNeighborsRegressor(),
    param_grid={
        'n_neighbors': [x for x in range(1, 15)],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
    },
    cv=5,
    scoring='neg_mean_squared_error',  # mean_squared_error 结果一定是正数，数字越大越表示偏差越大，加上负号后，越大表示偏差越小
)

grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))