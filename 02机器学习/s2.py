#!/usr/bin/env python
# @Project ：pySkill 
# @File    ：s2.py
# @Author  ：qiaogy
# @Email   ：qiaogy@example.com
# @Date    ：2024/12/3 08:51


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


data = pd.read_csv('./data/BostonHousing.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 归一化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 归一化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# x_test = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))