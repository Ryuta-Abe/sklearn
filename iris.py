#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn import cross_validation

iris = datasets.load_iris()
# 先頭から100個のデータ(setosaとversicolorを抽出)
# 特徴は0番目(sepal length)と2列目(petal length)を使用
data = iris.data[0:100][:,::2]
target= iris.target[0:100]

# 学習データとテストデータを4:1に分割
train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)
clf = LinearSVC() # 線形SVM
clf.fit(train_x, train_y) # 学習
pred = clf.predict(test_x) # テストデータの識別
print (list(pred == test_y).count(True) / float(len(test_y)))

# test