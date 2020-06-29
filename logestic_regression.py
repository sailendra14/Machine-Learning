from sklearn import datasets
import numpy as np 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(iris['data'].shape)
# print(list(iris.keys()))
# print(iris['target'])
# print(iris['DESCR'])

x = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(np.int)
# print(x)
# print(y)

# Train a logestic regression classifier
clf = LogisticRegression()
clf.fit(x, y)

eg = clf.predict(([[2.6]]))
# print(eg)

# Using matplotlib to plot the visulations
x_new = np.linspace(0, 3, 1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
# print(y_prob)

plt.plot(x_new, y_prob[:,1], "g-", lable="virginica")
plt.show()
# print(x_new)