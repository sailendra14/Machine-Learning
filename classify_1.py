# Loading Required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier as KNC

# Loading Datasets
iris = datasets.load_iris()

# Printing Description
# print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0], labels[0])

# Traning the Classifier
clf = KNC()  #KNC = KNeighborsClassifier
clf.fit(features, labels)

preds = clf.predict([[1, 1, 1, 1]])
print(preds)
