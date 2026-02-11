# K-nearst neighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
x , y  = iris.data , iris.target

X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train , Y_train)
Y_pred = knn.predict(X_test)

accuracy = accuracy_score(Y_test , Y_pred)

print(f'Accuracy : {accuracy : 0.2f}')