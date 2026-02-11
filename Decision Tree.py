# Decision Tree
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix

iris  = load_iris()
X = pd.DataFrame(iris.data , columns = iris.feature_names)
Y = pd.Series(iris.target)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 42)
clf = DecisionTreeClassifier()

clf.fit(X_train , Y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test , y_pred)
conf_matrix = confusion_matrix(Y_test , y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
