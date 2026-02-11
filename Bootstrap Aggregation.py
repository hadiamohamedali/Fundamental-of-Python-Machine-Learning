# Bootstrap Aggregation
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Bootstrap Aggregation : A Powerful technique to amplify the performance.
x , y = make_classification(n_samples = 1000 , n_features = 20 , n_informative = 10 , n_classes = 2 , random_state = 42)
X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)
base_model = DecisionTreeClassifier(random_state = 42)

bagging_model = BaggingClassifier(base_model , n_estimators = 50 , random_state = 42)
bagging_model.fit(X_train , Y_train)

Y_pred = bagging_model.predict(X_test)

accuracy = accuracy_score(Y_test , Y_pred)
print(f'Accuracy : {accuracy}')