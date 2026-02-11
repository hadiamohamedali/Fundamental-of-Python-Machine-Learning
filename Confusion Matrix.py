# Confusion Matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report

X , Y = make_classification(n_samples = 1000 , n_features = 10 , n_classes = 2 , random_state = 42)
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 42)

clf = RandomForestClassifier()

clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)

conf_matrix = confusion_matrix(y_test , y_pred)

print(f'Confusion Matrix : \n{conf_matrix}')
print(f'Classification  Report : \n{classification_report (y_test , y_pred)}')