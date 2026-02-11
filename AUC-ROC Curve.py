# AUC-ROC Curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve , auc
import matplotlib.pyplot as plt

x , y = make_classification(n_samples = 1000 , n_features = 20 , n_classes = 2 , random_state = 42)
X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

model = LogisticRegression()
model.fit(X_train , Y_train)

Y_prob = model.predict_proba(X_test)[: , 1]
fpr , tpr , thresholds = roc_curve(Y_test , Y_prob)

roc_auc = auc(fpr , tpr)
plt.figure(figsize = (8 , 6))
plt.plot(fpr , tpr , color = 'darkorange' , lw = 2 , label = f'AUC = {roc_auc : .2f}')
plt.plot([0 , 1] , [0 , 1] , color = 'navy' , lw = 2 , linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.legend(loc = 'lower right')
plt.show()