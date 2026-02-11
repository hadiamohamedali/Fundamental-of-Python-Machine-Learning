# Logistic Regression
import numpy as np
from sklearn import linear_model

X = np.array([3.78 , 2.34 , 3.56 , 2.09 , 0.14 , 1.67 , 2.69 , 5.88]).reshape(-1 , 1)
Y = np.array([0 , 0 , 0 , 0 , 0 , 1 , 1 , 1])

logr = linear_model.LogisticRegression()
logr.fit(X , Y)

log_odds = logr.coef_
odds = np.exp(log_odds)

# 1 usage
def logit2prob(logr , X):
    log_odds = logr.coef_ * X + logr.intercept_
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)
    return(probability)
print(logit2prob(logr , X))
