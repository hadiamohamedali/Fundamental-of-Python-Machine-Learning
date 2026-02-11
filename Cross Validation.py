# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x , y = iris.data , iris.target

knn = KNeighborsClassifier(n_neighbors = 3)

cv_scores = cross_val_score(knn , x , y , cv = 5)
print(f'Cross Validation Scores : {cv_scores}')
print(f'Mean CV Section : {cv_scores.mean()}')