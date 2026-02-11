# Multiple Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

x = ['feature1' , 'feature2' , 'feature3']
y = ['target_variable']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

model = LinearRegression()
model.fit(x_train , y_train)

y_predict = model.predict(x_test)

mse = mean_absolute_error(y_test , y_predict)
print(f'Mean Squared Error : {mse}')