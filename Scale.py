import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv('home_data.csv')
X = df[['floors' , 'bedrooms' , 'bathrooms']]
y = df['price']

x_train , x_test , y_train , y_test = train_test_split

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled , y_train)

y_predict = model.predict(x_test_scaled)
mse = mean_squared_error(y_test , y_predict)

print(f'Mean Squared ERORR : {mse}')
