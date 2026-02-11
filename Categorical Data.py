# Categorical Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoder = LabelEncoder()
colors = ['red' , 'green' , 'blue' , 'yellow' , 'red']

encoded_colors = label_encoder.fit_transform(colors)
print(f'Ordinal Colors: {colors}')
print(f'Encoded Colors: {encoded_colors}')

print("=====================================================================")
data = {'genre': ['action' , 'comedy' , 'drama' , 'action' , 'comedy']}
df = pd.DataFrame(data)

one_hot_encoded = pd.get_dummies(df['genre'] , prefix = 'genre')
print(f'Original Colors: \n{df}')
print(f'One-hot Encoded: \n{one_hot_encoded}')