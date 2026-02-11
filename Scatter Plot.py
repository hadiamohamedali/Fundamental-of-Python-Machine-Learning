import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = sns.load_dataset('iris')

sns.scatterplot(x = 'sepal_length' , y = 'sepal_width' , hue = 'species', data = df)
plt.title('Scatter Plot : Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()