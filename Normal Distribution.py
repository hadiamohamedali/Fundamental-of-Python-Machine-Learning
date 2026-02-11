import matplotlib.pyplot as plt
import numpy as np

exam_scores = np.random.normal(70 , 10 , 100)

plt.hist(exam_scores , bins = 30 , edgecolor = 'black')
plt.title('Distribution of Exam Score')
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.show()