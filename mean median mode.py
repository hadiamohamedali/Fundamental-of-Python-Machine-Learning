# Lesson 1 : Mean , Median and Mode

# First Mean in Python
data = [12 , 18 , 14 , 20 , 16]
mean = sum(data) / len(data)
print(f"Mean : {mean}")

# Median in Python
sorted_Data = sorted(data)
n = len(sorted_Data)
median = (sorted_Data[n // 2] + sorted_Data[(n - 1) // 2]) / 2
print(f"Median :{median}")

# Mode in Python
from statistics import mode
mode_value = mode(data)
print(f"Mode : {mode_value}")

