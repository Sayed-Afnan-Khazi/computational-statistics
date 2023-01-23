import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Data
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]

x = np.array(x).reshape(-1,1)
y = np.array(y)

# Creating the model
model = LinearRegression()

# Fitting to the model
model.fit(x,y)

# Predicting using the model
pY = model.predict(x)

# Drawing the line of best fit
plt.scatter(x,y,color="blue")
plt.plot(x,pY,color="red")
plt.show()