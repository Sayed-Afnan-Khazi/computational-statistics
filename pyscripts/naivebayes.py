from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Loading the data
data = pd.read_csv("datasets/Iris.csv")

# Getting the data
X = data.loc[:, data.columns!="Species"]
Y = data["Species"]

# Splitting the data into testing and training 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.3)

# Building the model
model = GaussianNB()
model.fit(Xtrain,Ytrain)
pY = model.predict(Xtest)

# Showing the confusion matrix to verify our results
confmat = confusion_matrix(Ytest,pY)

plt.imshow(confmat)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        plt.text(j, i, confmat[i, j], ha="center", va="center", color="red")
plt.show()