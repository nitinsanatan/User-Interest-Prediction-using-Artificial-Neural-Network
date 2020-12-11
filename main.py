#importing basic libraries
import pandas as pd
from sklearn.model_selection import train_test_split

#uploading buddymove.csv on Google Colab
from google.colab import files
uploaded= files.upload()

#importing the buddymove.csv file
import io
dataset= pd.read_csv('buddymove.csv')
dataset.head

#checking the shape of dataset
dataset.shape

#assigning target and feature variable
features=['Picnic','Religious','Nature','Theatre','Shopping']
X=dataset[features]
Y=dataset.Sports

#splitting thhe data into 70% training and 30% testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)

#implementing perceptron
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

clf= Perceptron()
clf.fit(X_train, Y_train) #training data using perceptron

#Accuracy using Perceptron Model
clf.score(X,Y)
