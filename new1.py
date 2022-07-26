from matplotlib.colors import hsv_to_rgb

V, H = np.mgrid[0:1:100j, 0:1:360j]
S = np.ones_like(V)

HSV_S100 = np.dstack((H, S * 1.0, V))
RGB_S100 = hsv_to_rgb(HSV_S100)
HSV_S20 = np.dstack((H, S * 0.2, V))
RGB_S20 = hsv_to_rgb(HSV_S20)

# HSV_S20.shape #getting size (x, y, HSV)
# image processing

import copy

image_Map = copy.deepcopy(HSV_S20[:,:,2]) #taking value of image, no need for Hue and Saturation

x_Map = HSV_S20.shape[0]
y_Map = HSV_S20.shape[1]

initX = x_Map / 2
initY = y_Map / 2 #can be changed

def hill_climbing(image_Map, initX, initY, searchSize):
    maxVal = 0
	for i in range(initX - searchSize, initX + searchSize):
        for j in range(initY - searchSize, initY + searchSize):
            if image_Map[i,j] > maxVal :
                hill_climbing(image_Map, i, j, searchSize)
            else :
                return [i, j]
# simple version

import numpy as np 
import pandas as pd  
import seaborn as sns   
import matplotlib.pyplot as plt

df = #image best values datas
df.columns = ["X", "Y", "sec_X", "sec_Y"]

df.dtypes  #type check
df.describe() #statics

sns.regplot(df, x="X", y="Y") #regression plot

df.isnull().sum() #check missing feature

df = df.dropna() #drop

y = df[[]]
X = df
X.columns

from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # train:testSplit = 80:20
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42) #train:validation = 75:25
# for decision tree

from utils import *
from sklearn.linear_model import Perceptron

data = copy.deepcopy(df)
labels = np.array(#how to label?)

plot_data(data, labels) #see plot

perceptron_xor = Perceptron()
perceptron_xor.fit(data, labels)

perceptron_xor.score(data, labels)

pred_scores = perceptron_xor.decision_function(data)
plot_decision_boundary(perceptron_xor, data, labels, "perceptron")

#can be AND, OR, XOR, for linear model

from sklearn.neural_network import MLPClassifier

xor_data = data
xor_labels = labels

mlp_xor = MLPClassifier(hidden_layer_sizes = (5, ), activation = 'log', max_iter = 10000)
mlp_xor.fit(xor_data, xor_labels) #fitting model on data
mlp_xor.score(xor_data, xor_labels) #scoring

plot_decision_boundary(mlp_xor, xor_data, xor_labels, "MLP") #see plot

#for non-linear model



