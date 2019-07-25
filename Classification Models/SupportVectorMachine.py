'''
Support Vector Machine
Dealing with vector space.
'''
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm

df = pd.read_csv('breast-cancer.csv')
df.replace('?',-99999, inplace = True)
df.drop(['id'], 1, inplace = True)

#Defining X and Y
X = np.array(df.drop(['class'], 1)) #Features
Y = np.array(df['class'])           #Labels

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)

clf = svm.SVC()

#Trainning the classifier

clf.fit(x_train, y_train)

#Testing the  classifier

accuracy = clf.score(x_test, y_test)

#Making a prediction

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]]) #Samples
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)

print prediction        #Prints [2 2]
print "Accuracy: ", accuracy
