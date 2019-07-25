import pandas as pd
import math, datetime
import quandl
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
#Linear Regression Example
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

#Applying the porcentage change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#This dataframe contains the features

df = df[['Adj. Close','HL_PCT', 'PCT_change','Adj. Volume']]

prediction_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

#Predicting tomorrows price by multiplying plus 0.01
prediction_out = int(math.ceil(0.1 * len(df)))

#Creating the label column; This way, each row , the label column for each row will be the adjusted close price one day into the future
df['label'] = df[prediction_col].shift(-prediction_out)


#Defining X and Y, x = Features and y = labels
X = np.array(df.drop(['label'], 1))
#Scaling X
X = preprocessing.scale(X)
X_lately = X[-prediction_out:]
X = X[:-prediction_out]


df.dropna(inplace = True)

y = np.array(df['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#Este codigo de aqui lo ejecuto la primera vez que entrenare al algoritmo y una vez generado el archivo pickle, puedo comentarlo ya que lo cargara.

#Defining classifier
clf = LinearRegression()
#If we want to use a different algorithm
#clf = svm.SVR()
clf.fit(X_train, y_train) #Train

#Pickle will help us to save the classifier in order not to start trainning everytime the algorithm

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) #Test

#print accuracy #The square error

#Predicting Stuff
prediction_set = clf.predict(X_lately) #The parameter can be one single value or an array of values
print prediction_set, accuracy, prediction_out

df['Forecast'] = np.nan

last_day = df.iloc[-1].name #Getting the last day
last_unix = last_day.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#Having the dates on the axis
for i in prediction_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
