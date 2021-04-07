import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Dataset/Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13 ].values

encoderx = preprocessing.LabelEncoder()
x[:,1] = encoderx.fit_transform(x[:,1])


encoderx2 = preprocessing.LabelEncoder()
x[:,2] = encoderx2.fit_transform(x[:,2])

ct = ColumnTransformer([("Country", preprocessing.OneHotEncoder(), [1])], remainder = 'passthrough') #the arg after OneHotEncoder() is the column to be encoded
x = ct.fit_transform(x)
# x = x.astype(int)


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Scaling the variable to small value.
sc_x = preprocessing.StandardScaler()
x_train = sc_x.fit_transform(x_train) #fit_transform is like join the columns and change
x_test = sc_x.transform(x_test) # transform is like directly change the table

#making ann

import keras
from keras.models import Sequential
from keras.layers import Dense,InputLayer

classifier = Sequential()
#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation='relu',input_dim = 11)) #output dim = (11+1)//2
#Adding the second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation='relu'))
#Adding the output layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation='sigmoid'))

#Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#fitting ANN to the training set
classifier.fit(x_train,y_train , batch_size = 10, nb_epoch = 100)

#Predicting results

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
#Making confusion matrix (It is used to evulate the performance)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) 