#Dataset template

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Dataset/Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
#To manage null values ...
"""imputer = SimpleImputer()
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])"""

#To manage duplicate values and encoding them
# encoderx = preprocessing.LabelEncoder()
# x[:,0] = encoderx.fit_transform(x[:,0])

#To manage encoder column to a binary repersentation

# ct = ColumnTransformer([("Country", preprocessing.OneHotEncoder(), [0])], remainder = 'passthrough')
# x = ct.fit_transform(x)
# x = x.astype(int)

#print(y)
# encodery = preprocessing.LabelEncoder()
# y = encodery.fit_transform(y)
#print(y)
#Training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Scaling the variable to small value.
"""sc_x = preprocessing.StandardScaler()
x_train = sc_x.fit_transform(x_train) #fit_transform is like join the columns and change
x_test = sc_x.transform(x_test)""" # transform is like directly change the table