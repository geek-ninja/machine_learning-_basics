#Dataset template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Dataset/Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

#Scaling the variable to small value.
"""sc_x = preprocessing.StandardScaler()
x_train = sc_x.fit_transform(x_train) #fit_transform is like join the columns and change
x_test = sc_x.transform(x_test)""" # transform is like directly change the table

#fitting linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#prediction using model
y_predict = regressor.predict(x_test)
print('test case'," ",'prediction')
for i in range(len(y_test)):
    print(y_test[i]," ",y_predict[i])
    
#Visualizaion of model
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title("Salary Vs Experence")
plt.xlabel('years of exp')
plt.ylabel('Salary')
plt.show()