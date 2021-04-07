#Dataset template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Dataset/Position_Salaries.csv')
#print(dataset)
x = dataset.iloc[:,1:2].values #Always x is a matrix.
y = dataset.iloc[:,2].values   #Always y is a vector (array)
#To manage null values ...
"""imputer = SimpleImputer()
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])"""

#To manage duplicate values and encoding them
# encoderx = preprocessing.LabelEncoder()
# x[:,0] = encoderx.fit_transform(x[:,0])

#To manage encoder column to a binary repersentation
#Encoder for independant variables

# ct = ColumnTransformer([("Country", preprocessing.OneHotEncoder(), [0])], remainder = 'passthrough') #the arg after OneHotEncoder() is the column to be encoded
# x = ct.fit_transform(x)
# x = x.astype(int)

#Encoder for dependant variable
# encodery = preprocessing.LabelEncoder()
# y = encodery.fit_transform(y)
#print(y)
#Training and testing
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Scaling the variable to small value.
"""sc_x = preprocessing.StandardScaler()
x_train = sc_x.fit_transform(x_train) #fit_transform is like join the columns and change
x_test = sc_x.transform(x_test)""" # transform is like directly change the table

#fitting linear reg to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
#Fitting poly reg to the dataset
poly_reg = preprocessing.PolynomialFeatures(degree=4) #We can change the degree to make a better predict
x_poly = poly_reg.fit_transform(x) #first fit ,then transforming x to x2,x3,x3......(polynomial)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#Visualize linear reg
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('bluff detect of demand salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Visualize poly reg
#To make the graph curve in between the range 
x_grid = np.arange(min(x),max(x),0.1) # range is 0.1 
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x,y,color = 'red')
plt.plot(x_grid,lin_reg2 .predict(poly_reg.fit_transform(x_grid)),color = 'blue') #We shouldn't use x_poly in predict as it is already define and it will not work for other matrix
plt.title('bluff detect of demand salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

print(x)
# Predicting the results
res1=lin_reg.predict([[6.5]])
res2 = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print(res1,res2)