#Dataset template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Dataset/Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
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

#print(y)
#Training and testing
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Scaling the variable to small value.
# sc_x = preprocessing.StandardScaler()
# sc_y = preprocessing.StandardScaler()
# print(x)
# y = np.reshape(y,(-1,1))
# x = sc_x.fit_transform(x) #fit_transform is like join the columns and change
# y = sc_y.fit_transform(y) # transform is like directly change the table

#Fitting svr to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
#Visualization
y_pred = regressor.predict([[6.5]])
print('Predicted value ',y_pred)
#To make the graph higer resolution for proper decision tree regressor in 1D . (As it takes average within the intervals) the it join the points.
#This mode is good in 2d system or more.
x_grid = np.arange(min(x),max(x,),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('Truth or Bluf (decision tree)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#This mode is very power full in 2D and 3D or more ... not good in 1D. 