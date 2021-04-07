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
sc_x = preprocessing.StandardScaler()
sc_y = preprocessing.StandardScaler()
print(x)
y = np.reshape(y,(-1,1))
x = sc_x.fit_transform(x) #fit_transform is like join the columns and change
y = sc_y.fit_transform(y) # transform is like directly change the table

#Fitting svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)
#Visualization
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))
print('Predicted value ',y_pred)
plt.scatter(x,y,color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('Truth or Bluf (svr)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()