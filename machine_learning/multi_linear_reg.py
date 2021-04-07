#Dataset template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Dataset/50_Startups.csv')
print(dataset)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
#To manage null values ...
"""imputer = SimpleImputer()
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])"""

#To manage duplicate values and encoding them
# encoderx = preprocessing.LabelEncoder()
# x[:,0] = encoderx.fit_transform(x[:,0])

#To manage encoder column to a binary repersentation
#independant variable
ct = ColumnTransformer([("Country", preprocessing.OneHotEncoder(), [3])], remainder = 'passthrough') #the arg after OneHotEncoder() is the column to be encoded
x = ct.fit_transform(x)
x = x.astype(int)

#Avoid dummpy variable trap
x = x[:,1:]

#Training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Scaling the variable to small value.
"""sc_x = preprocessing.StandardScaler()
x_train = sc_x.fit_transform(x_train) #fit_transform is like join the columns and change
x_test = sc_x.transform(x_test)""" # transform is like directly change the table

#fitting multi linear regression
from  sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
#predicting the Test set 
y_pred = regressor.predict(x_test)
print('test      predict')
for i in range(len(y_pred)):
    print(y_test[i],'   ',y_pred[i])
    
#Bulding backward Eleminaion
import statsmodels.api as sm
x = sm.add_constant(x)# cof of 1 is included.
#above line is add to create a equ of 1*b0 + x*b1 + x*b2 + .......
x_opt = x[:,[0,1,2,3,4,5]]
model = sm.OLS(y,x_opt).fit()
model.summary()
#print(model.summary())

#In first round of summary column 2 has max p value ,so remove that col and fit rest.
x_opt = x[:,[0,1,3,4,5]]
model = sm.OLS(y,x_opt).fit()
model.summary()
#print(model.summary())

#In second round of summary column 1 has max p value ,so remove that col and fit rest.
x_opt = x[:,[0,3,4,5]]
model = sm.OLS(y,x_opt).fit()
model.summary()
#print(model.summary())

#In third round of summary column 4 has max p value ,so remove that col and fit rest.
x_opt = x[:,[0,3,5]]
model = sm.OLS(y,x_opt).fit()
model.summary()
#print(model.summary())

#In forth round of summary column 5 has max p value ,so remove that col and fit rest.
x_opt = x[:,[0,3]]
model = sm.OLS(y,x_opt).fit()
model.summary()
print(model.summary())

#now every column has  p value less than 0.05 (significant)
#So model is ready by backward elemination.