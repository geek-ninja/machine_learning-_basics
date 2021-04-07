#This algo helpls to add new data to the right cateogry
#Dataset template
# Logistic Regression is a classifier technique(grouping)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Dataset/Social_Network_Ads.csv')
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
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
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

#Scaling the variable to small value.
sc_x = preprocessing.StandardScaler()
x_train = sc_x.fit_transform(x_train) #fit_transform is like join the columns and change
x_test = sc_x.transform(x_test) # transform is like directly change the table

#Fitting k neighbours algo
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
#Predicting results
y_pred = classifier.predict(x_test)

#Making confusion matrix (It is used to evulate the performance)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)#The trace of matrix give the correct prediction and other sum of diagonal give the incorrect prediction

#Visualization 
from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1,step = 0.01),
                    np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1,step = 0.01)) #setting the pexicles

#To draw the separator (straight line)
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap = ListedColormap(('red','green')))#if 0 (red) if 1 (green)
#now to color the scattered points either red or green we will iterate and check predict
#if 1 make it green else red
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],c = ListedColormap(('red','green'))(i),label = j)
    
#now label the axies
plt.title('K-NN(Test Set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
