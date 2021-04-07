import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Dataset/Restaurant_Reviews.tsv',delimiter='\t',quoting= 3)
#cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('stopwords')
corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #not remove a-z and A-Z
    review = review.lower() #convert the string to lower cases
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#create the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

#Fitting naive bayes classifier to the dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
#Predicting results
y_pred = classifier.predict(x_test)

#Making confusion matrix (It is used to evulate the performance)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)#The trace of matrix give the correct prediction and other sum of diagonal give the incorrect prediction
