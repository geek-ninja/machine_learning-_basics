import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Dataset/Market_Basket_Optimisation.csv',header=None) #remove the titles
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions,min_support = 0.003,min_confidence = 0.2,min_lift = 3,min_length = 2)
#Adjust below three parameter according to user target and dataset
#Support = (3*7)/7500
#confidence = 0.2 = 20%
#lift = 3

#Visualization
results = list(rules)
print(results)

#This algo used in recomandations of movies ,books,food_items