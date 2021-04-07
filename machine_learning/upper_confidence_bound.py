import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Dataset/Ads_CTR.csv')

#--------------------------- Random Selection Algo -----------------------
# import random 
# n = 10000
# d = 10
# ads_selected = []
# total_reward = 0
# for n in range(0,n):
#     ad = random.randrange(d)
#     ads_selected.append(ad)
#     reward = dataset.values[n,ad]
#     total_reward += reward

# plt.hist(ads_selected)
# plt.title("Histogram of ads")
# plt.xlabel('ads')
# plt.ylabel('no.of time each ad was selected')
# plt.show()
#---------------------------------------------------------------------------
#implementing ucb

d = 10
N = 10000

numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(0,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        if numbers_of_selection[i] > 0:
            average_reward = sums_of_rewards[i]/numbers_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selection[i]) #log(n+1) is to make the first index 1 instead of 0
            ucb = average_reward + delta_i
        else:
            ucb = 1e400
            
        if ucb > max_ucb:
            max_ucb = ucb
            ad = i 
    ads_selected.append(ad)
    numbers_of_selection[ad] +=1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
#print(total_reward)

#visualization of results
plt.hist(ads_selected)
plt.title("Histogram of ads")
plt.xlabel('ads')
plt.ylabel('no.of time each ad was selected')
plt.show()
    