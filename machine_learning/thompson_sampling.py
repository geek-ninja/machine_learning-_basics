import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

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
#implementing thompson_sampling

d = 10
N = 10000

numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0

for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)            
        if random_beta > max_random:
            max_random = random_beta
            ad = i 
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        numbers_of_rewards_1[ad] +=1
    else:
        numbers_of_rewards_0[ad] +=1
        
    total_reward += reward
print(total_reward)

#visualization of results
plt.hist(ads_selected)
plt.title("Histogram of ads")
plt.xlabel('ads')
plt.ylabel('no.of time each ad was selected')
plt.show()
    