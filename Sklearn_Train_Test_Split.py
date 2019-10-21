import numpy as np
#use sklearn to split training and testing data
from sklearn.model_selection import train_test_split

#generate array from 1 to 100
a = np.arange(1,101)
b = np.arange(501,601)

#this will do a 75:25 split and return an array [training array, testing array]
train_test_split(a)

#this split a and b with same random index. you can specify split ratio and random seed
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=105)