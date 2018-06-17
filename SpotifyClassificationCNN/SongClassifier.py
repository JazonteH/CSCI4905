
# coding: utf-8

# In[11]:


import pandas as pd #DataFrame
import numpy as np #Scientific Computing Packages -Array

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns

import graphviz
import pydotplus
import io
from scipy import misc

get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


#Imports data from file
data = pd.read_csv('data.csv') 


# In[13]:


#Splits data into two sets, training data and test data
#test data is size of 15 percent of all data
train, test = train_test_split(data, test_size = 0.15)


# In[14]:


#Creates decision tree classifier
c = DecisionTreeClassifier(min_samples_split = 100)


# In[15]:


#Defines a set of features the Decision Tree Classifier should consider
features = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness","key","speechiness", "duration_ms"]


# In[16]:


x_train = train[features]
y_train = train["target"]

x_test = test[features]
y_test = test["target"]


# In[17]:


dt = c.fit(x_train, y_train)


# In[18]:


y_pred = c.predict(x_test)
score = accuracy_score(y_test, y_pred)*100


# In[19]:


print("Accuracy using a single Decision Tree: ", round(score, 2), "%")


# In[20]:


#Improve Accuracy by using various random forests
#Random Forest with 4 features and 10 Decision Trees
random_forest1 = RandomForestClassifier(max_features= 4)

dt = random_forest1.fit(x_train, y_train)
y_pred = random_Forest1.predict(x_test)
score = accuracy_score(y_test, y_pred)*100
print("Accuracy using 10 Tree Random Forest: ", round(score, 2), "%")

#Random Forest with 5 trees and max features of 3
random_forest2 = RandomForestClassifier(n_estimators=5, max_features=3)
dt = random_forest2.fit(x_train, y_train)
y_pred = random_forest2.predict(x_test)
score = accuracy_score(y_test, y_pred)*100
print("Accuracy using 5 Tree Random Forest: ", round(score, 2), "%")

random_forest3 = RandomForestClassifier(n_estimators = 15)
dt = random_forest3.fit(x_train, y_train)
y_pred = random_forest3.predict(x_test)
score = accuracy_score(y_test, y_pred)*100
print("Accuracy using 15 Tree Random Forest: ", round(score, 2), "%")

