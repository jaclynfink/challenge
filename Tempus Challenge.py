#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #Import PCA function


# In[12]:


data = pd.read_csv("./Downloads/data.csv", index_col=0)


# In[22]:


print(data.head())
#Return the first five rows of data to make sure it was read in correctly


# In[24]:


print(data.shape)


# Dataset contains 350 Samples, 17,869 genes

# In[40]:


#Center and scale the data so that the means for each gene are 0 and the std devs for each gene are 1. 
scaled_data = StandardScaler().fit_transform(data.T)
print(scaled_data)


# Pass through the transpose of the data so that the samples are read as columns, and the genes as rows.
# 
# Ref. https://www.youtube.com/watch?v=Lsue2gEM9D0

# In[27]:


pca = PCA()


# In[42]:


pca.fit(scaled_data)
#fit() fits the model by calculating the mean and std. dev


# In[39]:


pca_data = pca.transform(scaled_data)
#transform() uses the parameters calculated by fit() to generate a transformed data set
print(pca_data)


# In[32]:


percent_variation = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
#Calculate the percent variation that each principal component accounts for


# In[34]:


labels = ['PC' + str(x) for x in range(1, len(percent_variation)+1)]


# In[35]:


plt.bar(x=range(1,len(percent_variation)+1), height=percent_variation, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# The most variation in the data is on the left side of the graph with the large bars. Too much data, can't even read the x axis labels!!

# In[43]:


df = pd.DataFrame(pca_data)


# Suppose you were given the following preference data from a movie viewing system which contains 100 users
# and 1 movie. Reported measurements are given as a binary outcome 1- viewed movie, and 0 - did not view the
# movie (100 measurements total).
# 
# You are also given features describing each of the users in the following forms:
# 20000 real valued measurements per user
# 600 sparse binary measurements per user
# 3 user types: age 20-30, age 31-40, age 41-50 per user
# 
# <b>Question 6: How would you find features that are most associated with reported movie viewing?</b>
# <br>Feature selection techniques in ML can be used to select features that are most closely associated with the desired output. In this case, I would use Univariate Selection to return the most relevant features (which ones are most strongly associated with an outcome of 1). The output will be numerical, and you can even display it as a bar graph to easily understand how the values compare to one another.
# 
# 
# <b>Question 7: How would you determine if these associations are statistically significant?
# Suppose we expanded the movie library to include 10 movies. The viewing system reported a 100x10 binary
# matrix that takes the values {1 - viewed, 0 - not viewed} with 100 rows (one per user) and 10 columns (one per
# movie). You are also given 50 sparse features describing each one of the movies. </b>
# 
# <b>Question 8: How would you find combinations of user and movie features that are associated with reported
# movie viewing across the 10 movies?</b>
# 
# <b>Question 9: How would you determine if these associations are statistically significant?</b>
# 
# <b>Question 10: How would you modify your algorithm to deal with a dataset comprising of 1 million users, and
# 10,000 movies?</b>
# 

# In[ ]:




