#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ## Importing the libraries

# In[47]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# ## Importing the dataset

# In[36]:


dataset=pd.read_csv("breast_cancer.csv")
dataset
## all features except the sample code are between 1-10.
## 2- if it benign and 4- if it is malignant(cancerous)


# In[53]:


X=dataset.iloc[:, 1:-1].values     ## sample code number wasn't a great feature in determining the class so we removed it.
y=dataset.iloc[:,[-1]].values
## dividing into dependent and independent variables.


# In[38]:


X


# In[39]:


y


# ## Splitting the dataset into the Training set and Test set

# In[40]:


from sklearn.model_selection import train_test_split  ## not importing the whole sklearn library. just one of its module.
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)
## random state can have any no.


# ## Training the Logistic Regression model on the Training set

# In[41]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# ## Predicting the Test set results

# In[42]:


y_pred=classifier.predict(X_test)


# ## Making the Confusion Matrix

# In[43]:


from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))
## out of 137
## 98% accuracy.


# ## Computing the accuracy with k-Fold Cross Validation

# In[44]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

