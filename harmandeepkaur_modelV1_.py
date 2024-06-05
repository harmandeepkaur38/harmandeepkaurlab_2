#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score 


# Load the dataset
data = pd.read_csv('wdbc.data', header=None)

# Set column names
columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data.columns = columns

# Drop the ID column
data = data.drop(columns=['ID'])

# Encode the Diagnosis column
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Split data into features and target variable
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']
# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=40)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[2]:


# Traning the model using Logistic regression()
model = LogisticRegression()
model.fit(X_train, y_train)


# In[3]:


# Evaluating the model
# Predict the test set results
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print('-'* 90)


# In[ ]:





# In[ ]:





# In[ ]:




