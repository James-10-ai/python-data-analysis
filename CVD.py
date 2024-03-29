#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sns


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


cvd = pd.read_csv(r"C:\Users\Hp Elitebook\3D Objects\CVD_cleaned.csv")


# In[6]:


cvd.head()


# In[7]:


cvd.groupby('Age_Category').sum()


# In[ ]:





# In[8]:


cvd.info()


# In[9]:


#encoding the data columns by booling


# In[10]:


cvd['General_Health'] = cvd['General_Health'].map ( {'Poor':0,'Very Good':1,'Good':2, 'Fair':3, 'Excellent':4} ).astype(float)


# In[11]:


cvd['Checkup'] = cvd['Checkup'].map ( {'Within the past 2 years':0,'Within the past year':1,'5 or more years ago':2,'Never':3,'Within the past 5 years':4} ).astype(float)


# In[12]:


cvd


# In[13]:


cvd['Exercise'] = cvd['Exercise'].map ( {'Yes':0,'No':1} ).astype(int)


# In[14]:


cvd.head()


# In[15]:


cvd['Heart_Disease'] = cvd['Heart_Disease'].map ( {'Yes':0,'No':1} ).astype(int)


# In[16]:


cvd['Skin_Cancer'] = cvd['Skin_Cancer'].map ( {'Yes':0,'No':1} ).astype(int)


# In[17]:


cvd['Other_Cancer'] = cvd['Other_Cancer'].map ( {'Yes':0,'No':1} ).astype(int)


# In[18]:


cvd['Depression'] = cvd['Depression'].map ( {'Yes':0,'No':1} ).astype(int)


# In[19]:


cvd['Diabetes'] = cvd['Diabetes'].map ( {'Yes':0,'No':1,'No, pre-diabetes or borderline diabetes':2, 'Yes, but female told only during pregnancy':3} ).astype(float)


# In[20]:


cvd['Arthritis'] = cvd['Arthritis'].map ( {'Yes':0,'No':1} ).astype(int)


# In[21]:


cvd['Sex'] = cvd['Sex'].map ( {'Male':0,'Female':1} ).astype(int)


# In[22]:


cvd['Smoking_History'] = cvd['Smoking_History'].map ( {'Yes':0,'No':1} ).astype(float)


# In[23]:


cvd.head()


# In[24]:


cvd.describe()


# In[25]:


#Checking Data cleanliness using heatmaps


# In[26]:


sns.heatmap(cvd.isnull(),yticklabels=False)


# In[27]:


# Data Visualization


# In[28]:


#pie charts for each variable


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
data = cvd['General_Health'].value_counts()
labels = data.index
colors = sns.color_palette('pastel', len(labels))

plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
plt.axis('equal')  # Ensures that pie chart is circular
plt.show()


# In[30]:


data = cvd['Sex'].value_counts()
labels = data.index
colors = sns.color_palette('pastel', len(labels))

plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
plt.axis('equal')  # Ensures that pie chart is circular
plt.show()


# In[31]:


data = cvd['Diabetes'].value_counts()
labels = data.index
colors = sns.color_palette('pastel', len(labels))

plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
plt.axis('equal')  # Ensures that pie chart is circular
plt.show()


# In[32]:


data = cvd['Arthritis'].value_counts()
labels = data.index
colors = sns.color_palette('pastel', len(labels))

plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
plt.axis('equal')  # Ensures that pie chart is circular
plt.show()


# In[33]:


data = cvd['Age_Category'].value_counts()
labels = data.index
colors = sns.color_palette('pastel', len(labels))

plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
plt.axis('equal')  # Ensures that pie chart is circular
plt.show()


# In[34]:


#Correlation between the variables


# In[35]:


sns.heatmap(cvd.corr(),cbar=True, linewidth=2)


# In[37]:


corr = cvd.corr()
corr


# In[38]:


#correlation matrix of features


# In[39]:


# Set the figure size
plt.figure(figsize=(12, 10))

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={'shrink': .5})
plt.title('Correlation Matrix of the Features')
plt.show()


# In[40]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Split the data into input features (X) and target variable (y)
X = cvd.iloc[:, :-15]  # Select all columns except the last one
y = cvd.iloc[:, -15]   # Select the last column as the target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


# In[ ]:




