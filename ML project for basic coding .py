#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load dataset


# In[3]:


df=pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


print(df.dtypes)


# In[9]:


df.dropna()


# # Check for missing values

# In[10]:


print("Missing Values Before Imputation:")


# In[11]:


df.isnull().sum()


# In[12]:


# Impute missing values in numerical columns with median
numerical_cols = ['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 
                  'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)


# In[13]:


# Verify no missing values remain
print("\nMissing Values After Imputation:")


# In[14]:


df.isnull().sum()


# # Convert Categorical Features into Numerical Using Encoding

# In[15]:


# Perform one-hot encoding on the 'gender' column
df_encoded = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Display the first few rows of the encoded dataset
print("Encoded Dataset (First 5 Rows):")
print(df_encoded.head())


# # Normalize/Standardize the Numerical Features

# In[16]:


# Identify numerical columns (excluding the target 'is_patient' and categorical 'gender_Male')
numerical_cols = ['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 
                  'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']

# Initialize the scaler
scaler = StandardScaler()

# Standardize numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the first few rows of the standardized dataset
print("Standardized Dataset (First 5 Rows):")
print(df.head())


# # Visualize Outliers Using Boxplots and Remove Them

# In[17]:


# Numerical columns for outlier detection
numerical_cols = ['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 
                  'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']

# Create boxplots for numerical columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('boxplots_before_outlier_removal.png')
plt.close()

# Remove outliers using IQR method
def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Apply outlier removal
df_clean = remove_outliers(df, numerical_cols)

# Create boxplots after outlier removal
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df_clean[col])
    plt.title(f'Boxplot of {col} (After Outlier Removal)')
plt.tight_layout()
plt.savefig('boxplots_after_outlier_removal.png')
plt.close()

# Display the shape of the dataset before and after outlier removal
print("Dataset Shape Before Outlier Removal:", df.shape)
print("Dataset Shape After Outlier Removal:", df_clean.shape)

# Save the cleaned dataset
df_clean.to_csv("Indian Liver Patient Dataset (ILPD).csv", index=False)


# In[ ]:




