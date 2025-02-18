#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install kagglehub')


# In[5]:


get_ipython().system('pip install --upgrade xgboost')


# In[7]:


import kagglehub

path = kagglehub.dataset_download("spscientist/students-performance-in-exams")

print("Path to dataset files:", path)


# In[10]:


import os

 # Set the directory path here
files = os.listdir(path)

print("Files in dataset directory:", files)


# In[17]:


import pandas as pd

csv_file_path = os.path.join(path, 'StudentsPerformance.csv')

data = pd.read_csv(csv_file_path)


# In[18]:


print(data.head())


# In[19]:


data.shape


# In[20]:


data.info()


# In[21]:


data.describe()


# In[22]:


#No missing variables
data.isnull().sum()


# In[25]:


import seaborn as sns

print('Histogram of race/ethnicity')
sns.displot(data=data['race/ethnicity'])


# In[26]:


print('Histogram of math score')
sns.displot(data=data['math score'])


# In[27]:


#Boxplot for math scores

import matplotlib.pyplot as plt

sns.boxplot(x='math score', data=data)

plt.show()



# In[28]:


print('Histogram of reading score')
sns.displot(data=data['reading score'])


# In[29]:


#Boxplot for reading scores


sns.boxplot(x='reading score', data=data)

plt.show()


# In[32]:


import pandas as pd

numeric_columns = data.select_dtypes(include=['number']).columns

mask = pd.Series([True] * len(data))

for column in numeric_columns:
   Q1 = data[column].quantile(0.25)
   Q3 = data[column].quantile(0.75)
   IQR = Q3 - Q1

   column_mask = (data[column] >= (Q1 - 1.5 * IQR)) & (data[column] <= (Q3 + 1.5 * IQR))
   
   mask &= column_mask

df_no_outliers = data[mask]

print(df_no_outliers)


# In[33]:


print('Histogram of writing score')
sns.displot(data=df_no_outliers['writing score'])


# In[34]:


#Boxplot for writing scores


sns.boxplot(x='writing score', data=df_no_outliers)

plt.show()


# In[35]:


num_vars = ['reading score', 'writing score', 'math score']
df_cat = df_no_outliers.drop(num_vars, axis = 1)
for col in list(df_cat):
  print("\n", data[col].value_counts(dropna=False).to_string())


# In[36]:


data_filtered = df_no_outliers

data_filtered.head()


# In[37]:


print(data_filtered.describe())


# In[38]:


# Use One Hot Encoding to turn categorical variables with 3+ values into dummy variables: 

df_encod = pd.get_dummies(data_filtered, columns=['parental level of education'])

df_encod[df_encod.select_dtypes(bool).columns] = df_encod.select_dtypes(bool).apply(lambda x: x.astype(int))


df_encod.describe()

df_encod.shape


# In[39]:


dataone = df_encod

dataone.shape


# In[42]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataone['race/ethnicity_num'] = le.fit_transform(df_encod['race/ethnicity'])

print(dataone['race/ethnicity_num'].describe())

dataone.describe()


# In[43]:


df_filt = dataone.drop(['race/ethnicity'], axis = 1)
df_filt.shape


# In[49]:


#Splitting Data and only keeping reading writing and math scores for assignment


X = df_filt.drop(columns = ["race/ethnicity_num", "gender", "lunch", "test preparation course", 
                            "parental level of education_associate's degree",
                            "parental level of education_bachelor's degree",
                            "parental level of education_high school",
                            "parental level of education_master's degree",
                            "parental level of education_some college",
                            "parental level of education_some high school"])

y = df_filt['race/ethnicity_num'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 88, stratify=y)

print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)



# In[53]:


Corrlelationmatrix = df_filt.drop(columns=["parental level of education_associate's degree",
                          "parental level of education_bachelor's degree",
                          "parental level of education_high school",
                          "parental level of education_master's degree",
                          "parental level of education_some college",
                          "parental level of education_some high school"
                                           "Gender"])





# In[59]:


correlation_matrix = df_filt.drop(columns=[
    "parental level of education_associate's degree",
    "parental level of education_bachelor's degree",
    "parental level of education_high school",
    "parental level of education_master's degree",
    "parental level of education_some college",
    "parental level of education_some high school",
    "gender",
    "lunch",
    "test preparation course"# Added missing comma here
])

# Assuming X and y are already defined, and X is the feature set
data_clean = X.copy()
data_clean["race/ethnicity_num"] = y  # Ensure y is correctly defined

# Calculate the correlation matrix
corr_matrix = correlation_matrix.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap (All Features)")
plt.show()


# In[60]:


X.hist(bins=20, figsize=(15, 15))


# In[63]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb


param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'alpha': [0, 0.1, 1],
    'lambda': [0, 0.1, 1]
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")


# In[64]:


#XGBoost
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor




model1 = XGBClassifier(
    alpha=0.1,
    reg_lambda= 0,
    learning_rate= 0.01,
    n_estimators=100,    
    max_depth=3
)

model1.fit(X_train, y_train)

predictions1= model1.predict(X_train)
print('\nTarget on Train:\n', predictions1)

accuracy_train1 = accuracy_score(y_train, predictions1) 
print('\nTraining Accuracy:', accuracy_train1)

predictions_test1= model1.predict(X_test)
print('\n Target on Test:\n', predictions_test1)

accuracy_test1= accuracy_score(y_test, predictions_test1)
print('\n Testing Accuracy:', accuracy_test1)


# In[65]:


#Confusion Matrix

from sklearn.metrics import confusion_matrix

y_pred = model1.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[68]:


import pickle

save_path = "/Users/evelyncasillas/PycharmProjects/Midterm/model1.pkl"

with open(save_path, "wb") as file:
    pickle.dump(model1, file)

print(f"\nModel saved at: {save_path}")


# In[70]:


file_path = os.path.abspath("../model1.pkl")
with open(file_path, "wb") as file:
    pickle.dump(model1, file)

print(f"\nModel saved at: {file_path}")




