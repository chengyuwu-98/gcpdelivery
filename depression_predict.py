#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier



# In[2]:


# print all columns
pd.set_option('display.max_columns', None)
# color palette
colors = ['#ff0000', '#BB2F00', '#815700', '#4C7C00', '#09AA00']


# In[3]:


# keys
DASS_keys = {'Depression': [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
             'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
             'Stress': [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]}
# scores
DASS_scores = {'Depression': [(0, 10), (10, 14), (14, 21), (21, 28)],
             'Anxiety': [(0, 8), (8, 10), (10, 15), (15, 20)],
             'Stress': [(0, 15), (15, 19), (19, 26), (26, 34)]}


# In[5]:


df = pd.read_csv("/hpc/group/jilab/changxin/course/chengyu/IDS721/data.csv", sep='\t')
df.shape


# In[6]:


df.head(5)


# In[7]:


# validity check
df['wrongansw'] = 0
df['wrongansw'] = df['wrongansw'].where(df['VCL6']== 0, df['wrongansw'] + 1)
df['wrongansw'] = df['wrongansw'].where(df['VCL9']== 0, df['wrongansw'] + 1)
df['wrongansw'] = df['wrongansw'].where(df['VCL12']== 0, df['wrongansw'] + 1)

df.wrongansw.value_counts()


# In[8]:


df = df[df['wrongansw'].isin([2, 3])]
df = df.drop(columns='wrongansw')
df.head(3)


# In[9]:


df.shape


# In[10]:


vcls = []
for i in range(1, 17):
    vcls.append('VCL' + str(i))
    
df = df.drop(columns=vcls)
df.shape


# In[11]:


### keep questions related to depression 
fltr = []
for i in DASS_keys["Anxiety"]:
    fltr.append('Q' + str(i) + 'A')
    fltr.append('Q' + str(i) + 'I')
    fltr.append('Q' + str(i) + 'E')

for i in DASS_keys["Depression"]:
    fltr.append('Q' + str(i) + 'I')
    fltr.append('Q' + str(i) + 'E')

for i in DASS_keys["Stress"]:
    fltr.append('Q' + str(i) + 'A')
    fltr.append('Q' + str(i) + 'I')
    fltr.append('Q' + str(i) + 'E')

print(fltr)

# drop filters
df = df.drop(columns=fltr)
df.shape


# In[12]:


df.isnull().sum()
## drop major
df = df.drop(['major'], axis=1)


# In[13]:


df.duplicated().sum()


# In[14]:


numerical = df.select_dtypes('number').columns
print('Numerical Columns: ', df[numerical].columns)


# In[15]:


categorical = df.select_dtypes('object').columns
print('Categorical Columns: ', df[categorical].columns)
# n unique categories 
print(df[categorical].nunique())


# In[16]:


# answers columns
depr = []
for i in DASS_keys["Depression"]:
    depr.append('Q' + str(i) + 'A')

# filter answers
df_depr = df.filter(depr)


# In[19]:


# Question indedx start from 0 
df[depr] -= 1 
df.head(5)


# In[20]:


# calculate degree of depression
def scores(df):
    df["Scores"] = df[depr].sum(axis=1)
    return df
df = scores(df)


# In[21]:


# Discretize scores 
def append(df, string):
    conditions = [
    ((df['Scores'] >= DASS_scores[string][0][0]) & (df['Scores'] < DASS_scores[string][0][1])),
    ((df['Scores'] >= DASS_scores[string][1][0]) & (df['Scores'] < DASS_scores[string][1][1])),
    ((df['Scores'] >= DASS_scores[string][2][0]) & (df['Scores'] < DASS_scores[string][2][1])),
    ((df['Scores'] >= DASS_scores[string][3][0]) & (df['Scores'] < DASS_scores[string][3][1])),
    (((df['Scores'] >= DASS_scores[string][3][1])))
    ]
    values = ['Normal','Mild', 'Moderate', 'Severe', 'Extremely Severe']
    df['Cat'] = np.select(conditions, values)
    return df
    
df = append(df, 'Depression')
df.head(3)


# In[36]:


y = df['Cat']
X = df.drop(['Cat','Scores','screensize','uniquenetworklocation','country','introelapse','testelapse','surveyelapse','source'], axis=1)
X.head()


# In[25]:


y=y.map({'Normal':1,'Mild':2,'Moderate':3, 'Severe':4, 'Extremely Severe':5})


# In[39]:


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)


# In[ ]:


import pickle

## write model into model_pickle 
with open('scaler', 'wb') as f:
    pickle.dump(scaler, f)


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# In[41]:


# hyperparameters
n_estimators = [20, 40, 80]
criterion = ['gini', 'entropy']
max_depth = [4, 8, 10]

# stratisfied k cross-validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# create list of all combinations
param_all = list(product(n_estimators, criterion, max_depth))

# filter not allowed combinations
param_grid = [{'n_estimators' : [n_estimators], 'criterion': [criterion], 'max_depth' : [max_depth]} for n_estimators, criterion, max_depth in param_all]

# create model
model = RandomForestClassifier(random_state=1)

# define gridsearch
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
# run gridsearch
grid_res = grid.fit(X_train, y_train)

# mean and deviation of each feature combination
means = grid_res.cv_results_['mean_test_score']
stds = grid_res.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, grid_res.cv_results_['params']):
    print(f"{mean: .3f} (+/-{2 * std: .3f}) for {params}")


# In[42]:


print("Optimal")
print("Accuracy:", grid_res.best_score_)
print("Std:", grid_res.cv_results_['std_test_score'][grid_res.best_index_])
print("Hyperparams:", grid_res.best_params_)


# In[43]:


model = RandomForestClassifier(criterion=grid_res.best_params_['criterion'],
                               max_depth=grid_res.best_params_['max_depth'],
                               n_estimators=grid_res.best_params_['n_estimators'],
                               random_state=1)
model.fit(X_train, y_train)


# In[44]:


score = model.score(X_test, y_test)
print(f"Accuracy: {score}")


# In[47]:


import pickle

## write model into model_pickle 
with open('model', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:




