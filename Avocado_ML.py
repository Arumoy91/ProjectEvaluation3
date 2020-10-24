#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# Loading the data file into Jupyter notebook

# In[3]:


df=pd.read_csv('Avacado.csv')


# In[4]:


df


# Trying to understand the Avocado dataset

# In[5]:


df.shape     #finding the data shape


# In[6]:


df.columns   # finding out the name of the columns


# In[7]:


df.info()  #finding the index,data-type,& memory information


# In[8]:


df.describe()


# In[9]:


df.isnull().sum() # trying to find out if there is any null value in any columns


# In[10]:


df.drop('Unnamed: 0',axis=1,inplace=True) #dropping unnecessary data from the dataset.


# In[11]:


df.head()


# In[12]:


df['Date']=pd.to_datetime(df['Date'])
df['Month']=df['Date'].apply(lambda x:x.month)
df['Day']=df['Date'].apply(lambda x:x.day)


# In[13]:


df.head(10)


# Data Visualization process

# In[18]:


sns.set(font_scale=1.6) 
from scipy.stats import norm
fig, ax = plt.subplots(figsize=(16, 10))
sns.distplot(a=df.AveragePrice, kde=False, fit=norm)   


# In[19]:


plt.figure(figsize=(20,10))
sns.lineplot(x="Month", y="AveragePrice", hue='type', data=df)
plt.show()


# In[21]:


region_list=list(df.region.unique())
average_price=[]

for i in region_list:
    x=df[df.region==i]
    region_average=sum(x.AveragePrice)/len(x)
    average_price.append(region_average)

df1=pd.DataFrame({'region_list':region_list,'average_price':average_price})
new_index=df1.average_price.sort_values(ascending=False).index.values
sorted_data=df1.reindex(new_index)

plt.figure(figsize=(30,10))
ax=sns.barplot(x=sorted_data.region_list,y=sorted_data.average_price)

plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.title('Average Price of Avocado According to Region')


# In[22]:


filter1=df.region!='TotalUS'
df1=df[filter1]

region_list=list(df1.region.unique())
average_total_volume=[]

for i in region_list:
    x=df1[df1.region==i]
    average_total_volume.append(sum(x['Total Volume'])/len(x))
df3=pd.DataFrame({'region_list':region_list,'average_total_volume':average_total_volume})

new_index=df3.average_total_volume.sort_values(ascending=False).index.values
sorted_data1=df3.reindex(new_index)

plt.figure(figsize=(25,10))
ax=sns.barplot(x=sorted_data1.region_list,y=sorted_data1.average_total_volume)

plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average of Total Volume')
plt.title('Average of Total Volume According to Region')


# In[25]:


plt.figure(figsize=(18,6))
sns.heatmap(df.corr(),annot=True) # Finding how datasets are correlated wth each other


# In[26]:


df['region'].nunique()


# In[27]:


df['type'].nunique()


# In[28]:


df_data=pd.get_dummies(df.drop(['region','Date'],axis=1),drop_first=True)


# In[29]:


df_data.head(10)


# Finding out the best suitable model for the datasets

# In[31]:


X=df_data.iloc[:,1:14]


# In[32]:


y=df_data['AveragePrice']


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.12,random_state=42)  


# In[35]:


from sklearn.linear_model import LinearRegression


# In[36]:


lr=LinearRegression()


# In[37]:


lr.fit(X_train,y_train)


# In[38]:


pred=lr.predict(X_test)


# In[56]:


pred


# In[39]:


from sklearn import metrics


# In[40]:


print('MAE:', metrics.mean_absolute_error(y_test, pred))


# In[41]:


print('MSE:', metrics.mean_squared_error(y_test, pred))


# In[42]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[43]:


from sklearn.tree import DecisionTreeRegressor


# In[44]:


dr=DecisionTreeRegressor()


# In[45]:


dr.fit(X_train,y_train)


# In[47]:


pred=dr.predict(X_test)


# In[55]:


pred


# In[48]:


print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[49]:


from sklearn.ensemble import RandomForestRegressor  


# In[50]:


rder = RandomForestRegressor()


# In[51]:


rder.fit(X_train,y_train)


# In[53]:


pred=rder.predict(X_test)


# In[54]:


pred


# In[57]:


print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

Conclusion: From the datasets it assist me to obtain the actionable insights about the data and also which model to choose in a datasets with a normal process flow. it also help me where to use LinearRegression, Decision Tree, and additional required models to find out the predictions of the datasets in the best possible way.  