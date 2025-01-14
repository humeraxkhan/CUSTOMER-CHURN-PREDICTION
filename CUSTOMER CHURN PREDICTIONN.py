#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv(r"C:\Users\DELL\Downloads\customer_churn.csv")
df.sample(5)


# In[5]:


df.drop('customerID',axis='columns',inplace=True)


# In[6]:


df.dtypes


# In[7]:


df.TotalCharges.values


# In[8]:


pd.to_numeric(df.TotalCharges,errors='coerce').isnull()


# In[9]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[10]:


df.shape


# In[11]:


df.iloc[488].TotalCharges


# In[12]:


df[df.TotalCharges!=' '].shape


# In[13]:


df1 = df[df.TotalCharges!=' ']
df1.shape


# In[14]:


df1.dtypes


# In[15]:


df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# In[17]:


df1.TotalCharges.values



# In[18]:


df1[df1.Churn=='No']


# In[19]:


tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]

plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[20]:


mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges      
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges      

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[21]:


def print_unique_col_values(df):
       for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}') 


# In[22]:


print_unique_col_values(df1)


# In[23]:


df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)


# In[24]:


print_unique_col_values(df1)


# In[25]:


yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)


# In[26]:


for col in df1:
    print(f'{col}: {df1[col].unique()}') 


# In[27]:


df1['gender'].replace({'Female':1,'Male':0},inplace=True)


# In[28]:


df1.gender.unique()


# In[29]:


df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
df2.columns


# In[30]:


df2.sample(5)


# In[31]:


df2.dtypes


# In[32]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])


# In[33]:


for col in df2:
    print(f'{col}: {df2[col].unique()}')


# In[34]:


X = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[35]:


X_train.shape



X_test.shape



# In[36]:


X_train[:10]


# In[37]:


len(X_train.columns)


# In[38]:


import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)


# In[39]:


model.evaluate(X_test, y_test)


# In[40]:


yp = model.predict(X_test)
yp[:5]


# In[41]:


y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[42]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[43]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[44]:


y_test.shape


# In[45]:


round((862+229)/(862+229+137+179),2)


# In[46]:


round(862/(862+179),2)


# In[ ]:




