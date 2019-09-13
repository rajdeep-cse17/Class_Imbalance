import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = 'yeast6'
num_features = 8

df = pd.read_csv(dataset+'.csv',sep=', ')
print(df.head)

print('\n')
print(df.groupby('class').count())
print('\n')

le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class']) 

print(df.groupby('class').count())
print('\n')

df_train, df_test = train_test_split(df, train_size = 0.7, stratify = df['class'])

scaler = MinMaxScaler()
df_train = scaler.fit_transform(df_train)
df_train = pd.DataFrame(df_train)

scaler = MinMaxScaler()
df_test = scaler.fit_transform(df_test)
df = pd.DataFrame(df_test)

df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)

print(df_train.head)
print('\n')

print(df_test.head)
print('\n')

df_train.to_csv(dataset+'_train.csv', index=False)
df_test.to_csv(dataset+'_test.csv', index=False)
