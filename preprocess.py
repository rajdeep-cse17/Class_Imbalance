import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = 'segment0'
num_features = 10

df = pd.read_csv('./data/'+dataset+'.csv',sep=',')
print(df.head)

print('\n')
print(df.groupby('class').count())
print('\n')

le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class']) 

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

print(df.groupby('class').count())
print('\n')
df.fillna(0,inplace=True)
df_train, df_test = train_test_split(df, train_size = 0.7, stratify = df['class'])

scaler = MinMaxScaler()
df_train = scaler.fit_transform(df_train)
df_train = pd.DataFrame(df_train)

scaler = MinMaxScaler()
df_test = scaler.fit_transform(df_test)
df = pd.DataFrame(df_test)
# columns=[str(i) for i in range(len(df.columns))].append('class')
df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)

print(df_train.head)
print('\n')

print(df_test.head)
print('\n')

df_train.to_csv('./data/'+dataset+'_train.csv', index=False)
df_test.to_csv('./data/'+dataset+'_test.csv', index=False)
