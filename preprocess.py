import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

<<<<<<< HEAD
dataset = 'SPECTF'
=======
dataset = 'poker-8-9_vs_5'
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785
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
<<<<<<< HEAD
df.fillna(0,inplace=True)
df_train, df_test = train_test_split(df, train_size = 0.7, stratify = df['class'])
=======

df_train, df_test = train_test_split(df, train_size = 0.6, stratify = df['class'])
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785

scaler = MinMaxScaler()
df_train = scaler.fit_transform(df_train)
df_train = pd.DataFrame(df_train)

scaler = MinMaxScaler()
df_test = scaler.fit_transform(df_test)
df = pd.DataFrame(df_test)
<<<<<<< HEAD
# columns=[str(i) for i in range(len(df.columns))].append('class')
=======

>>>>>>> 572e006689cde55bc436972abfbe64ab40614785
df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)

print(df_train.head)
print('\n')

print(df_test.head)
print('\n')

df_train.to_csv('./data/'+dataset+'_train.csv', index=False)
df_test.to_csv('./data/'+dataset+'_test.csv', index=False)
