import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle


dataset = 'poker-8-9_vs_5'
num_features = 8
minority_class_label = 1
majority_class_label = 0

df = pd.read_csv('./data/'+dataset+'_train.csv')
df.astype({'class': 'int32'})

X = df.drop(columns = 'class')
y = df['class']

#clf = SVC(gamma='auto',kernel='linear',probability=True, class_weight='balanced')#RandomForestClassifier(n_estimators=100, max_depth=2, class_weight='balanced')SVC(gamma='auto',kernel='linear',probability=True)
clf =  MLPClassifier(hidden_layer_sizes=(50,100,100,100,100,70,50,30,20,10,8 ))
clf.fit(X, y)	

y_pred = clf.predict(X)

print('train set')
print(classification_report(y, y_pred))



df = pd.read_csv('./data/'+dataset+'_test.csv')
df.astype({'class': 'int32'})

X_global = df.drop(columns='class')
y_global = df['class']
y_global_pred = clf.predict(X_global)

print('test set')
print(classification_report(y_global, y_global_pred))

'''
print('\n Proposed Method \n')
f = open('model.sav', 'rb')
model = pickle.load(f)	
y_global_pred = model.predict(X_global)
print(classification_report(y_global, y_global_pred))
'''
print('\n Prototype Generation using KMeans \n')
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))

print('\n Random Undersampling \n')
from imblearn.under_sampling import RandomUnderSampler
cc = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))


print('\n Near Miss Undersampling \n')
from imblearn.under_sampling import NearMiss
cc = NearMiss(version=1)
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))


print('\n EditedNearestNeighbours Cleaning Undersampling \n')
from imblearn.under_sampling import EditedNearestNeighbours
cc = EditedNearestNeighbours()
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))

print('\n CondensedNearestNeighbourUndersampling \n')
from imblearn.under_sampling import CondensedNearestNeighbour
cc = CondensedNearestNeighbour()
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))


print('\n SMOTE \n')
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))