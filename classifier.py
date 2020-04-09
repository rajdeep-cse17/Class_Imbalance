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
<<<<<<< HEAD
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
import pickle
import csv


dataset = 'SPECTF'
=======
import pickle


dataset = 'poker-8-9_vs_5'
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785
num_features = 8
minority_class_label = 1
majority_class_label = 0

df = pd.read_csv('./data/'+dataset+'_train.csv')
df.astype({'class': 'int32'})

X = df.drop(columns = 'class')
y = df['class']

#clf = SVC(gamma='auto',kernel='linear',probability=True, class_weight='balanced')#RandomForestClassifier(n_estimators=100, max_depth=2, class_weight='balanced')SVC(gamma='auto',kernel='linear',probability=True)
<<<<<<< HEAD
clf = MLPClassifier(hidden_layer_sizes=(19))
=======
clf =  MLPClassifier(hidden_layer_sizes=(50,100,100,100,100,70,50,30,20,10,8 ))
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785
clf.fit(X, y)	

y_pred = clf.predict(X)

<<<<<<< HEAD

print('train set')
print(classification_report(y, y_pred))
precision,recall,fscore,support=score(y,y_pred,average='macro')
y_prob= clf.predict_proba(X)[:,1]
print('auc score:' + str(roc_auc_score(y,y_prob)))
print('f score	:' + str(fscore))
trs=str(roc_auc_score(y,y_prob))
trsf=str(fscore)
=======
print('train set')
print(classification_report(y, y_pred))


>>>>>>> 572e006689cde55bc436972abfbe64ab40614785

df = pd.read_csv('./data/'+dataset+'_test.csv')
df.astype({'class': 'int32'})

X_global = df.drop(columns='class')
y_global = df['class']
y_global_pred = clf.predict(X_global)

print('test set')
print(classification_report(y_global, y_global_pred))
<<<<<<< HEAD
precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
y_global_prob= clf.predict_proba(X_global)[:,1]
print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
print('f score	:' + str(fscore))
tests=str(roc_auc_score(y_global,y_global_prob))
testsf=str(fscore)

=======

'''
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785
print('\n Proposed Method \n')
f = open('model.sav', 'rb')
model = pickle.load(f)	
y_global_pred = model.predict(X_global)
print(classification_report(y_global, y_global_pred))
<<<<<<< HEAD
precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
y_global_prob= model.predict_proba(X_global)[:,1]
print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
print('f score	:' + str(fscore))
propm=str(roc_auc_score(y_global,y_global_prob))
propmf=str(fscore)

=======
'''
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785
print('\n Prototype Generation using KMeans \n')
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))
<<<<<<< HEAD
precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
y_global_prob= clf.predict_proba(X_global)[:,1]
print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
print('f score	:' + str(fscore))
protkmeansf=str(fscore)
protkmeans=str(roc_auc_score(y_global,y_global_prob))
=======
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785

print('\n Random Undersampling \n')
from imblearn.under_sampling import RandomUnderSampler
cc = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))
<<<<<<< HEAD
precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
y_global_prob= clf.predict_proba(X_global)[:,1]
print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
print('f score	:' + str(fscore))
randundf=str(fscore)
randund=str(roc_auc_score(y_global,y_global_prob))
=======
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785


print('\n Near Miss Undersampling \n')
from imblearn.under_sampling import NearMiss
cc = NearMiss(version=1)
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))
<<<<<<< HEAD
precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
y_global_prob= clf.predict_proba(X_global)[:,1]
print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
print('f score	:' + str(fscore))
nearmissf=str(fscore)
nearmiss=str(roc_auc_score(y_global,y_global_prob))
=======
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785


print('\n EditedNearestNeighbours Cleaning Undersampling \n')
from imblearn.under_sampling import EditedNearestNeighbours
cc = EditedNearestNeighbours()
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))
<<<<<<< HEAD
precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
y_global_prob= clf.predict_proba(X_global)[:,1]
print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
print('f score	:' + str(fscore))
editnnf=str(fscore)
editnn=str(roc_auc_score(y_global,y_global_prob))
=======
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785

print('\n CondensedNearestNeighbourUndersampling \n')
from imblearn.under_sampling import CondensedNearestNeighbour
cc = CondensedNearestNeighbour()
X_resampled, y_resampled = cc.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
print(classification_report(y_global, y_global_pred))
<<<<<<< HEAD
precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
y_global_prob= clf.predict_proba(X_global)[:,1]
print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
print('f score	:' + str(fscore))
condnnf=str(fscore)
condnn=str(roc_auc_score(y_global,y_global_prob))
=======

>>>>>>> 572e006689cde55bc436972abfbe64ab40614785

print('\n SMOTE \n')
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
clf.fit(X_resampled, y_resampled)	
y_global_pred = clf.predict(X_global)
<<<<<<< HEAD
print(classification_report(y_global, y_global_pred))
precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
y_global_prob= clf.predict_proba(X_global)[:,1]
print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
print('f score	:' + str(fscore))
smotef=str(fscore)
smote=str(roc_auc_score(y_global,y_global_prob))

with open('results10.csv', mode='a') as results:
	results = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	results.writerow([dataset,'auc score', trs, tests, propm, protkmeans, randund, nearmiss, editnn, condnn, smote])	
	results.writerow(['','f1 score', trsf, testsf, propmf, protkmeansf, randundf, nearmissf, editnnf, condnnf, smotef])	
=======
print(classification_report(y_global, y_global_pred))
>>>>>>> 572e006689cde55bc436972abfbe64ab40614785
