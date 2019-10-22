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
from imblearn.over_sampling import SMOTE
import pickle


dataset = 'poker-8-9_vs_5'
num_features = 9
minority_class_label = 1
majority_class_label = 0
max_iter = 10
num_clusters= 40

df = pd.read_csv('./data/'+dataset+'_train.csv')
df.astype({'class': 'int32'})

def get_class_samples(label):
	df_class = df.loc[df['class']==label]
	df_class.drop(columns='class',inplace=True)

	return df_class

def cluster_class(df_class, num_clusters=40, max_iter=300):
	kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=max_iter).fit(df_class)
	cluster_centers = kmeans.cluster_centers_
	df_class['cluster'] = kmeans.predict(df_class) 	
	df_class['entropy'] = [0]*len(df_class)

	return cluster_centers

def create_train_set(X_maj, X_min, maj_label, min_label):
	cols=[]
	for i in X_maj.columns:
		cols.append(str(i))
	X_maj.columns = cols
	X_maj['class'] = [maj_label]*len(X_maj)

	cols=[]
	for i in X_min.columns:
		cols.append(str(i))
	X_min.columns = cols
	X_min['class'] = [min_label]*len(X_min)

	X = X_min.append(X_maj, ignore_index=True, sort=True)

	y = X['class']
	X.drop(columns='class', inplace = True)

	sm = SMOTE(random_state=42)
	X_res, y_res = sm.fit_resample(X, y)

	return X_res, y_res

def train(X, y, maj_num, min_num):


	#clf = SVC(gamma='auto',kernel='linear',probability=True, class_weight='balanced')#RandomForestClassifier(n_estimators=100, max_depth=2, class_weight='balanced')SVC(gamma='auto',kernel='linear',probability=True)
	clf =  MLPClassifier(hidden_layer_sizes=(30,50,100,100,70,50,20,10,8 ))
	clf.fit(X, y)
	
	y_pred = clf.predict(X)

	#print('\n')
	#print('train_set : '+str(clf.score(X, y)))
	#print('train_set_classwise : '+str(precision_recall_fscore_support(y, y_pred)))
	#print('train_set_avg : '+str(precision_recall_fscore_support(y, y_pred, average='macro')))	

	print('\n')

	print('train_set')
	print(classification_report(y,y_pred))

	print('\n')

	X_global = df.drop(columns='class')
	y_global = df['class']
	y_global_pred = clf.predict(X_global)

	#print('global_train_set : '+str(clf.score(X_global, y_global)))
	#print('global_train_set_classwise : '+str(precision_recall_fscore_support(y_global, y_global_pred)))
	#print('global_train_set_avg : '+str(precision_recall_fscore_support(y_global, y_global_pred, average='macro')))

	print('global_train_set')
	print(classification_report(y_global, y_global_pred))

	#print('\n')
	
	print('_______________________________________________________________________________________________________________________________')

	return clf	


def update_cluster_centers(df_maj, cluster_centers, clf):
	pred_maj = clf.predict_proba(df_maj.drop(columns=['cluster','entropy']))
	entropy = (-pred_maj*np.log2(pred_maj)).sum(axis=1)
	df_maj['entropy'] = entropy

	for curr_cluster in range(len(cluster_centers)):
		
		w_denom = 0
		w_num = [0]*(len(df_maj.iloc[0])-2) #eliminate columns cluster and entropy
		
		for j in range(len(df_maj)):

			if df_maj.iloc[j]['cluster'] == curr_cluster:
				point_coord = np.array(df_maj.iloc[j,:-2])
				entropy = df_maj.iloc[j]['entropy']
				
				if( clf.predict([point_coord]) == [majority_class_label]):
					
					w_num += 1.5 * point_coord*entropy
					w_denom += 1.5 * entropy
				
				else:
					
					w_num += point_coord*entropy
					w_denom += entropy	

		cluster_centers[curr_cluster] = w_num/w_denom #[num/w_denom for num in w_num]

	return cluster_centers		

def test(clf):

	df = pd.read_csv('./data/'+dataset+'_test.csv')
	df.astype({'class': 'int32'})

	X_global = df.drop(columns='class')
	y_global = df['class']
	y_global_pred = clf.predict(X_global)

	#print('test_set : '+str(clf.score(X_global, y_global)))
	#print('test_set_classwise : '+str(precision_recall_fscore_support(y_global, y_global_pred)))
	#print('test_set_avg : '+str(precision_recall_fscore_support(y_global, y_global_pred, average='macro')))

	print('test set')
	print(classification_report(y_global, y_global_pred))


def main():

	df_maj = get_class_samples(majority_class_label)
	df_min = get_class_samples(minority_class_label)

	cluster_centers = cluster_class(df_maj, num_clusters, max_iter)

	
	for i in range(max_iter):
		X, y = create_train_set( pd.DataFrame(cluster_centers), df_min, majority_class_label, minority_class_label)

		clf = train(X, y, len(cluster_centers), len(df_min))

		cluster_centers = update_cluster_centers(df_maj, cluster_centers, clf)

	print('\n.....................Testing...............')
	
	test(clf)	

	f = open('model.sav','wb')
	pickle.dump(clf,f)

main()
