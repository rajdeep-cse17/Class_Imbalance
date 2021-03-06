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
from sklearn.metrics import roc_auc_score
import pickle
from math import ceil
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import dunn_sklearn
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']

dataset = 'SPECTF'
num_features = 9
minority_class_label = 1
majority_class_label = 0
max_iter = 10
num_clusters = 40

maxauc=0.0
df = pd.read_csv('./data/' + dataset + '_train.csv',encoding='utf8')
df.astype({'class': 'int32'})


def get_class_samples(label):
	df_class = df.loc[df['class'] == label]
	df_class.drop(columns='class', inplace=True)

	return df_class




def get_cluster_samples(df_maj, label):
	df_cluster = df_maj.loc[df_maj['cluster'] == label]
	return df_cluster


def cluster_class(df_class, num_clusters=40, max_iter=300):
	kmeans = KMeans(n_clusters=num_clusters,
					random_state=0, max_iter=max_iter).fit(df_class)
	cluster_centers = kmeans.cluster_centers_
	df_class['cluster'] = kmeans.predict(df_class)
	df_class['entropy'] = [0] * len(df_class)
	pred = pd.DataFrame(df_class['cluster']) 
	df_classwithoutentropy=df_class.drop(columns=['entropy','cluster'],axis=1)
	"""c=range(0,num_clusters)
	for diameter_method in DIAMETER_METHODS:
		for cdist_method in CLUSTER_DISTANCE_METHODS:
			distances=pairwise_distances(df_classwithoutentropy)
			dund = dunn_sklearn.dunn(c, distances, diameter_method, cdist_method)
			dunk = dunn_sklearn.dunn(pred, distances, diameter_method, cdist_method)
			print(diameter_method, cdist_method, dund, dunk)
	print(diameter_method, cdist_method,dund,dunk)"""
	return cluster_centers


def create_train_set(X_maj, X_min, maj_label, min_label):
	cols = []
	for i in X_maj.columns:
		cols.append(str(i))
	X_maj.columns = cols
	X_maj['class'] = [maj_label] * len(X_maj)

	cols = []
	for i in X_min.columns:
		cols.append(str(i))
	X_min.columns = cols
	X_min['class'] = [min_label] * len(X_min)

	X = X_min.append(X_maj, ignore_index=True, sort=True)

	y = X['class']
	
	X.drop(columns='class', inplace=True)
	X_maj.drop(columns='class', inplace=True)
	sm = SMOTE(random_state=42)
	X_min.drop(columns='class', inplace=True)
	print(X_maj.shape)
	print(X_min.shape)
	X, y = sm.fit_resample(X, y)
	return X, y

def silhouette(X):

	for n_clusters in range(2,100):
		# Create a subplot with 1 row and 2 columns
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(18, 7)

		# The 1st subplot is the silhouette plot
		# The silhouette coefficient can range from -1, 1 but in this example all
		# lie within [-0.1, 1]
		ax1.set_xlim([-0.1, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

		# Initialize the clusterer with n_clusters value and a random generator
		# seed of 10 for reproducibility.
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(X)

		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# clusters
		silhouette_avg = silhouette_score(X, cluster_labels)
		print("For n_clusters =", n_clusters,
		  "The average silhouette_score is :", silhouette_avg)

		# Compute the silhouette scores for each sample
		sample_silhouette_values = silhouette_samples(X, cluster_labels)

		y_lower = 10
		for i in range(n_clusters):
		# Aggregate the silhouette scores for samples belonging to
		# cluster i, and sort them
			ith_cluster_silhouette_values=sample_silhouette_values[cluster_labels == i]


			ith_cluster_silhouette_values.sort()
	 
			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.nipy_spectral(float(i) / n_clusters)
			ax1.fill_betweenx(np.arange(y_lower, y_upper),
							  0, ith_cluster_silhouette_values,
							  facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		ax1.set_title("The silhouette plot for the various clusters.")
		ax1.set_xlabel("The silhouette coefficient values")
		ax1.set_ylabel("Cluster label")

		# The vertical line for average silhouette score of all the values
		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax1.set_yticks([])  # Clear the yaxis labels / ticks
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

		# 2nd Plot showing the actual clusters formed
		colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
		ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

		# Labeling the clusters
		centers = clusterer.cluster_centers_
		# Draw white circles at cluster centersS
		ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
				c="white", alpha=1, s=200, edgecolor='k')

		for i, c in enumerate(centers):
			ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')


def train(X, y):

	#clf = SVC(gamma='auto',kernel='rbf',probability=True,
	 #class_weight='balanced')#RandomForestClassifier(n_estimators=100,
	# max_depth=2,
	# class_weight='balanced')SVC(gamma='auto',kernel='linear',probability=True)
	clf = MLPClassifier(
	   hidden_layer_sizes=(19))
	clf.fit(X, y)


	y_pred = clf.predict(X)
	y_prob= clf.predict_proba(X)[:,1]

	# print('\n')
	# print('train_set : '+str(clf.score(X, y)))
	# print('train_set_classwise : '+str(precision_recall_fscore_support(y, y_pred)))
	# print('train_set_avg : '+str(precision_recall_fscore_support(y, y_pred,
	# average='macro')))

	print('\n')

	print('train_set')
	print(classification_report(y, y_pred))
	print('auc score	:' + str(roc_auc_score(y,y_prob)))

	print('\n')

	X_global = df.drop(columns='class')
	y_global = df['class']
	y_global_pred = clf.predict(X_global)
	y_global_prob= clf.predict_proba(X_global)[:,1]
	# print('global_train_set : '+str(clf.score(X_global, y_global)))
	# print('global_train_set_classwise : '+str(precision_recall_fscore_support(y_global, y_global_pred)))
	# print('global_train_set_avg :
	# '+str(precision_recall_fscore_support(y_global, y_global_pred,
	# average='macro')))

	print('global_train_set')
	print(classification_report(y_global, y_global_pred))
	precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
	print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
	print('f score	:' + str(fscore))
	# print('\n')

	print(
		'_______________________________________________________________________________________________________________________________')
	global maxauc
	if fscore>maxauc:
		print('sadffsgfsf')
		maxauc=fscore
		f = open('model.sav', 'wb')
		pickle.dump(clf, f)

	return clf,roc_auc_score(y_global,y_global_prob),fscore


def update_cluster_centers(df_maj, cluster_centers, clf):
	pred_maj = clf.predict_proba(df_maj.drop(columns=['cluster', 'entropy']))
	entropy = (-pred_maj * np.log2(pred_maj)).sum(axis=1)
	df_maj['entropy'] = entropy

	for curr_cluster in range(len(cluster_centers)):

		w_denom = 0
		w_num = [0] * (len(df_maj.iloc[0]) - 2)
					 #eliminate columns cluster and entropy

		for j in range(len(df_maj)):

			if df_maj.iloc[j]['cluster'] == curr_cluster:
				point_coord = np.array(df_maj.iloc[j, :-2])
				entropy = df_maj.iloc[j]['entropy']

				if(clf.predict([point_coord]) == [majority_class_label]):

					w_num += 1.5 * point_coord * entropy
					w_denom += 1.5 * entropy

				else:

					w_num += point_coord * entropy
					w_denom += entropy

		cluster_centers[curr_cluster] = w_num / \
			w_denom  # [num/w_denom for num in w_num]

	return cluster_centers


def topn_features(df_maj, clf,n):
	pred_maj = clf.predict_proba(df_maj.drop(columns=['cluster', 'entropy']))
	entropy = pred_maj[:,1]
	parameters=norm.fit(entropy)
	fitted_pdf=norm.pdf(entropy,loc = parameters[0],scale = parameters[1])
	df_maj['entropy'] = fitted_pdf
	df_maj.sort_values(by=['entropy'], ascending=False,inplace=True)
	[length, breadth] = df.shape

	d_obj = pd.DataFrame()

	for i in range(num_clusters):
		df_clust = get_cluster_samples(df_maj, i)		
		d_topn=df_clust.head(ceil(n*df_clust.shape[0]))
		d_obj=d_obj.append(d_topn.mean(axis=0),ignore_index=True)
	return d_obj
def test():

	df = pd.read_csv('./data/' + dataset + '_test.csv',encoding='utf8')
	df.astype({'class': 'int32'})
	f = open('model.sav', 'rb')
	model = pickle.load(f)	
	X_global = df.drop(columns='class')
	y_global = df['class']
	y_global_pred = model.predict(X_global)
	y_global_prob= model.predict_proba(X_global)[:,1]
	# print('test_set : '+str(clf.score(X_global, y_global)))
	# print('test_set_classwise : '+str(precision_recall_fscore_support(y_global, y_global_pred)))
	# print('test_set_avg : '+str(precision_recall_fscore_support(y_global,
	# y_global_pred, average='macro')))

	print('test set')
	print(classification_report(y_global, y_global_pred))
	precision,recall,fscore,support=score(y_global,y_global_pred,average='macro')
	print('auc score	:' + str(roc_auc_score(y_global,y_global_prob)))
	print('f score	:' + str(fscore))

def getauc(k):
	df_maj = get_class_samples(majority_class_label)
	df_min = get_class_samples(minority_class_label)
	global num_clusters
	num_clusters=int(k*df_min.shape[0])
	cluster_centers = cluster_class(df_maj, num_clusters, max_iter)

	df_x = pd.DataFrame(cluster_centers)
	
	n=0.75
	prev=0.0
	curr=100.0
	auc=[]
	continue_loop=True
	count = 0
	while continue_loop:
		count =count +1
	
		print("a")
		X, y = create_train_set(
			df_x, df_min, majority_class_label, minority_class_label)

		clf,curr,fscore = train(X, y)
		auc.append(fscore)
		df_x = topn_features(df_maj, clf,n)

		df_x = df_x.drop(columns=['cluster', 'entropy'])
		if count>10 and maxauc>np.max(auc[-10:]):
				continue_loop=False
	print('\n.....................Testing...............')

	test()

def main():


	from sklearn.cluster import KMeans
	from sklearn.datasets import make_blobs

	from yellowbrick.cluster import KElbowVisualizer
	df_maj = get_class_samples(majority_class_label)
	df_min = get_class_samples(minority_class_label)
# Generate synthetic dataset with 8 random clusters

# Instantiate the clustering model and visualizer
	"""model = KMeans()
	visualizer = KElbowVisualizer(model, k=(1,100))

	visualizer.fit(df_maj)		# Fit the data to the visualizer
	visualizer.show()		# Finalize and render the figure"""
	"""X, y = create_train_set(
			df_maj, df_min, majority_class_label, minority_class_label)"""

	#silhouette(df_maj.to_numpy())
	global maxauc
	scores={}
	maxk=0
	for i in np.arange(1,2.1,0.1):
		maxauc=0.0
		print(i)
		if int(i*df_min.shape[0])>df_maj.shape[0]:
			break
		print(i)
		print('k='+str(i))
		getauc(i)
		if i==1.0:
			maxk=1.0
		elif maxauc>np.max(list(scores.values())):
			maxk=i
		scores[i]=maxauc
	print('k='+str(maxk))
	maxauc=0.0
	getauc(maxk)
	print('Obtained using k='+str(maxk))
	"""f = open('model.sav', 'wb')
	pickle.dump(clf, f)"""

main()
