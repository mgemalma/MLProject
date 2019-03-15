import numpy as np
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#from sklearn.model_selection import cross_val_score
import os
from joblib import dump,load
from sklearn.neural_network import MLPClassifier
#import scikitplot as skplt
#os.path.isfile('./file.txt')
from sklearn.metrics.pairwise import check_pairwise_arrays
from scipy.linalg import cholesky
from sklearn.kernel_approximation import Nystroem
#from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler


def anova_kernel(X, Y=None, gamma=None, p=1):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1. / X.shape[1]

    diff = X[:, None, :] - Y[None, :, :]
    diff **= 2
    diff *= -gamma
    np.exp(diff, out=diff)
    K = diff.sum(axis=2)
    K **= p
    return K


with open('data_v1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = list(csv.reader(csv_file))


ndata = np.array(data)


abs_time = np.array( ndata[:,1])
abs_time[abs_time=='']='0'
abs_time = np.asarray(abs_time, dtype=np.float64, order='C')


source_ip =  ndata[:,2]
source_ip[source_ip=='']='0'

dest_ip =  ndata[:,3]
dest_ip[dest_ip=='']='0'

size =  ndata[:,5] 
size[size=='']='0'
size = np.asarray(size, dtype=np.float64, order='C')

inter_arr_time =  ndata[:,15] 
inter_arr_time[inter_arr_time=='']='0'
inter_arr_time = np.asarray(inter_arr_time, dtype=np.float64, order='C')

labels =  ndata[:,16]
labels[labels=='']='0'
labels = np.asarray(labels, dtype=np.float64, order='C')

absTimeName = 'AbsoluteTime'
source_ipName = 'SourceIP'
dest_ipName = 'DestinationIP'
sizeName = 'Size'
IATName = 'InterArrivalTime'
labelName = 'Labels'
example_id = np.array(['%d' % i for i in range(len(labels))])
example_id_column_name = 'example_id'

X = np.array([inter_arr_time,size,source_ip,dest_ip])
#X = np.array([inter_arr_time])
y = np.array([labels])
y = y.reshape(y.shape[1:])
X = X.transpose()
print X.shape
print y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from matplotlib import pyplot as plt
#plt.scatter(x, alpha=0.2, )   #plot(X_train, y_train, 'bo')

#print X_train[0]

#plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
#plt.xlabel(X_train)
#plt.ylabel(y_train)
#plt.show()
#plt.scatter(X_test[:,0],X_test[:,1], c=y_test*200)   #plot(X_test,y_test, 'bo')
#plt.show()
if not os.path.isfile('./SVCPacketClassifier.joblib'):
    clfSVC = svm.SVC(gamma='scale', kernel='rbf')
    clfSVC.fit(X_train, y_train)
#scores = cross_val_score(clf, X, y, cv=4)
#print scores
else:
    print "Fetching SVM Modle from Disk"
    clfSVC = load('./SVCPacketClassifier.joblib')
y_pred = clfSVC.predict(X_test)

dump(clfSVC,'SVCPacketClassifier.joblib')

correct = 0
for i  in range(len(y_test)):
    correct += (y_pred[i] == y_test[i])
print "SVM: "
print (correct*1.0)/len(y_test) 

if not os.path.isfile('./LRPacketClassifier.joblib'):
 #   X_Anova = anova_kernel(X_train)
    rbf_feature = RBFSampler(gamma=1, random_state=1)
    X_train = X_train.astype(np.float64)   
    K_train = rbf_feature.fit_transform(X_train)
#    K_train = X_train.astype(np.float64)
#    clfLR = Pipeline([
#    ('nys', Nystroem(kernel='precomputed', n_components=100)),
#    ('lr', LogisticRegression())
#])
  #  clfLR.fit(K_train, y_train)
    clfLR = LogisticRegression(random_state=0, solver='lbfgs',max_iter=4000 ,multi_class='ovr').fit(K_train, y_train)
    dump(clfLR,'./LRPacketClassifier.joblib')
else:
    print "Fetching LR Model from Disk"
    clfLR = load('./LRPacketClassifier.joblib')
X_test = X_test.astype(np.float64)
#K_test = anova_kernel(X_test, X_train)
rbf_feature = RBFSampler(gamma=1, random_state=1)
y_pred = clfLR.predict(rbf_feature.fit_transform(X_test));
#y_pred = clfLR.predict(K_test)



#skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
 
dump(clfLR,'LRPacketClassifier.joblib')
correct = 0
for i  in range(len(y_test)):
    correct += (y_pred[i] == y_test[i])
print "Logistic Regression: " 
print (correct*1.0)/len(y_test)

if not os.path.isfile('./MLPPacketClassifier.joblib'):
    	clfMLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
	X_train = X_train.astype(np.float64)
	clfMLP.fit(X_train, y_train)
	dump(clfMLP,'./MLPPacketClassifier.joblib')
else:
    print "Fetching MLP Model from Disk"
    clfMLP = load('./MLPPacketClassifier.joblib')
X_test = X_test.astype(np.float64)
y_pred = clfMLP.predict(X_test)

correct = 0
for i  in range(len(y_test)):
    correct += (y_pred[i] == y_test[i])
print "Neural Nets: "
print (correct*1.0)/len(y_test)
