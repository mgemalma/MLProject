import numpy as np
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from sklearn import svm
#from sklearn.model_selection import cross_val_score

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
y = np.array([labels])
y = y.reshape(y.shape[1:])
X = X.transpose()
print X.shape
print y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = svm.SVC(gamma='scale')
clf.fit(X, y)
#scores = cross_val_score(clf, X, y, cv=4)
#print scores
y_pred = clf.predict(X_test)

correct = 0
for i  in range(len(y_test)):
    correct += (y_pred[i] == y_test[i])
print (correct*1.0)/len(y_test)

 
