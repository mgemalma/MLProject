import numpy
import tensorflow as tf
import csv
import pandas as pd
with open('data_v1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = list(csv.reader(csv_file))

#df_test = pd.read_csv('data_v1.csv')
#df_test.dtypes
#data = list(df_test)

ndata = numpy.array(data)
#print len(ndata)

abs_time = numpy.array( ndata[:,1])
abs_time[abs_time=='']='0'
abs_time = numpy.asarray(abs_time, dtype=numpy.float64, order='C')
#print abs_time

source_ip =  ndata[:,2]
source_ip[source_ip=='']='0'
#source_ip = numpy.asarray(source_ip, dtype=numpy.float64, order='C')
#print source_ip

dest_ip =  ndata[:,3]
dest_ip[dest_ip=='']='0'
#dest_ip = numpy.asarray(dest_ip, dtype=numpy.float64, order='C')

size =  ndata[:,5] 
size[size=='']='0'
size = numpy.asarray(size, dtype=numpy.float64, order='C')

inter_arr_time =  ndata[:,15] 
inter_arr_time[inter_arr_time=='']='0'
inter_arr_time = numpy.asarray(inter_arr_time, dtype=numpy.float64, order='C')

labels =  ndata[:,16]
labels[labels=='']='0'
labels = numpy.asarray(labels, dtype=numpy.float64, order='C')

#with open('your_file1.txt', 'w') as f:
#    for item in abs_time:
#        f.write("%s\n" % item)
#data_tf = (abs_time, np.float32)
#estimator = tf.contrib.learn.SVM( example_id_column='example_id'	,
#    feature_columns=[real_feature_column, sparse_feature_column],
#    l2_regularization=10.0)
absTimeName = 'AbsoluteTime'
source_ipName = 'SourceIP'
dest_ipName = 'DestinationIP'
sizeName = 'Size'
IATName = 'InterArrivalTime'
labelName = 'Labels'
example_id = numpy.array(['%d' % i for i in range(len(labels))])
example_id_column_name = 'example_id'
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={absTimeName: abs_time, source_ipName:source_ip, dest_ipName:dest_ip, sizeName:size, IATName:inter_arr_time, labelName:labels, example_id_column_name: example_id},
    y=labels,
    num_epochs=None,
    shuffle=True)
svm = tf.contrib.learn.SVM(
    example_id_column=example_id_column_name,
    feature_columns=(tf.contrib.layers.real_valued_column(
        column_name=IATName),),l2_regularization=0.1)

svm.fit(input_fn=train_input_fn, steps=10)
