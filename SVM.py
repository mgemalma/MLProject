import numpy
import tensorflow as tf
import csv

with open('data_v1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = list(csv.reader(csv_file))


ndata = numpy.array(data)


abs_time = numpy.array( ndata[:,1])
abs_time[abs_time=='']='0'
abs_time = numpy.asarray(abs_time, dtype=numpy.float64, order='C')

abs_time_train = abs_time[0:int(len(abs_time)*(0.8)):1]
abs_time_test = abs_time[int(len(abs_time)*(0.8)):len(abs_time):1]

source_ip =  ndata[:,2]
source_ip[source_ip=='']='0'

source_ip_train =  source_ip[0:int(len(source_ip)*(0.8)):1]
source_ip_test = source_ip[int(len(source_ip)*(0.8)):len(source_ip):1]

dest_ip =  ndata[:,3]
dest_ip[dest_ip=='']='0'

dest_ip_train = dest_ip[0:int(len(dest_ip)*(0.8)):1]
dest_ip_test = dest_ip[int(len(dest_ip)*(0.8)):len(dest_ip):1]

size =  ndata[:,5] 
size[size=='']='0'
size = numpy.asarray(size, dtype=numpy.float64, order='C')

size_train = size[0:int(len(size)*(0.8)):1]
size_test = size[int(len(size)*(0.8)):len(size):1]

inter_arr_time =  ndata[:,15] 
inter_arr_time[inter_arr_time=='']='0'
inter_arr_time = numpy.asarray(inter_arr_time, dtype=numpy.float64, order='C')

inter_arr_time_train = inter_arr_time[0:int(len(inter_arr_time)*(0.8)):1]
inter_arr_time_test = inter_arr_time[int(len(inter_arr_time)*(0.8)):len(inter_arr_time):1]


labels =  ndata[:,16]
labels[labels=='']='0'
labels = numpy.asarray(labels, dtype=numpy.float64, order='C')

labels_train = labels[0:int(len(labels)*(0.8)):1]
labels_test = labels[int(len(labels)*(0.8)):len(labels):1]

absTimeName = 'AbsoluteTime'
source_ipName = 'SourceIP'
dest_ipName = 'DestinationIP'
sizeName = 'Size'
IATName = 'InterArrivalTime'
labelName = 'Labels'
example_id_train = numpy.array(['%d' % i for i in range(int(len(labels)*(0.8)))])
example_id_column_name = 'example_id'
example_id_test = numpy.array(['%d' % i for i in range(int(len(labels)*(0.2)))])

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={absTimeName: abs_time_train, source_ipName:source_ip_train, dest_ipName:dest_ip_train, sizeName:size_train, IATName:inter_arr_time_train, labelName:labels_train, example_id_column_name: example_id_train},
    y=labels_train,
    num_epochs=None,
    shuffle=True)

svm = tf.contrib.learn.SVM(
    example_id_column=example_id_column_name,
    feature_columns=(tf.contrib.layers.real_valued_column(
        column_name=IATName),),l2_regularization=0.1)

input_fn_eval = tf.estimator.inputs.numpy_input_fn(
    x={absTimeName: abs_time_test, source_ipName:source_ip_test, dest_ipName:dest_ip_test, sizeName:size_test, IATName:inter_arr_time_test, labelName:labels_test, example_id_column_name: example_id_test},
    y=labels_test,
    num_epochs=None,
    shuffle=True)

svm.fit(input_fn=train_input_fn, steps=10)
svm.evaluate(input_fn=input_fn_eval)
