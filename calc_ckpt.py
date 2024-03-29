from __future__ import print_function
import tensorflow as tf
ckpt = tf.train.get_checkpoint_state("./ckpt_path/").model_checkpoint_path
saver = tf.train.import_meta_graph(ckpt+'.meta')
variables = tf.trainable_variables()
total_parameters = 0
for variable in variables:
  shape = variable.get_shape()
  variable_parameters = 1
  for dim in shape:
    variable_parameters *= dim.value
  total_parameters += variable_parameters
print(total_parameters)
