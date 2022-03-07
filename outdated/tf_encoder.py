# %%
import matplotlib.pyplot as plt
from scipy.sparse.construct import random

import sklearn
import scipy
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.models import Model, load_model
from keras.layers import Input, Dense, Layer, InputSpec
from keras import callbacks
from keras import regularizers, activations, initializers, constraints, Sequential
from keras import backend as K
from keras.constraints import Constraint
from keras.regularizers import Regularizer

class Tied(Layer):
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               #kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               #kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               #kernel_constraint=None,
               bias_constraint=None,
               tied_to = None,
               **kwargs):
    super(Tied, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.tied_to = tied_to

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)

    #self.kernel_initializer = initializers.get(kernel_initializer)
    #self.kernel_regularizer = regularizers.get(kernel_regularizer)
    #self.kernel_constraint = constraints.get(kernel_constraint)
    self.kernel_regularizer = None
    self.kernel_initializer = None
    self.kernel_constraint = None

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True

  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Tied` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

    weights = self.tied_to.weights[0]
    self.transposed_weights = tf.transpose(weights, name='{}_kernel_transpose'.format(self.tied_to.name))

    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):

    if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
      inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

    rank = inputs.shape.rank
    if rank == 2 or rank is None:
      if isinstance(inputs, tf.SparseTensor):
        inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
        ids = tf.SparseTensor(
            indices=inputs.indices,
            values=inputs.indices[:, 1],
            dense_shape=inputs.dense_shape)
        weights = inputs
        outputs = tf.nn.embedding_lookup_sparse(
            self.transposed_weights, ids, weights, combiner='sum')
      else:
        outputs = tf.raw_ops.MatMul(a=inputs, b=self.transposed_weights)

    else:
      outputs = tf.tensordot(inputs, self.transposed_weights, [[rank - 1], [0]])
      if not tf.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.kernel.shape[-1]]
        outputs.set_shape(output_shape)
      
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % (input_shape,))
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(Dense, self).get_config()
    config.update({
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        #'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        #'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        #'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    })
    base_config = super(Tied, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def create_model(input_dim, encoding_dim):
  Encoder = Dense(encoding_dim, activation="relu", use_bias=False, kernel_regularizer = tf.keras.regularizers.l1(0.2), input_shape=(input_dim,), name="Encoder" )
  #Decoder = Dense(input_dim, activation="relu", use_bias=False, name="Decoder" )
  #Decoder.set_weights([Encoder.get_weights()[0].T])
  Decoder = Tied(input_dim, activation="relu", use_bias = False, tied_to=Encoder, name="Decoder" )
  #Decoder.trainable=False
  Autoencoder = Sequential([Encoder,Decoder], name="Autoencoder")

  Autoencoder.compile(metrics=['mse'], loss='mean_squared_error', optimizer='adam')
  return Autoencoder

dataset = "bearingset"
filename = "health_20_0.csv"

df = pd.read_csv(f"raw_{dataset}/raw_{filename}")
#train_data, test_data = train_test_split(df, test_size = 0.3, random_state = 124)
#train_data, validation_data= train_test_split(train_data, test_size = 0.14, random_state = 31)

#scaler = MinMaxScaler().fit(train_data)
#train_data = scaler.transform(train_data)
#validation_data = scaler.transform(validation_data)
#test_data = scaler.transform(test_data)

nb_epoch = 25
batch_size = 1024
input_dim = df.shape[1] #num of predictor variables, 
encoding_dim = 2
learning_rate = 1e-2 

Autoencoder = create_model(input_dim, encoding_dim)

Autoencoder.fit(df, df, epochs=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1)

# earlystop = callbacks.EarlyStopping(monitor ="val_loss", 
#                                     mode ="min", patience = 5, 
#                                     restore_best_weights = True)
# callbacks = [earlystop]

# Autoencoder.evaluate(test_data,test_data, verbose=2)
# Autoencoder.save_weights(f"models/{dataset}/{filename}/weights")



# testModel = create_model(input_dim, encoding_dim)
# testModel.load_weights(f"models/{dataset}/{filename}/weights")
# testModel.evaluate(test_data,test_data, verbose=2)
