from .train_global_args import *
from keras.models import Sequential
import os
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.layers import Input
from keras.optimizers import Adam
import tensorflow as tf
from keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import MaxPooling1D
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
import keras
from keras import layers
STRIDES = 1
# number of cross validation of model training
RELU = 'relu'
LINEAR = 'linear'
SIGMOID = 'sigmoid'
MSE = 'mse'





@register_keras_serializable()
def pearson_correlation(y_true, y_pred): 
    y_true = tf.reshape(y_true, (-1, 1))  # reshape y to match the shape of y_pred
    return tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=-1)



def get_metric(binary):
    # print(f'binary - {binary}')
    return AUC(from_logits=True,name='auc')  if binary else pearson_correlation

def build_model(n_motif, length_motif, input_shape, dropout_rate,
                learning_rate,hidden_layer,expirement_id, binary, **kwargs):
    # print all arguments
    # print(f"n_motif: {n_motif}, length_motif: {length_motif}, input_shape: {input_shape}, dropout_rate: {dropout_rate}, learning_rate: {learning_rate}, hidden_layer: {hidden_layer}")

    model = Sequential(name=expirement_id)
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=n_motif,
                     kernel_size=(length_motif,),
                     strides=STRIDES,
                     activation='relu',
                    #  kernel_initializer=LogUniformInitializer(motif_initializer_seed),
                    #  kernel_regularizer=regularizers.l1(motif_regularizer)
                    )
                     )
    model.add(MaxPooling1D(pool_size=(length_motif -  1,)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Flatten()) 
    if hidden_layer:
        model.add(Dense(units=32,
                        activation=RELU,
                        ))

    model.add(Dense(units=1,
                    activation= SIGMOID if binary else LINEAR,
                    )) 
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss= BinaryCrossentropy() if binary else MSE,
                   metrics=[get_metric(binary)])
    return model



def build_model2(n_motif, length_motif, dropout_rate,
                learning_rate, hidden_layer,expirement_id, l1=0.0, l2=0.0,
                binary=False, **kwargs):
    # Define the regularizer
    kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2)
    
    model = Sequential(name=expirement_id)
    model.add(Input(shape=(None, 4)))
    model.add(Conv1D(filters=n_motif,
                     kernel_size=(length_motif,),
                     strides=STRIDES,
                     activation=RELU,
   
                     kernel_regularizer=kernel_regularizer  # Add kernel regularizer
                     )
              )
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(rate=dropout_rate))
    if hidden_layer:
        model.add(Dense(units=32,
                        activation=RELU,
                        ))

    model.add(Dense(units=1,
                    activation='sigmoid' if binary else 'linear',
                    )) 
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy() if binary else MSE,
                   metrics=[get_metric(binary)])
    return model

import random
import string


def generate_random_ID():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

def model_wrapper(model):
    name = model.name if model.name != 'model' else generate_random_ID()
    input_shape = model.input_shape[1:] 
    inputs = keras.Input(input_shape)
    output = model(inputs)
    new_model = keras.Model(inputs=inputs, outputs=output, name = name)
    return new_model

def get_ensemble_model(model_list, model_name= ''):
    name = model_name if len(model_name) else generate_random_ID()
    input_shape = model_list[0].input_shape
    inputs = keras.Input(shape=input_shape[1:])
    outputs = [model_wrapper(model)(inputs) for model in model_list]  # get the output tensor of the model
    ensemble_model = keras.Model(inputs=inputs, outputs=outputs, name = name)
    return ensemble_model



def save_ensamble_model(model_list, model_id, output_dir, binary):
    save_path= os.path.join(output_dir, 'models')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ensemble_model = get_ensemble_model(model_list)
    outputs = layers.average(ensemble_model.outputs)
    ensemble_model = keras.Model(inputs=ensemble_model.inputs, outputs=outputs)
    ensemble_model.compile(loss=BinaryCrossentropy() if binary else MSE,
                metrics=[get_metric(binary)])
    save_name = f'{save_path}/{model_id}.keras'
    ensemble_model.save(save_name)
    return save_name 



VERSION_DICT = {'original_DB': build_model, # original deepbind model
                'original_v2': build_model2 }

def get_model(version,parameter_dict):
    model = VERSION_DICT[version](**parameter_dict)
    print(model.summary())
    print(model.metrics)
    return model