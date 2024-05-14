from .train_global_args import *
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.layers import TimeDistributed, LSTM
from keras.optimizers import Adam
from keras import backend as K
from keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import MaxPooling1D
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras import regularizers
STRIDES = 1
# number of cross validation of model training
RELU = 'relu'
LINEAR = 'linear'
SIGMOID = 'sigmoid'
MSE = 'mse'

@register_keras_serializable()
def __tf_pearson_correlation(y_true, y_pred): 
    # use smoothing for not resulting in NaN values
    # pearson correlation coefficient
    # https://github.com/WenYanger/Keras_Metrics
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)



def get_metric(binary):
    print(f'binary - {binary}')
    return AUC(from_logits=True,name='auc')  if binary else __tf_pearson_correlation

def build_model(n_motif, length_motif, input_shape, dropout_rate,
                learning_rate,hidden_layer, binary=False, **kwargs):
    # print all arguments
    # print(f"n_motif: {n_motif}, length_motif: {length_motif}, input_shape: {input_shape}, dropout_rate: {dropout_rate}, learning_rate: {learning_rate}, hidden_layer: {hidden_layer}")

    model = Sequential()
    model.add(Conv1D(filters=n_motif,
                     kernel_size=(length_motif,),
                     strides=STRIDES,
                     activation='relu',
                     input_shape=input_shape,
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
                learning_rate, hidden_layer, l1=0.0, l2=0.0,
                binary=False, **kwargs):
    # Define the regularizer
    kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2)
    
    model = Sequential()
    model.add(Conv1D(filters=n_motif,
                     kernel_size=(length_motif,),
                     strides=STRIDES,
                     activation=RELU,
                     input_shape=(None, 4),
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


def build_fcn_model(n_motif, length_motif, dropout_rate, learning_rate, hidden_layer,
                        dense_output, lstm_output, dense2_output ,binary=False, **kwargs):
    # Create a Sequential model
    model = Sequential()

    # Add a Conv1D layer
    model.add(Conv1D(filters=n_motif,
                     kernel_size=(length_motif,),
                     strides=STRIDES,
                     activation=RELU,
                     input_shape=(None, 4),  # None indicates that any input length is acceptable
                     ))

    # Add a MaxPooling1D layer
    model.add(MaxPooling1D(pool_size=(length_motif -  1,)))

    # Add a Dropout layer
    model.add(Dropout(rate=dropout_rate))

    # Add a TimeDistributed layer
    model.add(TimeDistributed(Dense(dense_output, activation=RELU)))

    # Add an LSTM layer
    model.add(LSTM(lstm_output))

    if hidden_layer:
        model.add(Dense(units=32,
                        activation=RELU,
                        ))
    # Add a Dense layer
    model.add(Dense(units=1,
                    activation= SIGMOID if binary else LINEAR,
                    )) 

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss= BinaryCrossentropy() if binary else MSE,
                  metrics=get_metric(binary))

    return model




VERSION_DICT = {'original': build_model, # original deepbind model
                'fcn_model': build_fcn_model, # by chatGPT
                'original_v2': build_model2 }

def get_model(version,parameter_dict):
    model = VERSION_DICT[version](**parameter_dict)
    print(model.summary())
    print(model.metrics)
    return model