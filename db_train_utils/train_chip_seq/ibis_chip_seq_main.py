from ..train_global_args import *
from ..train_global_function import *
from .chip_seq_utils import   prepare_chs_train_test, process_init_result, prepare_chs_ibis_df, get_train_test_data
import concurrent.futures
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras import regularizers
from multiprocessing import Pool

from tqdm import tqdm
DEBUG = False


POSITIVE_DATA = TRAIN_SET
NEGATIVE_DATA = TEST_SET

TMP_DIR = '/tmp/toozig/'

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






def build_model2(n_motif, length_motif, dropout_rate,
                learning_rate, hidden_layer, l1=0.0, l2=0.0,
                binary=False, **kwargs):
    # Define the regularizer
    kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2)
    
    model = Sequential()
    model.add(Conv1D(filters=n_motif,
                     kernel_size=(length_motif,),
                     strides=STRIDES,
                     activation='relu',
                     input_shape=(None, 4),
                     kernel_regularizer=kernel_regularizer  # Add kernel regularizer
                     )
              )
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(rate=dropout_rate))
    if hidden_layer:
        model.add(Dense(units=32,
                        activation='relu',
                        ))

    model.add(Dense(units=1,
                    activation='sigmoid' if binary else 'linear',
                    )) 
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy() if binary else MSE,
                   metrics=[__tf_pearson_correlation])
    return model


def build_fcn_model(n_motif, length_motif, dropout_rate, learning_rate, hidden_layer,
                        dense_output, lstm_output, dense2_output ,binary=False, **kwargs):
    # Create a Sequential model
    model = Sequential()

    # Add a Conv1D layer
    model.add(Conv1D(filters=n_motif,
                     kernel_size=(length_motif,),
                     strides=STRIDES,
                     activation='relu',
                     input_shape=(None, 4),  # None indicates that any input length is acceptable
                     ))

    # Add a MaxPooling1D layer
    model.add(MaxPooling1D(pool_size=(length_motif -  1,)))

    # Add a Dropout layer
    model.add(Dropout(rate=dropout_rate))

    # Add a TimeDistributed layer
    model.add(TimeDistributed(Dense(dense_output, activation='relu')))

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
                   metrics=[__tf_pearson_correlation])

    return model




VERSION_DICT = {'v1': build_fcn_model,
                'v2': build_model2 }

def train_ibis(parameter_dict, x_train, y_train, x_test, y_test, version='v1', final=False):
    model = VERSION_DICT[version](**parameter_dict)
    if final:
        model, train_eval_dict = train_final_model(model, x_train, y_train, x_test, y_test, parameter_dict[LEARNING_STEP])
    else:
        model, train_eval_dict = train_model(model, x_train, y_train,
                                        x_test, y_test, parameter_dict[LEARNING_STEP])
    val_eval_dict = eval_model(model, x_train, y_train)
    test_eval_dict = eval_model(model, x_test ,y_test)
    val_dict = {TRAIN_STR: train_eval_dict, VAL_STR: val_eval_dict, TEST_STR: test_eval_dict}
    return model, val_dict






def run_ibis_single_expirement(row, to_split, x_train, y_train, version='v1', final=False):
    print('running expirement', row[EXP_ID], 'model:', version)
    # print(row)

    x_train, y_train, x_test, y_test  = prepare_chs_train_test(x_train, y_train, to_split)
    print('prepared data')
    parameter_dict = row if isinstance(row, dict) else row.to_dict()
    print('got parms')
    model, train_eval_dict = train_ibis(parameter_dict, x_train, y_train, x_test, y_test, version, final)
    print('finished running expirement')
    if DEBUG:
        print(f"finished expirement {row.expirement_id}")
    if to_split:
        model = None
    return {EXP_ID: row[EXP_ID],'model': model, 'score_dict' :train_eval_dict}





def run_ibis_chip_seq(positive_fasta, negative_fasta, n_exp, commetAPIKey, db_chs_obj, version='v1'):
    configuration_df = prepare_chs_ibis_df(n_exp,db_chs_obj,commetAPIKey, positive_fasta, negative_fasta)
    global_model_data = db_chs_obj.get_db_data()
    exp_id = global_model_data['model_id']
    print(f"Starting expirement {exp_id}")
    print(f"Starting expirement {exp_id}")
    init_results = []
    results = []
    rows = [row for _, row in configuration_df.iterrows()]
    negative_data = negative_fasta
    x_train, y_train = get_train_test_data(positive_fasta, negative_data, False)
    print(f'converted data to oneHot encode and test, there are {len(x_train)} samples')
    with concurrent.futures.ProcessPoolExecutor(60) as executor:
        # Submit each configuration to the executor for parallel execution
        init_results = executor.map(run_ibis_single_expirement, rows, [True] * len(rows),[x_train] * len(rows), [y_train] * len(rows), [version] * len(rows))
        # get the top 01 config with best results
        top_ten = process_init_result(configuration_df,init_results,exp_id)
        rows = top_ten.to_dict(orient='records')    

    with concurrent.futures.ProcessPoolExecutor(20) as executor:
        results = list(executor.map(run_ibis_single_expirement, rows, [False] * len(rows),[x_train] * len(rows),
                                 [y_train] * len(rows), [version] * len(rows), [True] * len(rows) ))
    print('finished running final test')
    df = get_n_save_final_df(results,top_ten,exp_id, False)
    print('processed results')
    process_result(results, df, exp_id)
    global_model_data['model_id'] = exp_id
    # copy the data to the output folder
    copy_model_data(df[POSITIVE_DATA].values[0], exp_id)
    # add model to the model tabel
    db_chs_obj.save_model_to_table()
    print(f"Finished expirement {exp_id}")
