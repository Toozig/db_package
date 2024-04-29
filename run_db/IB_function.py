from Bio import SeqIO
import re
from tqdm import tqdm
import numpy as np
from keras import backend as K
from keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv1D
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


import os
import concurrent
from .run_general import get_reversed_record, GENERATED, DEBUG, SHIFT_PARAM, fasta_from_seq_record, save_prediction_df, N_PROCESS
import pandas as pd
import ast
from tqdm import tqdm
from ..predictionArchiver.predictionArchiver import PredictionSaver
from ..db_train_utils.db_train_utils import oneHot_encode
from ..db_train_utils.train_global_function import build_model
from ..db_train_utils.train_global_args import *


IB_MODEL_PATH ='/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models/IB_models/%s/models'
MODEL_PARMS = [EXP_ID,N_MOTIF, LENGTH_MOTIF, DROPOUT_RATE, LEARNING_RATE, HIDDEN_LAYER, BINARY]

DEBUG = False



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


def predict_IB_model(model_path, onehot_seqs):
    model = load_model(model_path)
    score = model.predict(onehot_seqs, verbose=0)
    model_id = model_path.split('/')[-1].split('.keras')[0]
    return {model_id : score}

def get_IB_model_prediction(model_id,onehot_seqs):
    model_path = IB_MODEL_PATH % model_id
    models = os.listdir(model_path)
    models = [os.path.join(model_path,model) for model in models]
    result_dict = {}
    
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        results = [executor.submit(predict_IB_model, model, onehot_seqs) for model in models]
        for f in concurrent.futures.as_completed(results):
            score = f.result()
            result_dict.update(score)
    return {"model_id": model_id,'score':result_dict}


def get_converted_model(original_model, n_nucleotides):
    new_model = get_simple_model(original_model.optimizer, original_model.loss, n_nucleotides = n_nucleotides)

    # copy all weights
    for i in range(len(original_model.layers)):
        new_model.layers[i].set_weights(original_model.layers[i].get_weights())

    return new_model
    
def run_model_on_subsequence(model_id, fasta_file_path):
    seq_records  = [str(recored.seq) for recored in SeqIO.parse(fasta_file_path, "fasta")]
    one_hot_seqs = np.stack([oneHot_encode(record) for record in seq_records])
    results = get_IB_model_prediction(model_id,one_hot_seqs)
    results.update({'subsequnces': seq_records})
    return results
        
        
def P1_to_P2_aggregation_func(df):
    return df.mean(axis=1)

def process_IB_results(results, window, shift):

    seq_id = results['seq_id']
    df_dict = {k: v.flatten() for k, v in results['score'].items()}
    result_df = pd.DataFrame(df_dict, index = results['subsequnces'],)
    model_id = results['model_id']

    PredictionSaver().save_P1_model(result_df, model_id, seq_id, window, shift)
    return P1_to_P2_aggregation_func(result_df)


def process_P1_to_P2_result(result_dicts):
    result = pd.concat([i.reset_index(drop=False) for i in result_dicts], axis=1)
    return result.T.drop_duplicates().T.set_index('index')


def run_IB_on_sequence(record,model_df, shift):
    models_ids = model_df['id'].values
    input_shape = model_df.input_shape.values[0]
    if len(record.seq) < input_shape:
        return [{'seq_id': record.id, 'error': 'sequence is too short'}]
    saver = PredictionSaver()
    save_path = saver.get_P2_saving_path(GENERATED, record.id, input_shape, shift)
    if os.path.exists(save_path):
        # saver.save_P1_model(self,df,model_id,record.id, input_shape, shift)
        return saver.json()
    fasta_file_path = fasta_from_seq_record(record,input_shape,shift) # todo - shift paramaeter
    result_dicts = []

    # for model_id in models_ids:
    #     cur_prediciton = run_model_on_subsequence(model_id, fasta_file_path)
    #     cur_prediciton.update({'seq_id': record.id}) 
    #     model_seq_series =  process_IB_results(cur_prediciton,input_shape, shift)
    #     result_dicts.append(model_seq_series.rename(cur_prediciton['model_id']))
    #     return result_dicts
    with concurrent.futures.ProcessPoolExecutor(N_PROCESS) as executor:
        results = [executor.submit(run_model_on_subsequence, model_id, fasta_file_path) for model_id in models_ids]
        for f in concurrent.futures.as_completed(results):
            cur_prediciton = f.result()
            cur_prediciton.update({'seq_id': record.id}) 
            model_seq_series =  process_IB_results(cur_prediciton,input_shape, shift)
            result_dicts.append(model_seq_series.rename(cur_prediciton['model_id']))
    
    result_df = process_P1_to_P2_result(result_dicts)
    saver.save_prediction_df(result_df, save_path)
    return saver.json()
            


def count_sequences(fasta_file):
    with open(fasta_file, 'r') as file:
        count = 0
        for line in file:
            if line.startswith('>'):
                count += 1
        return count

def get_input_shape(exp_details):
    return ast.literal_eval(exp_details)['input_shape'][0]


def main_IB(fasta_file,model_df, shift):

    seq_records =list(SeqIO.parse(fasta_file, "fasta"))
    print(len(seq_records))

    model_df.loc[:, 'input_shape'] = model_df['experiment_details'].apply(get_input_shape)

    grouped = model_df.groupby('input_shape')
    saver = PredictionSaver(os.path.basename(fasta_file))
    saver_df = saver.get_model_df()
    if not len(saver_df):
        saver_df = model_df
    saver_df.loc[model_df.index, 'input_shape'] = model_df['input_shape']
    all_results = []
    print('len grouped:', len(grouped))
    for group in grouped.groups:
        cur_model_df = grouped.get_group(group)
        print(f'running on model with input shape {group}')
        with concurrent.futures.ProcessPoolExecutor(N_PROCESS) as executor:
            model_df_list = [cur_model_df]*len(seq_records)
            shift_list = [shift]*len(seq_records)
    
            results = list(tqdm(executor.map(run_IB_on_sequence, seq_records, model_df_list, shift_list), total=len(seq_records)))
            #results = [executor.submit(run_IB_on_sequence, seq_req, cur_model_df,shift) for seq_req in seq_records]
            for cur_result in results:
                saver = saver.merge_json(cur_result)

    for k,v in saver.get_dict('P2').items():
        all_results += [vi.path for vi in v]

    # print(all_results)
    saver.save_data()
    return {'model_type':GENERATED,'path_list': all_results}



def run_IB_on_sequence_no_shift(record,model_df, fasta_file_path, project_name =''):
    input_shape = len(record.seq)
    models_ids = model_df['id'].values
    basename_fatsa = os.path.basename(fasta_file_path).split('.')[0]
    saver = PredictionSaver(basename_fatsa if project_name == '' else project_name)
    save_path = saver.get_P2_saving_path(GENERATED, record.id, input_shape, -1)
    if os.path.exists(save_path):
        # saver.save_P1_model(self,df,model_id,record.id, input_shape, shift)
        return saver.json()
    result_dicts = []
    with concurrent.futures.ProcessPoolExecutor(N_PROCESS) as executor:

        results = [executor.submit(run_model_on_subsequence, model_id, fasta_file_path) for model_id in models_ids]
        for f in concurrent.futures.as_completed(results):
            cur_prediciton = f.result()
            cur_prediciton.update({'seq_id': cur_prediciton['model_id']}) 
            model_seq_series =  process_IB_results(cur_prediciton,input_shape, -1)
            result_dicts.append(model_seq_series.rename(cur_prediciton['model_id']))
    
    result_df = process_P1_to_P2_result(result_dicts)
    saver.save_prediction_df(result_df, save_path)
    return saver.json()



def run_model_on_subsequence_ibis(model_id, one_hot_seqs, seq_ids):
    results = get_IB_model_prediction(model_id,one_hot_seqs)
    results.update({'subsequnces': seq_ids})
    return results

from tqdm import tqdm

def count_lines_starting_with_gt(filepath):
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def run_ibis_batch(seq_records, models_ids):
    one_hot_seqs = np.stack([oneHot_encode(str(record.seq)) for record in seq_records])
    seq_ids = [record.id for record in seq_records]
    print(one_hot_seqs.shape)
    print(f'running on {len(seq_records)} sequences')
    result_dicts = []
    with concurrent.futures.ProcessPoolExecutor(20) as executor:
        results = [executor.submit(run_model_on_subsequence_ibis, model_id, one_hot_seqs, seq_ids) for model_id in models_ids]
        for f in concurrent.futures.as_completed(results):
            cur_prediciton = f.result()
            cur_prediciton.update({'seq_id': cur_prediciton['model_id']}) 
            model_seq_series =  process_IB_results(cur_prediciton,-1, -1)
            result_dicts.append(model_seq_series.rename(cur_prediciton['model_id']))
    result_df = process_P1_to_P2_result(result_dicts)
    return result_df
        


def run_ibis_on_sequence_no_shift(model_df, fasta_file_path, project_name, filter_ids=[]):
    models_ids = model_df['id'].values
    basename_fatsa = os.path.basename(fasta_file_path).split('.')[0]
    saver = PredictionSaver( project_name)
    save_path = saver.get_P2_saving_path(GENERATED,project_name, -1, -1)
    if os.path.exists(save_path):
        return saver.json()
    result_dfs = []
    counter = 0
    seq_records = []
    fasta_length = count_lines_starting_with_gt(fasta_file_path)
    print(f'running on {fasta_length} sequences')
    for record in tqdm(SeqIO.parse(fasta_file_path, "fasta"), total=fasta_length):
        if record.id in filter_ids:
            continue
        counter += 1
        seq_records.append(record)
        if counter % 3500 == 0 or counter == fasta_length:
            result_df = run_ibis_batch(seq_records, models_ids)
            result_dfs.append(result_df)
            seq_records = []
            one_hot_seqs = []
    if len(seq_records):
        result_df = run_ibis_batch(seq_records, models_ids)
        result_dfs.append(result_df)
    result_df = pd.concat(result_dfs)
    saver.save_prediction_df(result_df, save_path)
    return saver.json()



































































    

# def change_input_shape(model, new_input_shape):
#     # Create a new model with the desired input shape
    
#     new_model = Sequential()
#     new_model.add(Conv1D(filters=model.layers[0].filters,
#                          kernel_size=model.layers[0].kernel_size,
#                          strides=model.layers[0].strides,
#                          padding=model.layers[0].padding,
#                          activation=model.layers[0].activation,
#                          input_shape=new_input_shape))

#     # Copy the weights from the original model to the new model
#     new_model.layers[0].set_weights(model.layers[0].get_weights())

#     # Add the remaining layers from the original model to the new model
#     for layer in model.layers[1:]:
#         new_model.add(layer)

#     return new_model

from tensorflow.keras.layers import GlobalAveragePooling1D
STRIDES = 1

RELU = 'relu'
LINEAR = 'linear'
SIGMOID = 'sigmoid'
MSE = 'mse'

def build_fcn_model(n_motif, length_motif, dropout_rate, learning_rate, binary=False, **kwargs):
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

    # Add a Flatten layer
    model.add(GlobalAveragePooling1D())  # Replaces Flatten for FCN

    # Add a Dense layer
    model.add(Dense(units=1,
                    activation= SIGMOID if binary else LINEAR,
                    )) 

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss= BinaryCrossentropy() if binary else MSE,
                   metrics=[__tf_pearson_correlation])

    return model



def change_input_shape(model, new_input_shape):
    # Create a new model with the new input shape
    new_model = build_fcn_model(n_motif=model.layers[0].filters,
                                length_motif=model.layers[0].kernel_size[0],
                                input_shape=new_input_shape,
                                dropout_rate=model.layers[2].rate,
                                learning_rate=model.optimizer.learning_rate,
                                hidden_layer=len(model.layers) > 4,
                                binary=model.layers[-1].activation == 'sigmoid')

    # Copy the weights from the old model to the new model
    for new_layer, layer in zip(new_model.layers, model.layers):
        if not isinstance(new_layer, GlobalAveragePooling1D):
            new_layer.set_weights(layer.get_weights())

    return new_model


def predict_wrapper(model_path, X, input_shape):
    model = load_model(model_path)
    new_model = change_input_shape(model, input_shape)
    model_id = model_path.split('/')[-1].split('.keras')[0]
    return {model_id: model.predict(X)}


def get_models_path(model_id):
    """
    create a model using the weight of the previous model and the new input shape
    """
    # load the original model
    model_path =  IB_MODEL_PATH % model_id
    models_path = [model_path + '/' + i for i in os.listdir(model_path)]
    return models_path


def predict_on_different_input_shape(model_id, X):
    """
    predict on the same model with different input shape
    """
    input_shape = X.shape
    print(input_shape)
    models_path = get_models_path(model_id)
    for model_path in models_path:
        model = load_model(model_path)
        new_model = change_input_shape(model, input_shape)
        model_id = model_path.split('/')[-1].split('.keras')[0]
        return {model_id: model.predict(X)}
    with concurrent.futures.ProcessPoolExecutor(len(models_path)) as executor:
        predictions = executor.map(predict_wrapper, models_path, 
                                        [X]*len(models_path),
                                        [input_shape]*len(models_path))
    return pd.DataFrame(predictions)
    

    
# def predict_wrapper2(model_record, X, model_id):
#     model = get_model(model_record)

#     model_id = model_path.split('/')[-1].split('.keras')[0]
#     return {model_id: model.predict(X)}

# def get_model_parms_df(input_shape,model_id):
#     """
#     create a model parameter dataframe from the model table, adding the wanted input shape
#     """
#     model_parms_path = IB_MODEL_PATH % model_id
#     parms_df = pd.read_csv(model_parms_path)
#     parms_df[BINARY] = FALSE if BINARY not in parms_df.columns else parms_df[BINARY]
#     params_df = parms_df[MODEL_PARMS]
#     params_df[INPUT_SHAPE] = input_shape
#     return params_df

# def get_model(model_record):
#     """
#     create a model using the weight of the previous model and the new input shape
#     """
#     # load the original model
#     exp_id = model_record.pop(EXP_ID)
#     model_path = os.path.join(MODEL_FOLDER,model_id, 'models', f'{exp_id}.keras')
#     old_model = load_model(model_path)

#     # create model with same parameters
#     new_model = build_model(**model_record)
#     # copy the weights
#     for i in range(len(old_model.layers)):
#         new_model.layers[i].set_weights(old_model.layers[i].get_weights())
#     return new_model