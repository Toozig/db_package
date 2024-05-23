from Bio import SeqIO
import re
from tqdm import tqdm
import numpy as np
from keras import backend as K
from keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from tensorflow import convert_to_tensor


import os
import concurrent
from .run_general import get_reversed_record, GENERATED, DEBUG, SHIFT_PARAM, fasta_from_seq_record, save_prediction_df
import pandas as pd
import ast
from tqdm import tqdm
from ..predictionArchiver.predictionArchiver import PredictionSaver
from ..db_train_utils.db_train_utils import oneHot_encode
from ..db_train_utils.train_global_args import *
import configparser

config = configparser.ConfigParser()
cur_dir = os.path.dirname(__file__)
# set the work dir to this dir
os.chdir(cur_dir)
config.read('../config.ini')


BASE_PATH = config['MODEL_RUN']['BASE_PATH']

IB_MODEL_PATH = f'{BASE_PATH}/%s/models'
IB_MODEL_PATH2 = f'{BASE_PATH}/%s/models/%s.keras'
N_PROCESS = 5

# IB_MODEL_PATH ='/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models/IB_models/%s/models'
# IB_MODEL_PATH2 ='/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models/IB_models/%s/models/%s.keras'
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
    """
    prediction of multi samples of single moodel
    """
    model = load_model(model_path)
    score = model.predict(onehot_seqs, verbose=0)
    model_id = model_path.split('/')[-1].split('.keras')[0]
    return {model_id : score}



def get_IB_model_prediction2(model_id,onehot_seqs):
    """
    Calculates the score for each sub-model (1/10) of the model_id
    """
    model_path = IB_MODEL_PATH2 % (model_id , model_id)
    model = load_model(model_path)
    score = model.predict(onehot_seqs, verbose=0)
    return score


def get_IB_models(model_ids: list) -> list:
    """
    Returns a list of models from the model_ids
    """
    models = []
    for model_id in model_ids:
        model_path = IB_MODEL_PATH2 % (model_id, model_id)
        models += [load_model(model_path)]
    return models

def get_IB_model_prediction(model_id,onehot_seqs):
    """
    Calculates the score for each sub-model (1/10) of the model_id
    """
    model_path = IB_MODEL_PATH % model_id
    models = os.listdir(model_path)
    models = [os.path.join(model_path,model) for model in models]
    result_dict = {}
    
    with concurrent.futures.ProcessPoolExecutor(N_PROCESS) as executor:
        results = [executor.submit(predict_IB_model, model, onehot_seqs) for model in models]
        for f in concurrent.futures.as_completed(results):
            score = f.result()
            result_dict.update(score)
    return {"model_id": model_id,'score':result_dict}

def get_processed_result(model_id,onehot_seqs):
    """
    given ID and onehot encoded sequences, returns the prediction of the models
     (using mean to aggregate the results of the 10 models)
    """
    result = get_IB_model_prediction(model_id,onehot_seqs)
    result = {k: v.flatten() for k, v in result['score'].items()}
    return pd.DataFrame(result).mean(axis=1).to_numpy()


    
def run_model_on_subsequence(model_id, fasta_file_path):
    seq_records  = [str(recored.seq) for recored in SeqIO.parse(fasta_file_path, "fasta")]
    one_hot_seqs = np.stack([oneHot_encode(record) for record in seq_records])
    one_hot_seqs = convert_to_tensor(one_hot_seqs)
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

    exp_dict = ast.literal_eval(exp_details)
    input_shape = exp_dict['input_shape']
    return input_shape


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
    with concurrent.futures.ProcessPoolExecutor(N_PROCESS) as executor:
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


