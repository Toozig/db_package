import numpy as np
import pandas as pd
import concurrent.futures
import os
## set the sworkdir tto the file folder
os.chdir(os.path.dirname(__file__))

from .run_general import get_subsequences, fasta_from_seq_string, get_model_df
from .IB_function import get_IB_model_prediction, get_input_shape, process_IB_results
from ..db_train_utils import oneHot_encode
from .original_function import get_original_model_prediction, save_model_list
from .process_P2 import aggregate_postion_score 

TMP_DIR = '/tmp/toozig/'
N_PROCESS=60


def get_oneHot_sequence(seq, window, shift):
    subseqs = get_subsequences(seq, window, shift)
    one_hot_seqs = np.stack([oneHot_encode(record) for record in subseqs])
    return  one_hot_seqs
    

def process_result(results, seq_len, window, shift):
    df_dict = {k: v.flatten() for k, v in results['score'].items()}
    result_df = pd.DataFrame(df_dict)
    #aggregate the scores
    scores =  result_df.mean(axis=1)

    return aggregate_postion_score(seq_len=seq_len, window=window, shift=shift, 
                    n_prediction=len(scores), scores=scores.to_numpy())



def __get_RT_IB_helper(group_df, seq, window, shift):
    one_hot_seqs = get_oneHot_sequence(seq, window, shift)
    all_results = {}
 
    with concurrent.futures.ProcessPoolExecutor(N_PROCESS) as executor:
            results = [executor.submit(get_IB_model_prediction, model_id, one_hot_seqs) 
                                                for model_id in group_df['id'].tolist()]
            for f in concurrent.futures.as_completed(results):
                cur_result = f.result()
              
                processed = process_result(cur_result, len(seq), window, shift)
                all_results[cur_result['model_id']] = processed[0]

    return all_results

def get_RT_IB_predictions(model_df, seq, shift):
    # get the input shape fromt he model df
    model_df.loc[:, 'input_shape'] = model_df['experiment_details'].apply(get_input_shape)

    # on each different input shape run the model (need to create subseq in the input shape length)
    grouped = model_df.groupby('input_shape')
    all_results = {}
    # running on each group size in paralel
    with concurrent.futures.ProcessPoolExecutor(max(len(grouped.groups),1)) as executor:
        results = [executor.submit(__get_RT_IB_helper, grouped.get_group(group), seq, group, shift) 
                                            for group in grouped.groups]
        for f in concurrent.futures.as_completed(results):
            result = f.result()
            all_results.update(result)
    return pd.DataFrame(all_results)



def get_RT_original_predictions(model_df, seq, shift, seq_id='', window=16):
    fasta_path = fasta_from_seq_string(seq, seq_id, window, shift)
    model_list = save_model_list(model_df)
    result_df = get_original_model_prediction(fasta_path, model_list)
    aggregate = aggregate_postion_score(seq_len=len(seq), window=window, shift=shift, 
                        n_prediction=len(result_df), scores=result_df.to_numpy())
    result = pd.DataFrame(aggregate.T, columns=result_df.columns)
    return result

import os


def get_cached_prediction(seq_id)->pd.DataFrame:
    path = f'cached_predictions/{seq_id}.pkl'
    if os.path.exists(path):
        return pd.read_pickle(path)
    return pd.DataFrame()
    
def save_cached_prediction(df, seq_id):
    if len(seq_id):
        path = f'cached_predictions/{seq_id}.pkl'
        df.to_pickle(path)

def get_RT_prediction(protein_list, seq, shift, seq_id=''):
    # Get the cached prediction
    cached_df = get_cached_prediction(seq_id)

    # Get the available and non-available models
    avilable_models, nonavilable_models = get_model_df(protein_list)

    # Filter the cached_df to only include columns that are in available_models
    is_in_available_models = cached_df.columns.isin(avilable_models.id)
    cached_df = cached_df[cached_df.columns[is_in_available_models]]

    # Get the IB_generated models that are not in cached_df
    is_IB_generated = avilable_models.source == 'IB_generated'
    IB_models = avilable_models[is_IB_generated]
    is_in_cached_df = IB_models.id.isin(cached_df.columns)
    IB_models = IB_models[~is_in_cached_df]

    # Get the db_original models that are not in cached_df
    is_db_original = avilable_models.source == 'db_original'
    original_models = avilable_models[is_db_original]
    is_in_cached_df = original_models.id.isin(cached_df.columns)
    original_models = original_models[~is_in_cached_df]

    print(f'proteins: {protein_list}, dtype = {type(protein_list)}')

    # Use a ProcessPoolExecutor to get the IB and original predictions
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        IB_result = get_predictions(executor, get_RT_IB_predictions, IB_models, seq, shift)
        original_result = get_predictions(executor, get_RT_original_predictions, original_models, seq, shift, seq_id)

    # Concatenate the results and the cached_df
    df = pd.concat([IB_result, original_result, cached_df], axis=1)

    # Save the cached prediction if any change accured and return the DataFrame
    if len(original_models) or len(IB_models):
        save_cached_prediction(df, seq_id)
    return df

def get_predictions(executor, prediction_function, models, seq, shift, seq_id=None):
    if len(models) > 0:
        if seq_id:
            return executor.submit(prediction_function, models, seq, shift, seq_id).result()
        else:
            return executor.submit(prediction_function, models, seq, shift).result()
    else:
        return pd.DataFrame()





def main():
    protein_list = "SRY, SOX9, SOX8, SOX10, DMRT1, GATA4, SF1, NR5A1, WT1, FOXL2, RUNX1, LHX9, EMX2, TCF3, TCF12, LEF1, ESR1, ESR2, AR".replace(' ','').lower().split(',')
    seq = 'CTGAAGTGCATTTCACATAACTCTAGTGCCAGCCCCTGCCCCAAACTAGCAGCCCTGCATCTTTATTTTCTACAAAACCCTGCCCCCGCAGTAACCATGTGTGCCCTCCTCTCCCAGGGGCCAAGCCTGCCCCCAGCCCTGGATCTCACTTGGCAGGGACTGAGAACACTCGCTGGGCCACAGGGGTCACAGGCTGCAGGGCCCTGACCCCAGCCCCCAGGCAGCCAGGATGTTATTGGCGAGGGGCCCTGCCCCAGGTCTATAGCTGATACCGTGTCTCCAGGGCAGGTACTGCAGCTCC'
    shift = 4
    print(get_RT_prediction(protein_list,seq, shift))

if __name__ == '__main__':
    main()