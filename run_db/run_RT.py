import numpy as np
import pandas as pd
import concurrent.futures
import os

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
## set the sworkdir tto the file folder
os.chdir(os.path.dirname(__file__))

from .run_general import get_subsequences, fasta_from_seq_string, get_model_df
from .IB_function import get_IB_model_prediction, get_input_shape, get_IB_models
from ..db_train_utils import oneHot_encode
from .original_function import get_original_model_prediction, save_model_list
from ..db_train_utils.model_design import get_ensemble_model 
from .process_P2 import aggregate_postion_score 
import os
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')
TMP_DIR =config['GENERAL']['TMP_DIR'] 
N_PROCESS=60


def get_oneHot_sequence(seq, window, shift):
    subseqs = get_subsequences(seq, window, shift)
    one_hot_seqs = np.stack([oneHot_encode(record) for record in subseqs])
    return  one_hot_seqs
    

def process_result(results, seq_len, window, shift, per_position=True):
    # per position - score per position - other wise  maxscore per window
    df_dict = {k: v.flatten() for k, v in results['score'].items()}
    result_df = pd.DataFrame(df_dict)
    #aggregate the scores
    scores =  result_df.mean(axis=1)
    if per_position:
        result =  aggregate_postion_score(seq_len=seq_len, window=window, shift=shift, 
                    n_prediction=len(scores), scores=scores.to_numpy())
    else:
        result = scores.max()
    return result, result_df



def __get_RT_IB_helper(group_df, seq, window, shift):
    one_hot_seqs = get_oneHot_sequence(seq, window, shift)
    all_results = {}
 
    with concurrent.futures.ProcessPoolExecutor(N_PROCESS) as executor:
            results = [executor.submit(get_IB_model_prediction, model_id, one_hot_seqs) 
                                                for model_id in group_df['id'].tolist()]
            for f in concurrent.futures.as_completed(results):
                cur_result = f.result()
              
                processed = process_result(cur_result, len(seq), window, shift, per_position=True)
                print(processed)
                all_results[cur_result['model_id']] = processed[0][0]

    return all_results


def get_window(shape):
    if isinstance(shape, int):
        return shape
    return shape[0]

def get_RT_IB_predictions(model_df, seq, shift):
    # get the input shape fromt he model df
    model_df.loc[:, 'input_shape'] = model_df['experiment_details'].apply(get_input_shape)

    # on each different input shape run the model (need to create subseq in the input shape length)
    grouped = model_df.groupby('input_shape')
    all_results = {}
    # running on each group size in paralel
    with concurrent.futures.ProcessPoolExecutor(max(len(grouped.groups),1)) as executor:
        results = [executor.submit(__get_RT_IB_helper, grouped.get_group(group), seq, get_window(group), shift) 
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




################################## max function ################################


def build_ensemble_model(model_ids):
    model_list = get_IB_models(model_ids)
    if len(model_list) == 1:
        model = model_list[0]
    else:
        model = get_ensemble_model(model_list)
    return model



def process_prediction(predictions, model_ids, ids):
    if not isinstance(predictions, list):
        predictions = [predictions]
    df = pd.DataFrame([i.flatten() for i in predictions]).T
    df.columns = model_ids
    df['seq_id'] = ids
    # group each col by 'seq_id' and get the max value
    df = df.groupby('seq_id').max()
    return df




def create_subsequences(input_fasta, output_fasta, window, shift):
    with open(input_fasta, "r") as input_handle, open(output_fasta, "w") as output_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            subseq = get_subsequences(str(record.seq), window, shift)
            subseq_records =[SeqRecord(Seq(s), id=f"{record.id}_{i}", description="") for i, s in enumerate(subseq)]
            SeqIO.write(subseq_records, output_handle, "fasta-2line")
    return output_fasta



def get_max_IB_fasta(group_df,fasta):
    model_ids = group_df['id'].tolist()
    model = build_ensemble_model(model_ids)
    fasta = SeqIO.parse(fasta, 'fasta')
    seq_data = {'_'.join(seq.id.split('_')[:-1]):oneHot_encode(str(seq.seq)) for seq in fasta}
    ids = list(seq_data.keys())
    seqs = list(seq_data.values())
    seqs = np.stack(seqs)
    predictions = model.predict(seqs)
    result_df = process_prediction(predictions, model_ids, ids)
    result_df.to_csv('tmp_IB.csv')
    return result_df

def __get_max_IB_helper(model_df, fasta, shift, group):
    window = get_window(group)
    fasta_name = '.'.join(os.path.basename(fasta).split('.')[:-1]) + f'IB_w{window}_s{shift}_subseq.fa'
    tmp_file = os.path.join(TMP_DIR, fasta_name)
    tmp_file = create_subsequences(fasta, tmp_file, window, shift)
    return get_max_IB_fasta(model_df, tmp_file)

def get_max_IB_predictions_fasta(model_df, fasta, shift):
    model_df.loc[:, 'input_shape'] = model_df['experiment_details'].apply(get_input_shape)
    grouped = model_df.groupby('input_shape')
    all_results = pd.DataFrame()
    # running on each group size in paralel
    with concurrent.futures.ProcessPoolExecutor(max(len(grouped.groups),1)) as executor:
        results = [executor.submit(__get_max_IB_helper, grouped.get_group(group), fasta, shift, group) for group in grouped.groups]
        for f in concurrent.futures.as_completed(results):
            result = f.result()
            all_results = pd.concat([all_results, result], axis=1)

    return all_results




def get_max_original_predictions_fasta(model_df, fasta, shift, window=16):
    fasta_name = '.'.join(os.path.basename(fasta).split('.')[:-1]) + f'original_w{window}_s{shift}_subseq.fa'
    tmp_file = os.path.join(TMP_DIR, fasta_name)
    fasta_path = create_subsequences(fasta, tmp_file, window, shift)
    model_list = save_model_list(model_df)
    result_df = get_original_model_prediction(fasta_path, model_list)
    seqid = ['_'.join(seq.id.split('_')[:-1]) for seq in  SeqIO.parse(fasta_path, 'fasta')]
    result_df['seq_id'] = seqid
    result_df = result_df.groupby('seq_id').max()

    return result_df




def get_maximal_score_fasta(protein_list, fasta, shift):

    # Get the available and non-available models
    avilable_models, nonavilable_models = get_model_df(protein_list)

    # Get the IB_generated models that are not in cached_df
    is_IB_generated = avilable_models.source == 'IB_generated'
    IB_models = avilable_models[is_IB_generated]
    # Get the db_original models that are not in cached_df
    is_db_original = avilable_models.source == 'db_original'
    original_models = avilable_models[is_db_original]

    print(f'proteins: {protein_list}, dtype = {type(protein_list)}')
    result = []
    # Use a ProcessPoolExecutor to get the IB and original predictions
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(get_max_IB_predictions_fasta, IB_models, fasta, shift), 
                executor.submit(get_max_original_predictions_fasta, original_models, fasta, shift)]
        results = []
        for future in concurrent.futures.as_completed(futures):
            result_df = future.result()
            results.append(result_df)

    # Concatenate all result dataframes
    final_df = pd.concat(results, axis=1)
    final_df.to_csv('tmp.csv')
    return final_df





def get_max_original_predictions(model_df, seq, shift, seq_id='', window=16):
    fasta_path = fasta_from_seq_string(seq, seq_id, window, shift)
    model_list = save_model_list(model_df)
    result_df = get_original_model_prediction(fasta_path, model_list).max(axis=0)
    return result_df





def main():
    protein_list = "SRY, SOX9, SOX8, SOX10, DMRT1, GATA4, SF1, NR5A1, WT1, FOXL2, RUNX1, LHX9, EMX2, TCF3, TCF12, LEF1, ESR1, ESR2, AR".replace(' ','').lower().split(',')
    seq = 'CTGAAGTGCATTTCACATAACTCTAGTGCCAGCCCCTGCCCCAAACTAGCAGCCCTGCATCTTTATTTTCTACAAAACCCTGCCCCCGCAGTAACCATGTGTGCCCTCCTCTCCCAGGGGCCAAGCCTGCCCCCAGCCCTGGATCTCACTTGGCAGGGACTGAGAACACTCGCTGGGCCACAGGGGTCACAGGCTGCAGGGCCCTGACCCCAGCCCCCAGGCAGCCAGGATGTTATTGGCGAGGGGCCCTGCCCCAGGTCTATAGCTGATACCGTGTCTCCAGGGCAGGTACTGCAGCTCC'
    shift = 4
    print(get_RT_prediction(protein_list,seq, shift))

if __name__ == '__main__':
    main()