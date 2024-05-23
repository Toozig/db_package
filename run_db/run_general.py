import os
import numpy as np
import pandas as pd
import sys
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ..db_train_utils.train_global_args import *

import concurrent.futures

SAVE_DIR = '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/DB_predictions'
ORIGINAL = 'db_original'
GENERATED = 'IB_generated'
IB_MODEL_RESULT = f'{SAVE_DIR}/{GENERATED}'
ORIGINAL_MODEL_RESULT = f'{SAVE_DIR}/{ORIGINAL}'
ORIGINAL_WINDOW_SIZE = 16
N_PROCESS = 90
SHIFT_PARAM = 4
DEBUG = False
MAX_ID_LEN = 100

TMP_SAVE_PATH = '/tmp/deepbind/'
MODEL_TABLE = '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models/model_table.tsv'
OUTPUT_DIR= '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models'
SAVE_DIR = f'{OUTPUT_DIR}/IB_models'
MODEL_TABLE = f'{OUTPUT_DIR}/model_table.tsv'




def get_model_df(protein_list=[], all_models=False):
    """
    get the models DF for the given protein list
    """
    protein_list = [i.lower() for i in protein_list]
    model_df = pd.read_csv(MODEL_TABLE, sep='\t', comment='#')
    if all_models or not len(protein_list):
        return model_df
    model_df['protein'] = model_df['protein'].str.lower()
    avilable_models = model_df[model_df['protein'].str.lower().isin(protein_list)]
    nonavilable_models = [i for i in protein_list if i.lower() not in avilable_models['protein'].str.lower().tolist()]
    return avilable_models, nonavilable_models



def get_subsequences(sequence, window, shift):
    """
    given a sequence, return a list of subsequences with the given window and shift
    """
    start = np.arange(0, len(sequence)-window, shift)
    start = np.append(start, len(sequence)-window)
    end = start + window
    # start, end = get_subsequence_s_e(len(sequence), window, shift)
    subseq = [sequence[s:e] for s, e in zip(start, end)]
    return subseq

def fasta_from_seq_string(seq_string,seq_id, window, shift):
    subseqs = get_subsequences(seq_string.upper(), window, shift)
    records = [f'>{seq_id[:MAX_ID_LEN]}_{i}\n{subseqs[i]}' for i in range(len(subseqs))]
    path = os.path.join(TMP_SAVE_PATH, 'fasta')
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = f'{path}/{seq_id[:MAX_ID_LEN]}.fa'
    with open(save_path, 'w') as f:
        f.write('\n'.join(records))
        # print(f'fasta file saved at {path}/{seq_record.id}.fa')
    return save_path

def fasta_from_seq_record(seq_record, window, shift):
    seq_string = str(seq_record.seq)
    seq_id = seq_record.id
    return fasta_from_seq_string(seq_string, seq_id, window, shift)


def get_P1_saving_path(model_type,model_id ,seq_id,window_size,shift):
    if model_type == ORIGINAL:
        save_dir = f'{ORIGINAL_MODEL_RESULT}/P1/{model_id}'
    else:
        save_dir = f'{IB_MODEL_RESULT}/P1/{model_id}'
    save_dir += f'w{window_size}s{shift}'
    return save_dir + f'/{seq_id}.tsv'

def get_P2_saving_path(model_type,seq_id,window_size,shift):
    save_path = ORIGINAL_MODEL_RESULT if model_type == ORIGINAL else IB_MODEL_RESULT
    save_path = f'{save_path}/P2/w{window_size}s{shift}/'
    return save_path + f'{seq_id[:MAX_ID_LEN]}.tsv'

def save_P1_model(df, model_type,model_id, seq_id, window_size, shift):
    save_path = get_P1_saving_path(model_type, model_id, seq_id, window_size, shift)
    save_prediction_df(df, save_path)

    return save_path

def save_prediction_df(df, save_path):
    # checkes if the folder exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        # add to the file
        cur_df = pd.read_csv(save_path, sep='\t', index_col=0)
        df = cur_df.combine_first(df)
    df.to_csv(save_path, sep='\t')
    # print(f'saved at {save_path}')
    

def get_reversed_record(record):
    # Create a new Seq object with the reversed sequence
    reversed_seq = Seq(str(record.seq.reverse_complement()))

    # Create a new SeqRecord with the reversed sequence and modified ID
    reversed_record = SeqRecord(reversed_seq, id=record.id + '_reversed', description="")
    # print(f'reversed record created for {record.id}')
    return reversed_record


