from ..db_train_utils import oneHot_encode
from ..deepbind_interface import DeepbindModel
from ..parameter_generate import generate_variables_configurations, genretate_fcn_configurations
from ..train_global_function import  process_result, get_n_save_final_df
from ..train_global_args import *
from Bio import SeqIO
import concurrent.futures
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

POSITIVE_DATA = 'positive_data'
NEGATIVE_DATA = 'negative_data'
CHIPS_SEQ_COL_REPLACE_DICT = { TEST_SET: POSITIVE_DATA, TRAIN_SET: NEGATIVE_DATA}


def get_db_chs_object(protein, species, experiment, lab, cite, input_shape,
                        source_path=None, cell_line='',antibody=''):
    experiment_details = {'lab': lab, 'cell_line': cell_line, 'antibody': antibody}
    return DeepbindModel(protein, species, experiment, experiment_details, cite,
                              input_shape, source_path=source_path)


def get_chip_processed_data(fasta_path, is_positive):
    # Load the fasta file
    fasta_sequences = SeqIO.parse(open(fasta_path), 'fasta')
    X = list(map(oneHot_encode, [str(fasta.seq) for fasta in fasta_sequences]))
  

    max_length = max([len(x) for x in X])
    # Pad the sequences to the same length
    X = np.array([np.pad(x, ((0, max_length - len(x)), (0, 0))) for x in X])
    X = np.array(X)
    y = [[1] if is_positive else [0]] * len(X)
    return X, np.array(y)



def get_train_test_data(positive_data, negative_data, to_split):
    x_pos, y_pos = get_chip_processed_data(positive_data, True)
    x_neg, y_neg = get_chip_processed_data(negative_data, False)
    x_train = np.concatenate((x_pos, x_neg))
    y_train =  np.concatenate((y_pos, y_neg))
    return x_train, y_train

def prepare_chs_train_test(x_train, y_train, to_split):
    if to_split:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
    else:
        # traiing on all data
        x_test, y_test = x_train, y_train
    return x_train, y_train, x_test, y_test

def prepare_chs_df(n_exp,db_chs_obj,commetAPIKey, positive_data, negative_data):
    configuration_list = generate_variables_configurations(n_exp)
    configuration_df = pd.DataFrame(configuration_list)
    # in chips seq - the positive data is the test set
    configuration_df[TEST_SET] = positive_data
    configuration_df[TRAIN_SET] = negative_data
    configuration_df[COMMET_API_KEY_ARG] = commetAPIKey
    configuration_df[BINARY] = True
    configuration_df[EXP_ID] = db_chs_obj.generate_id(db_chs_obj.cite) + '_' + configuration_df.index.astype(str)
    return configuration_df

def prepare_chs_ibis_df(n_exp,db_chs_obj,commetAPIKey, positive_data, negative_data):
    configuration_df = genretate_fcn_configurations(n_exp)
    # in chips seq - the positive data is the test set
    configuration_df[TEST_SET] = positive_data
    configuration_df[TRAIN_SET] = negative_data
    configuration_df[COMMET_API_KEY_ARG] = commetAPIKey
    configuration_df[BINARY] = True
    configuration_df[EXP_ID] =  db_chs_obj.generate_id(db_chs_obj.cite) + '_' + configuration_df.index.astype(str)
    return configuration_df


def process_init_result(config_df,results,exp_id):
    final_df = get_n_save_final_df(results,config_df,exp_id ,True)
    # means there is only one data set, need to train top 10 on all data
    top_ten = final_df.sort_values(PEARSON_VALIDATION_COL, ascending=False).iloc[:10,:]
    top_ten = top_ten.loc[:,list(set([EXP_ID, TRAIN_SET, TEST_SET] + config_df.columns.tolist()))]
    print(top_ten[EXP_ID])
    print(type(top_ten[EXP_ID]))
    top_ten[EXP_ID] = top_ten[EXP_ID].str.replace("_train", '') 
    return top_ten