import os
import numpy as np
import pandas as pd
import re
from predictionArchiver import PredictionSaver


def generate_base_matrix(seq_len, window, shift, n_prediction):
    """
    create matrix of shape (n_prediction, seq_len) where each row is a vector of 1s and 0s
    that represent the position of the window in the sequence
    """

    base_vector = np.asarray([1]*window + [0]*(seq_len - window))
    base_matrix = [np.roll(base_vector, i*shift) for i in range(n_prediction - 1)]
    base_matrix.append( np.asarray([0]*(seq_len - window) + [1]*window ))
    base_matrix = np.asarray(base_matrix)
    return base_matrix



def aggregate_postion_score(seq_len,window,shift,scores, n_prediction=-1):
    n_prediction = n_prediction if n_prediction != -1 else len(scores)
    base_matrix = generate_base_matrix(seq_len, window, shift, n_prediction)
    results = scores.T[..., np.newaxis] * base_matrix[np.newaxis,... ]

    return results.mean(axis=1)


def merge_exist_df_with_new(df1,df2):
    # Use a list comprehension to build the to_drop list
    to_drop = [col for col in df2.columns if col in df1.columns and not df1[col].equals(df2[col])]
    df2.drop(columns=to_drop, inplace=True)
    return pd.concat([df1,df2], axis=1)



def save_p3_result(df, save_path):
        # checkes if the folder exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        # add to the file
        cur_df = pd.read_csv(save_path, sep='\t')
        df = merge_exist_df_with_new(cur_df, df)
    df.to_csv(save_path, sep='\t', index=False)
    # print(f'saved at {save_path}')
    return save_path



def extract_numbers_from_path(path):
    # Use regular expressions to find the numbers after 'w' and 's'
    window = int(re.search('w(\d+)', path).group(1))
    shift = int(re.search('s(\d+)', path).group(1))
    return window, shift



def read_file(file_path):
    try:
        df = pd.read_csv(file_path,  sep='\t', index_col=0)
    except pd.errors.EmptyDataError:
        print(f"File {file_path} is empty.")
        df = pd.DataFrame()
    return df

def get_score_per_position(p2_path, seq_len):
    saver = PredictionSaver()
    p2_pred_file = saver.get_prediction_file_by_path('P2', p2_path)
    window = p2_pred_file.window
    shift = p2_pred_file.shift
    df = read_file(p2_path)
    if df.empty:
        return df
    models = [f"{model}_w{window}s{shift}" for model in df.columns]
    scores = df.values
    score_per_position = aggregate_postion_score(seq_len,window,shift,len(df),scores)
    final_df = pd.DataFrame(score_per_position.T, columns=models)
    return final_df
