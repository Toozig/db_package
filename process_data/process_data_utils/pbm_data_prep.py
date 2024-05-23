import os
import numpy as np
from ...db_train_utils.db_train_utils import oneHot_encode
from . import general_data_process as gdp


def normalize_PBM_target(pbm_df, sginal_col='mean_signal_intensity', background_col='mean_background_intensity'):#, deepBind_data):
    """
    Normalize PBM data, as described in the deepBind paper
    input:
        pbm: PBMFile, PBM data
        db_id: int, deepBind parameters id
    output:
        pbm_data: ndarray, normalized PBM data
    """
    norm_intensity = pbm_df[sginal_col] / pbm_df[background_col]
    result =  norm_intensity.to_numpy()
    # raise error if value is nan
    if np.isnan(result).any():
        raise ValueError("nan value in the normalized data")    
    return result



def get_pbu_exp_details(HKME, to_norm, linker, path_ME, path_HK, path ,signal_col, background_col, seq_col):

    exp_details = {'HKME': HKME, 'to_norm': to_norm, 'linker': linker,
                    'signal_col': signal_col, 'background_col': background_col, 'seq_col': seq_col}
    if HKME:
        exp_details['path_ME'] = path_ME
        exp_details['path_HK'] = path_HK
    else:
        exp_details['path'] = path
    return exp_details
    

def get_encoded_pbm(data_path, sginal_col, background_col, seq_col, to_norm, linker='', header='infer'):
    pbm_df = open_file(data_path, header=header)
    pbm_df = pbm_df.dropna(subset=[seq_col])
    pbm_df[seq_col] = pbm_df[seq_col].str.replace(linker, '')
    # take the seq length with maximux val
    seq_len = pbm_df[seq_col].str.len().value_counts().idxmax()
    pbm_df = pbm_df[pbm_df[seq_col].str.len() == seq_len]
    X = np.array([oneHot_encode(seq) for seq in pbm_df[seq_col].to_numpy()])
    y = normalize_PBM_target(pbm_df, sginal_col, background_col) if to_norm else pbm_df[sginal_col].to_numpy()
    return X, y


def get_MEHK_project_dir(id1,id2):
    """
    HK and ME are the same project, so the project dir is the same
    """
    opt1 = f'{gdp.DATA_DIR}/{id1}_{id2}'
    if os.path.exists(opt1):
        return opt1
    return f'{gdp.DATA_DIR}/{id2}_{id1}'

def process_pbm_data(data_id,  path, signal_col, background_col, seq_col, to_norm, linker, header, data2_id=''):
    project_dir = f'{gdp.DATA_DIR}/{data_id}'
    if len(data2_id):
        project_dir = get_MEHK_project_dir(data_id, data2_id)
    X_data, y_data = get_encoded_pbm(path, signal_col, background_col, seq_col, to_norm, linker, header=header)
    X_path, y_path, new_path = gdp.save_data(X_data, y_data, project_dir, data_id, path)
    n_samples = X_data.shape[0]
    return X_path, y_path, n_samples, new_path


def open_file(path, header,sep='\t' ,n_rows=None):
    return gdp.open_file(path, header, sep, n_rows)



