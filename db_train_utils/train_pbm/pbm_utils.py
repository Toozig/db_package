import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import concurrent.futures
from ..train_global_args import *
from ..train_global_function import get_parms_dict,process_result,copy_model_data, get_n_save_final_df, run_deepBind_expiriment
from ..parameter_generate import generate_variables_configurations
from ..db_train_utils import oneHot_encode
from ..deepbind_interface import DeepbindModel



TMP_DIR = '/tmp/toozig/'


def get_db_pbm_object(protein, species, array_design, cite, input_shape,
                        source_path=None):
    experiment_details = {'array_design': array_design}
    return DeepbindModel(protein, species, 'PBM', experiment_details, cite,
                              input_shape, source_path=source_path)

def normalize_PBM_target(pbm_df, sginal_col='mean_signal_intensity', background_col='mean_background_intensity'):#, deepBind_data):
    """
    description:
        A major component of the 2013 revised challenge is comprehensive preprocessing of
    the probe intensities. Each competing algorithm is evaluated using the best
    combination of up to 8 preprocessing steps (c.f. Weirauch et al.22
    , Supplementary
    Note 3). We used the spatially de-trended data provided by the organizers, which was
    reported to improve the performance of all algorithms. For DeepBind we only
    evaluated two additional pre-processing possibilities from the DREAM5 paper:
    subtracting each probe’s median intensity, or dividing by each probe’s median
    intensity. The per-probe normalization is intended to factor out biases in the
    microarray design or experiments; we computed each probe’s median intensity across
    all 66 experiments for that microarray design. Dividing by median intensity improved
    the overall PBM performance of DeepBind from 3rd place (no pre-processing) to 1st
    place overall.-
        
    input:
        pbm: PBMFile, PBM data
        db_id: int, deepBind parameters id
    output:
        pbm_data: ndarray, normalized PBM data
    """
    # score_col = deepBind_data['target_col']
    # median_intensity_col = deepBind_data['median_intensity_col']
    norm_intensity = pbm_df[sginal_col] / pbm_df[background_col]
    result =  norm_intensity.to_numpy()
    # raise error if value is nan
    if np.isnan(result).any():
        raise ValueError("nan value in the normalized data")    
    return result


def get_encoded_pbm(data_path, sginal_col, background_col, seq_col, to_norm=True):
    pbm_df = pd.read_csv(data_path, sep='\t')
    pbm_df = pbm_df.dropna(subset=[seq_col])
    # take the seq length with maximux val
    seq_len = pbm_df[seq_col].str.len().value_counts().idxmax()
    pbm_df = pbm_df[pbm_df[seq_col].str.len() == seq_len]
    X = np.array([oneHot_encode(seq) for seq in pbm_df[seq_col].to_numpy()])
    y = normalize_PBM_target(pbm_df, sginal_col, background_col) if to_norm else pbm_df[sginal_col].to_numpy()
    return X, y

def process_pbm_data(data_path, data_path2, to_norm, sginal_col, background_col, seq_col, split=0.15):
    X1,y1 = get_encoded_pbm(data_path, sginal_col, background_col, seq_col)
    # if there are two data sets (HK/ME) create the second data set, split otherwise
    if len(data_path2):
        X2,y2 = get_encoded_pbm(data_path2, sginal_col, background_col, seq_col)
        return X1,y1, X2,y2
    else:
        return train_test_split(X1, y1, test_size=split, random_state=42)
    

def __get_test_set(train_set, test_set_path):
    if not len(test_set_path):
        return train_set
    return 'HK' if train_set == 'ME' else 'ME'


def get_pbm_config(conf_df,data_path,data_path2,array_design,protein,spicies,cite, seq_col,
                    to_norm, sginal_col, background_col, version):
    seq_len = pd.read_csv(data_path, sep='\t', nrows=1)[seq_col].str.len().max()
    obj_model = get_db_pbm_object(protein, spicies, array_design, cite, (seq_len, 4), data_path)
    X_train,y_train, X_test ,y_test = process_pbm_data(data_path, data_path2, to_norm, sginal_col, background_col, seq_col)
    conf_df[TRAIN_SET] = data_path
    conf_df[TEST_SET] = __get_test_set(array_design, data_path2)
    conf_df[EXP_ID] = array_design +'.' + obj_model.get_protein() +'.' + obj_model.get_id() +'_' + conf_df.index.astype(str)
    configuration = (conf_df, X_train,y_train, X_test ,y_test, obj_model, version, False)
    return configuration


def generate_pbm_conf_df(n_exp, commetAPIKey, to_norm, version):
    configuration_list = generate_variables_configurations(n_exp)
    configuration_df = pd.DataFrame(configuration_list)
    configuration_df[BINARY] = False
    configuration_df['version'] = version
    configuration_df[COMMET_API_KEY_ARG] = commetAPIKey
    configuration_df['norm'] = to_norm
    return configuration_df


def get_configurations(protein, spicies, n_exp, HK_path, cite_HK, ME_path, cite_ME, to_norm, sginal_col, background_col, seq_col, version, commetAPIKey):
    configuration_df = generate_pbm_conf_df(n_exp, commetAPIKey, to_norm, version)
    conf_list =[]
    if len(HK_path):
        hk_conf = get_pbm_config(configuration_df.copy(), HK_path, ME_path, 'HK', protein, spicies,
                                  cite_HK, seq_col, to_norm, sginal_col, background_col, version)
        conf_list.append(hk_conf)
    if len(ME_path):
        me_conf = get_pbm_config(configuration_df.copy(), ME_path, HK_path, 'ME', protein, spicies,
                                  cite_ME, seq_col, to_norm, sginal_col, background_col, version)
        conf_list.append(me_conf)
    return conf_list


def run_final_deepBind_expirement(final_df,configuration_df,exp_id,X,y,obj_model, version):
        # split the data to 97% train and 3% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)
        top_ten = final_df.sort_values(PEARSON_VALIDATION_COL, ascending=False).iloc[:10,:]
        top_ten = top_ten[[EXP_ID, TRAIN_SET, TEST_SET] + configuration_df.columns.tolist()]
        top_ten[EXP_ID] = top_ten[EXP_ID].str.replace("_train", '') 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_deepBind_expiriment,df, X_train,y_train, X_test ,y_test, obj_model, version, True ) for _,row in top_ten.iterrows()]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            df = get_n_save_final_df(results,top_ten, exp_id, False)
        return df, results

def train_on_pbm_data(protein, spicies, n_exp,HK_path = '',cite_HK='',ME_path = '',  cite_ME='',to_norm=True,
             sginal_col='mean_signal_intensity',
               background_col='mean_background_intensity',
               seq_col='pbm_sequence',
               version= 'original',
               commetAPIKey=''):
    conf_list = get_configurations(protein, spicies, n_exp, HK_path, cite_HK, ME_path, cite_ME, to_norm,
                                    sginal_col, background_col, seq_col, version, commetAPIKey)
    same_data = not len(HK_path) or not len(ME_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each configuration to the executor for parallel execution
        futures = [executor.submit(run_deepBind_expiriment, *conf) for conf in conf_list]
        for future in concurrent.futures.as_completed(futures):
            exp_id, df, results, model_data = future.result()
            print(f"Finished expirement {exp_id}")
            final_df = get_n_save_final_df(results,df,exp_id ,same_data)
            if same_data:
                path = final_df[TRAIN_SET].values[0]
                X, y =  get_encoded_pbm(path, sginal_col, background_col, seq_col)
                df, results = run_final_deepBind_expirement(final_df,df,exp_id,X,y,model_data,version)
            process_result(results, df, exp_id)
            model_data.save_model_to_table()
            print('finished running final test')
    df = get_n_save_final_df(results,top_ten,exp_id, False)
    print('processed results')
    process_result(results, df, exp_id)
    global_model_data['model_id'] = exp_id
    # copy the data to the output folder
    copy_model_data(df[POSITIVE_DATA].values[0], exp_id)
    # add model to the model tabel
    model_object.save_model_to_table()
    print(f'finished saving models {exp_id}')
    


