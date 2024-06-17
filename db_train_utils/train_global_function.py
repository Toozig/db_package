import comet_ml
from comet_ml import Experiment
from .train_global_args import *
from .deepbind_interface import DeepbindModel
import pandas as pd
import concurrent.futures
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras import backend as K
from keras.saving import register_keras_serializable
from typing import Dict
import json
import os
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import MaxPooling1D

from .model_design import get_model
import os
import sys

if 'config' not in  sys.modules:

    sys.path.append(os.path.dirname(__file__))
    from  .. import  config 
else:
    import config 



## variables
# numbers od strides for the convolution layer
STRIDES = 1
# number of cross validation of model training
RELU = 'relu'
LINEAR = 'linear'
SIGMOID = 'sigmoid'
MSE = 'mse'

CROSS_VAL_NUM = 3
BATCH_SIZE = 64

# model table cols
ID_COL = 0

MODEL_PATH = config.get_IB_model_path()
MODEL_TABLE = config.get_model_table_path()

YARON_PARMS = {
    N_MOTIF : 128,
    LENGTH_MOTIF : 5,
    DROPOUT_RATE : 0,
    LEARNING_RATE : 0.001, # default value
    HIDDEN_LAYER : True
}


def copy_model_data(json_path, model_id):
    """
    copy the model data to the given model id
    """
    new_path = f'{MODEL_PATH}/{model_id}/original_data'
    if json_path.endswith('.json'):
        model_data = json.load(open(json_path))
        model_data['file_data']['path'] = new_path
        with open(f'{new_path}/model_data.json', 'w') as f:
            json.dump(model_data, f)
        file_path = model_data['file_data']['path']
    else:
        file_path = json_path
    # copy the file to the model directory
    os.makedirs(new_path, exist_ok=True)
    os.system(f'cp {file_path} {new_path}/.')


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


def train_model_k_fold(model, X_train, y_train, X_test,y_test, learning_steps):
    # print the shape of the data

    early_stopping = EarlyStopping(monitor='val_loss',
                                      patience=5, verbose=0, 
                                      restore_best_weights=True)

    kf = KFold(n_splits=CROSS_VAL_NUM, shuffle=True, random_state=42)

    evaluation_dict = {}
    # Loop through the folds
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        # Split the data into training and validation sets for this fold
        kX_train, X_val = X_train[train_index], X_train[val_index]
        ky_train, y_val = y_train[train_index], y_train[val_index]
        model.fit(kX_train, ky_train, epochs=learning_steps, 
                            batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val),
                    callbacks=[early_stopping], verbose=0)
        print(f"Fold {fold} finished")
    # Evaluate the model on the test set or perform any other required actions
    loss, metric= model.evaluate(X_train, y_train)
    evaluation_dict['validation'] = {'loss': loss, 'metric': metric}
    loss, metric = model.evaluate(X_test, y_test)
    evaluation_dict['test'] = {'loss': loss, 'metric': metric}
    return evaluation_dict

def regular_train(model, X_train, y_train, X_test, y_test, learning_steps):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5, verbose=0, 
                                   restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=learning_steps, 
              batch_size=BATCH_SIZE,
              validation_split=0.2,  # Use a validation split instead of cross-validation
              callbacks=[early_stopping], verbose=0)
    evaluation_dict = {}
    # Evaluate the model on the test set
    loss, metric= model.evaluate(X_train, y_train)
    evaluation_dict['validation'] = {'loss': loss, 'metric': metric}
    loss, metric = model.evaluate(X_test, y_test)
    evaluation_dict['test'] = {'loss': loss, 'metric': metric}
    return model, evaluation_dict


def eval_model(model, x,y)->Dict[str,float]:
    test_results = model.evaluate(x, y)
    evaluation_dict = {name: value for name, value in zip(model.metrics_names, test_results)}
    return evaluation_dict



def add_model_to_table(model_id, protein, species, experiment, experiment_details, cite, input_shape, source=None):
    """
    add a given model to the model table
    """
    model_df = pd.read_csv(MODEL_TABLE, sep='\t')
    if model_id in model_df[model_df.columns[ID_COL]].values:
        print(f'model with id {model_id} already exists in the model table')
        print(f'new model id generated: {model_id}')
    experiment_details[INPUT_SHAPE] = input_shape
    new_model_data = {
        'model_id': model_id,
        'protein': protein,
        'species': species,
        'experiment': experiment,
        'experiment_details': experiment_details,
        'cite': cite,
        'source': source or 'IB_generated'
    }
    new_model_df = pd.DataFrame([new_model_data])
    new_model_df.to_csv(MODEL_TABLE, sep='\t', mode='a', header=False, index=False)
    return (f'model with id {model_id} saved to the model table - {MODEL_TABLE}')




def get_parms_dict(input_shape,
    train_set,
    test_set,
    learning_rate,
    n_motif, 
    length_motif, 
    dropout_rate, 
    hidden_layer, 
    learning_step, 
    expirement_id,
    binary):
    return {
        INPUT_SHAPE : input_shape,
        TRAIN_SET: train_set,
        TEST_SET: test_set,
        LEARNING_RATE: learning_rate,
        N_MOTIF: n_motif,
        LENGTH_MOTIF: length_motif,
        DROPOUT_RATE: dropout_rate,
        HIDDEN_LAYER: hidden_layer,
        LEARNING_STEP: learning_step,
        EXP_ID: expirement_id,
        BINARY :binary,
    }


def fix_metric_name(binary, evaluation_dict):
    metric_name =  'auc' if binary else 'pearson_correlation'
    metric_val = evaluation_dict['validation'].pop('metric')
    evaluation_dict['validation'][metric_name] = metric_val
    metric_val = evaluation_dict['test'].pop('metric')
    evaluation_dict['test'][metric_name] = metric_val
    # evaluation_dict = {f'{key}_train': value for key, value in evaluation_dict.items()}
    return evaluation_dict

def hyper_parameter_search(conf_dict, model_version, x_train, y_train, x_test, y_test):
    model = get_model(model_version, conf_dict)
    learning_step = conf_dict[LEARNING_STEP]
    evaluation_dict =  train_model_k_fold(model, x_train, y_train, x_test, y_test, learning_step) 
    evaluation_dict = fix_metric_name(conf_dict[BINARY], evaluation_dict)
    return {conf_dict[EXP_ID] : evaluation_dict}


def middle_model_training(conf_dict, model_version, x_train, y_train, x_test, y_test):
    model = get_model(model_version, conf_dict)
    learning_step = conf_dict[LEARNING_STEP]
    model, evaluation_dict = regular_train(model, x_train, y_train, x_test, y_test, learning_step)
    evaluation_dict = fix_metric_name(conf_dict[BINARY], evaluation_dict)

    return {conf_dict[EXP_ID] : evaluation_dict}

def final_model_training(conf_dict, model_version, x_train, y_train, x_test, y_test):
    model = get_model(model_version, conf_dict)
    learning_step = conf_dict[LEARNING_STEP]
    model, evaluation_dict = regular_train(model, x_train, y_train, x_test, y_test, learning_step)
    evaluation_dict = fix_metric_name(conf_dict[BINARY], evaluation_dict)

    return {conf_dict[EXP_ID] : evaluation_dict},  model


def run_expirements(df: pd.DataFrame,train_func):
    if DEBUG:
        print("Running experiments, df shape:", df.shape)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each experiment as a separate task
        futures = [executor.submit(train_func, row) for _, row in df.iterrows()]

        # Wait for all experiments to finish
        concurrent.futures.wait(futures)

        # Collect the results in a list
        results_list = [future.result() for future in futures]
        print("the len of results_list", len(results_list))
        if DEBUG:
            print("Finished all experiments")
        return results_list
    

### RESULT PROCESSING #####


def prepare__score_df(results):
    [i['score_dict'].update({'expirement_id': i['expirement_id']}) for i in results]
    score_df = pd.DataFrame([i['score_dict'] for i in results])
    return score_df.set_index('expirement_id')


def __get_validation_pearson(val_dict):
    return val_dict[VAL_STR]['pearson_correlation']

def save_models(top_ten_items, results ,  output_dir):
    save_path = f'{output_dir}/models'
    # create directory if not exits
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for _, cur_model_id in top_ten_items:
        result_list = [result['model'] for result in results if result[EXP_ID] == cur_model_id]
        if DEBUG:
            print(f"model_id: {cur_model_id}, result_lis: {result_list}")
        model = result_list[0]
        save_name = f'{save_path}/{cur_model_id}.keras'
        model.save(save_name)
        print(f"Model {cur_model_id} saved to {save_name}")
    

def save_model(model, exp_id,output_dir):
    save_path = f'{output_dir}/models/submodels'
    # create directory if not exits
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = f'{save_path}/{exp_id}.keras'
    model.save(save_name)
    print(f"Model {exp_id} saved to {save_name}")

def save_result_df(df, output_dir,name) : #final_result=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(f'{output_dir}/{name}_results.csv', index=False)
    print(f"Results saved to {output_dir}/results.csv")

def save_final_df(df,save_path, model_id):
    if COMMET_API_KEY_ARG in df.columns:
        df.drop(columns=[COMMET_API_KEY_ARG])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(f'{save_path}/{model_id}_parameters.csv', index=False)
    print(f"Results saved to {save_path}/{model_id}_parameters.csv")



def create_output_dir(data_id, init=0):
    dir_name = f'{MODEL_PATH}/{data_id}.{init}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        return dir_name
    return create_output_dir(data_id, init+1)
    

def get_output_dir(model_id):
    return f'{MODEL_PATH}/{model_id}'


def get_n_save_final_df(results:list,config_df:pd.DataFrame, model_id:str, same_data:bool):
    output_dir = get_output_dir(model_id)
    score_df = [pd.DataFrame(pd.json_normalize(result['score_dict']).squeeze().rename(result[EXP_ID])).T for result in results]

    score_df = pd.concat(score_df)
    final_df = pd.concat([config_df.set_index(EXP_ID), score_df], axis=1)
    final_df = final_df.reset_index()
    final_df = final_df.rename(columns={'index': EXP_ID})
    if same_data:
        final_df[EXP_ID] = config_df[EXP_ID] + "_train"
        model_id += "_train"
    save_final_df(final_df, output_dir, model_id)
    return final_df


def process_result(results, df, model_id,top_n=10):
    score_df = [pd.DataFrame(pd.json_normalize(result['score_dict']).squeeze().rename(result[EXP_ID])).T for result in results]
    score_df = pd.concat(score_df)

    
    score_model_dict = {__get_validation_pearson(result['score_dict']): result[EXP_ID] for result in results}
    if DEBUG:
        top_n = len(score_model_dict)
        # print(results)
    top_ten_items = list(sorted(score_model_dict.items(), key=lambda x: x[0], reverse=True))[:top_n]
    # save top 10 models
    output_dir = f'{MODEL_PATH}/{model_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_models(top_ten_items, results, output_dir)


# def run_single_deepBind_expirement(row, X_train, y_train, X_test, y_test, version, final):
#     parameter_dict = get_parms_dict(X_train.shape[1:], row[TRAIN_SET],
#                                     row[TEST_SET], row[LEARNING_RATE],
#                                     row[N_MOTIF], row[LENGTH_MOTIF],
#                                     row[DROPOUT_RATE], row[HIDDEN_LAYER],
#                                     row[LEARNING_STEP], row[EXP_ID],row[BINARY])

#     model = get_model(version, parameter_dict)
#     train_func = train_final_model if final else train_model
#     model, train_eval_dict = train_func(model, X_train, y_train, X_test, y_test, parameter_dict[LEARNING_STEP])
#     val_eval_dict = eval_model(model, X_train, y_train)
#     test_eval_dict = eval_model(model, X_test, y_test)
#     val_dict = {TRAIN_STR: train_eval_dict, VAL_STR: val_eval_dict, TEST_STR: test_eval_dict}

#     if DEBUG:
#         print(f"finished expirement {row[EXP_ID]}")

#     return {EXP_ID: row[EXP_ID], 'model': model, 'score_dict': val_dict}


# def run_deepBind_expiriment(df, X_train,y_train, X_test ,y_test, obj_model, version, final):

#     exp_id = obj_model.get_id()
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         # Submit each configuration to the executor for parallel execution
#         futures = [executor.submit(run_single_deepBind_expirement, row,  X_train,y_train, X_test ,y_test, version, final) for _,row in df.iterrows()]
#         results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
#     return exp_id, df, results, obj_model


