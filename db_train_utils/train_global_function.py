import comet_ml
from comet_ml import Experiment
from .train_global_args import *
from .deepbind_interface import DeepbindModel
import pandas as pd
import concurrent.futures
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.layers import TimeDistributed, LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras import backend as K
from keras.saving import register_keras_serializable
from typing import Dict
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import MaxPooling1D
import os
## variables
# numbers od strides for the convolution layer
STRIDES = 1
# number of cross validation of model training
CROSS_VAL_NUM = 3
BATCH_SIZE = 64



MODEL_PATH = '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models/IB_models'


RELU = 'relu'
LINEAR = 'linear'
SIGMOID = 'sigmoid'
MSE = 'mse'


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

def create_comet_project(commet_API,TF_name):
    api = comet_ml.api.API(commet_API)
    workspace = api.get('deepBind')
    if TF_name.lower() in workspace:
        print(f"Project {TF_name} already exists")
        return
    description = f"DeepBind model for {TF_name} "

    api.create_project('deepBind', project_name=TF_name, project_description=description,
        public=True)

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






def build_model(n_motif, length_motif, input_shape, dropout_rate,
                learning_rate,hidden_layer, binary=False, **kwargs):
    # print all arguments
    # print(f"n_motif: {n_motif}, length_motif: {length_motif}, input_shape: {input_shape}, dropout_rate: {dropout_rate}, learning_rate: {learning_rate}, hidden_layer: {hidden_layer}")

    model = Sequential()
    model.add(Conv1D(filters=n_motif,
                     kernel_size=(length_motif,),
                     strides=STRIDES,
                     activation='relu',
                     input_shape=input_shape,
                    #  kernel_initializer=LogUniformInitializer(motif_initializer_seed),
                    #  kernel_regularizer=regularizers.l1(motif_regularizer)
                    )
                     )
    model.add(MaxPooling1D(pool_size=(length_motif -  1,)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Flatten()) 
    if hidden_layer:
        model.add(Dense(units=32,
                        activation=RELU,
                        ))

    model.add(Dense(units=1,
                    activation= SIGMOID if binary else LINEAR,
                    )) 
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss= BinaryCrossentropy() if binary else MSE,
                   metrics=[__tf_pearson_correlation])
    return model



def train_model(model, X_train, y_train, X_test,y_test, learning_steps):

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
    # Evaluate the model on the test set or perform any other required actions
        train_accuray= model.evaluate(X_test, y_test)
        cur_eval =  {f"fold_{fold}_pearson_corr" : train_accuray[1], 
                     f"fold_{fold}_MSE" : train_accuray[0]}
        evaluation_dict.update(cur_eval)
    return model , evaluation_dict

def train_final_model(model, X_train, y_train, X_test, y_test, learning_steps):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5, verbose=0, 
                                   restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=learning_steps, 
              batch_size=BATCH_SIZE,
              validation_split=0.2,  # Use a validation split instead of cross-validation
              callbacks=[early_stopping], verbose=0)
    
    # Evaluate the model on the test set
    test_accuracy = model.evaluate(X_test, y_test)
    
    evaluation_dict = {
        "test_pearson_corr": test_accuracy[1], 
        "test_MSE": test_accuracy[0]
    }
    
    return model, evaluation_dict


def eval_model(model, x,y)->Dict[str,float]:
        train_accuray= model.evaluate(x, y)
        return {"pearson_correlation" : train_accuray[1],
                "MSE" : train_accuray[0],}

def train_with_commet(parameter_dict, commetAPIKey, x_train, y_train, x_test, y_test):
    projectName = parameter_dict[EXP_ID].split('_')[0]
    experiment = Experiment(project_name=projectName, api_key= commetAPIKey,)
    experiment.log_parameters(parameter_dict)
    model = build_model(**parameter_dict)
 
    with experiment.train():
            model, train_eval_dict = train_model(model, x_train, y_train,
                                    x_test, y_test, parameter_dict[LEARNING_STEP])
            experiment.log_metrics(train_eval_dict)
    with experiment.validate():
        val_eval_dict = eval_model(model, x_train, y_train)
        experiment.log_metrics(val_eval_dict)
    with experiment.test():
        test_eval_dict = eval_model(model, x_test ,y_test)
        experiment.log_metrics(test_eval_dict)
    experiment.end()
    val_dict = {TRAIN_STR: train_eval_dict, VAL_STR: val_eval_dict, TEST_STR: test_eval_dict}
    return model, val_dict



def train_no_commets(parameter_dict, x_train, y_train, x_test, y_test):
    model = build_model(**parameter_dict)
    model, train_eval_dict = train_model(model, x_train, y_train,
                                        x_test, y_test, parameter_dict[LEARNING_STEP])
    val_eval_dict = eval_model(model, x_train, y_train)
    test_eval_dict = eval_model(model, x_test ,y_test)
    val_dict = {TRAIN_STR: train_eval_dict, VAL_STR: val_eval_dict, TEST_STR: test_eval_dict}
    return model, val_dict


def add_model_to_table(model_id, protein, species, experiment, experiment_details, cite,input_shape, source=None):
    """
    add a given model to the model table
    """
    model = DeepbindModel(protein, species, experiment, experiment_details, cite, model_id,input_shape, 'IB_generated')
    model.save_model_to_table()

import json
import os





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
    binary=False):
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
        EXP_ID: expirement_id
    }


def run_single_experiment(parameter_dict, commetAPIKey, x_train, y_train, x_test, y_test):
    if commetAPIKey:
        return train_with_commet(parameter_dict, commetAPIKey, x_train, y_train, x_test, y_test)
    else:
        return train_no_commets(parameter_dict, x_train, y_train, x_test, y_test)

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

# need to delete this and take the one from deepbing_global_args
SAVE_DIR = '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models/IB_models'



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
    

def save_final_df(df,save_path, model_id):
    if COMMET_API_KEY_ARG in df.columns:
        df.drop(columns=[COMMET_API_KEY_ARG])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(f'{save_path}/{model_id}_parameters.csv', index=False)
    print(f"Results saved to {save_path}/{model_id}_parameters.csv")


def get_n_save_final_df(results:list,config_df:pd.DataFrame, model_id:str, same_data:bool):
    output_dir = f'{SAVE_DIR}/{model_id}'
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
    output_dir = f'{SAVE_DIR}/{model_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_models(top_ten_items, results, output_dir)


def process_all_exps_results(results, df, exp_id):

    process_result(results, df, exp_id)
    # need to understand what todo with the results if they are trhe same maybe change the id beforew3sazx6
    model_data['model_id'] = exp_id
    # copy the data to the output folder
    copy_model_data(df.trainSet.values[0], exp_id)
    # add model to the model tabel
    add_model_to_table(**model_data)