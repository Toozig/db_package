#!/usr/bin/env python
from .train_global_args import *
import numpy as np
from itertools import product

import argparse
import pandas as pd


DROP_OUT_PARAMS = [0.0, 0.25, 0.5]
LEARNING_STEPS_PARAMS = [4,8,16]
NUMBER_OF_MOTIF_PARAMS = [16,64,124,256]
MOTIF_LENGTH_PARAMS=[8,16]
HIDDEN_LAYER_PARAMS = [True, False]
LEARNING_RATE_PARAMS = (0.005,0.05)
FCN_OUTPUT = [16,34,64]
HK = 'HK'
ME = 'ME'
DATA_SETS = [HK, ME]

##### variables sampling functions ####

def __log_uniform_sampler(a, b, size=None):

    # Sample x from a uniform distribution in [0, 1)
    x = np.random.uniform(0, 1, size)
    # Calculate log-uniformly distributed values in [a, b)
    result = 10**((np.log10(b) - np.log10(a)) * x + np.log10(a))
    return result


def __get_dropout_val(): return np.random.choice(DROP_OUT_PARAMS)


def generate_model_variables(shape,n_filters, motif_length=16):
    """
    Generates a dictionary of model variables for the deepBind model.
    all parameters are sampled from the paper's hyperparameter space
    """
    model_variables = {
        'filters': n_filters,
        'kernel_size': motif_length,
        'input_shape': shape,
        'dropout_rate': __get_dropout_val(),
        'learing_rate': __log_uniform_sampler(LEARNING_RATE_PARAMS[0], 
                                              LEARNING_RATE_PARAMS[1]),
    }
    
    return model_variables

def get_random_configuration():
    conf_dict = {DROPOUT_RATE: __get_dropout_val(),
                    LEARNING_RATE: __log_uniform_sampler(LEARNING_RATE_PARAMS[0], 
                                                        LEARNING_RATE_PARAMS[-1]),
                    N_MOTIF: np.random.choice(NUMBER_OF_MOTIF_PARAMS)}
    return conf_dict



def generate_variables_configurations(n:int):
    """
    input:
        n - number of configurations to generate
        project_name - the name of the project

        NOTE: This functio does not include train and test. this need to be added later depends on data type

    """

    # Generate n random configurations
    random_conf = [get_random_configuration() for _ in range(n)]

    # Generate all possible hyperparameter combinations
    parameter_combinations = product(LEARNING_STEPS_PARAMS, MOTIF_LENGTH_PARAMS, HIDDEN_LAYER_PARAMS)

    # Create configurations using list comprehensions
    configurations = [
        {
            LEARNING_STEP: learning_step,
            LENGTH_MOTIF: motif_length,
            HIDDEN_LAYER: hidden_layer,
        }
        for learning_step, motif_length, hidden_layer in parameter_combinations
        ]
    # Update random configurations with generated configurations
    random_conf_with_parameters = [dict(conf, **config) for conf in random_conf for config in configurations]
    return pd.DataFrame(random_conf_with_parameters)



def parse_args():
    parser = argparse.ArgumentParser(description='Your script description.')
    
    parser.add_argument('project_name', type=str ,help='The project name (use the TF name)')
    parser.add_argument('n_exp',type=int, help='Number of expirements')
    parser.add_argument('--ouputDir', type=str,help='Output directory')

    return parser.parse_args()


def genretate_fcn_configurations(n_exp):
    configuration_df = pd.DataFrame(generate_variables_configurations(n_exp))
    # add random LSTM configurations
    configuration_df[LSTM_OUTPUT] = [np.random.choice(FCN_OUTPUT) for _ in range(len(configuration_df))]
    configuration_df[DENSE_OUTPUT] = [np.random.choice(FCN_OUTPUT) for _ in range(len(configuration_df))]
    configuration_df[DENSE2_OUTPUT] = [np.random.choice(FCN_OUTPUT) for _ in range(len(configuration_df))]
    return configuration_df

def main():
    args= parse_args()
    

    print("generating parameters")
    configuration_df = generate_variables_configurations(args.n_exp, args.project_name)
    # save dfs
    df = pd.DataFrame(configuration_df)
    df = df.set_index('expirementID')
    df.to_csv(f"{args.ouputDir}/{args.project_name}_configurations.tsv", sep='\t')
    print(f"Configurations saved to {args.ouputDir}/{args.project_name}_configurations.tsv")
if __name__ == "__main__":
    main()

