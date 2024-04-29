from ..train_global_args import *
from ..train_global_function import run_single_experiment, get_parms_dict, add_model_to_table, copy_model_data, SAVE_DIR ,  process_result, get_n_save_final_df
from .chip_seq_utils import   prepare_chs_train_test, process_init_result, prepare_chs_df
import concurrent.futures


from multiprocessing import Pool

from tqdm import tqdm
DEBUG = False


POSITIVE_DATA = TRAIN_SET
NEGATIVE_DATA = TEST_SET

TMP_DIR = '/tmp/toozig/'


def run_chip_seq_single_expirement(row, to_split):
    print('running expirement', row[EXP_ID])
    # print(row)

    x_train, y_train, x_test, y_test  = prepare_chs_train_test(row[POSITIVE_DATA], row[NEGATIVE_DATA], to_split)
    print('prepared data')
    parameter_dict = get_parms_dict((len(x_train[0] ),4), row[POSITIVE_DATA],
                                        row[NEGATIVE_DATA], row[LEARNING_RATE],
                                        row[N_MOTIF], row[LENGTH_MOTIF],
                                        row[DROPOUT_RATE], row[HIDDEN_LAYER],
                                        row[LEARNING_STEP], row[EXP_ID], row[BINARY])
    print('got parms')
    model, train_eval_dict = run_single_experiment(parameter_dict, row[COMMET_API_KEY_ARG], x_train, y_train, x_test, y_test)
    print('finished running expirement')
    if DEBUG:
        print(f"finished expirement {row.expirement_id}")
    if to_split:
        model = None
    return {EXP_ID: row[EXP_ID],'model': model, 'score_dict' :train_eval_dict}




def run_chip_seq(positive_fasta, negative_fasta, n_exp, commetAPIKey, db_chs_obj):
    configuration_df = prepare_chs_df(n_exp,db_chs_obj,commetAPIKey, positive_fasta, negative_fasta)
    global_model_data = db_chs_obj.get_db_data()
    exp_id = global_model_data['model_id']
    print(f"Starting expirement {exp_id}")
    init_results = []
    results = []
    rows = [row for _, row in configuration_df.iterrows()]

    # for row in rows:
    #     init_results.append(run_chip_seq_single_expirement(row, True))
    # exit()
    with concurrent.futures.ProcessPoolExecutor(60) as executor:
        # Submit each configuration to the executor for parallel execution
        init_results = executor.map(run_chip_seq_single_expirement, rows, [True] * len(rows))
        # get the top 01 config with best results
        top_ten = process_init_result(configuration_df,init_results,exp_id)
        rows = top_ten.to_dict(orient='records')    

    with concurrent.futures.ProcessPoolExecutor(20) as executor:
        results = list(executor.map(run_chip_seq_single_expirement, rows, [False] * len(rows)))
    print('finished running final test')
    df = get_n_save_final_df(results,top_ten,exp_id, False)
    print('processed results')
    process_result(results, df, exp_id)
    global_model_data['model_id'] = exp_id
    # copy the data to the output folder
    copy_model_data(df[POSITIVE_DATA].values[0], exp_id)
    # add model to the model tabel
    db_chs_obj.save_model_to_table()
    print(f"Finished expirement {exp_id}")

