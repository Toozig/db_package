from ..train_global_args import *
from ..train_global_function import *
from . import chip_seq_utils as chip_utils 
from ..  import train_global_function as glob_func
from ..db_train_utils import get_input_shape_from_fasta
from ...ibis_utils import ibis_utils as iu
import subprocess
import os
from sklearn.model_selection import train_test_split

TMP_DIR = '/tmp/toozig/'
BG_BED = 'background_bed'
SHUFFLE = 'Shuffle'

def negative_bed(positive_bed, background_bed, padding, n_samples,genome='hg38', **kwargs):
    # Define the output file name
    name = os.path.basename(positive_bed).split('.')[0]
    negative_bed_name = f'{name}_negative.bed'

    # Construct the BEDTools command
    cmd = f'bedtools subtract -a {background_bed} -b {positive_bed} > {negative_bed_name}'
    print(f"\n\n\n\n\n=======================\ncmd : {cmd}\n\n\n\n\n")
    # Run the command
    subprocess.run(cmd, shell=True, check=True)
    bed_df = pd.read_csv(negative_bed_name, sep='\t', header=None)
    length = bed_df[2] - bed_df[1]
    bed_df = bed_df[length >= 2*padding]
    centers = (bed_df[1] + bed_df[2]) // 2
    bed_df[1] = centers - padding
    bed_df[2] = centers + padding
    n_samples = min(n_samples, len(bed_df))
    # choose n_samples random samples
    negative_bed = bed_df.sample(n_samples)
    name = os.path.basename(positive_bed).split('.')[0] 
    # use os to create the path
    save_path = os.path.join(TMP_DIR, f'{name}_negative.bed')
    negative_bed[3] = negative_bed[3] + '_negative'
    negative_bed.to_csv(save_path, sep='\t', index=False, header=False)
   
    return save_path

    
NEGATIVE_TYPE_DICT = {SHUFFLE: iu.negative_shuffle,
                      BG_BED: negative_bed}



def run_chipseq_db(df, X, y, obj_model, version,final=False, split= 0.15):
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    exp_id, df, results, obj_model = glob_func.run_deepBind_expiriment(df, X_train,y_train, X_test ,y_test, obj_model, version, final)
    get_n_save_final_df(results ,df, obj_model.get_id(), True)
    return exp_id, df, results, obj_model


def run_final_chip_seq(configuration_df,init_results,exp_id,version, X, y, obj_model):
    top_ten = chip_utils.process_init_result(configuration_df,init_results,exp_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.03)
    results = []
    for i, row in top_ten.iterrows():
        cur_result = glob_func.run_single_deepBind_expirement(row, X_train, y_train, X_test, y_test, version=version, final=True)
        results.append(cur_result)

    # exp_id, df, results, obj_model = glob_func.run_deepBind_expiriment(top_ten, X_train,y_train, X_test ,y_test, obj_model, version, final=True)
    print('finished running final test')
    df = glob_func.get_n_save_final_df(results,top_ten,exp_id, False)
    print('processed results')
    glob_func.process_result(results, df, exp_id)

    # copy the data to the output folder
    glob_func.copy_model_data(df[TRAIN_SET].values[0], exp_id)
    # add model to the model tabel
    obj_model.save_model_to_table()
    copy_model_data(df[TRAIN_SET].values[0], exp_id)
    return df


def prepare_chipseq(positive_fasta,
                    negative_fasta,
                    protein,
                    species,
                    lab,
                    cite,
                    n_exp,
                    id_prefix ='',
                    version='v2'):
    input_shape = get_input_shape_from_fasta(positive_fasta)
    chs_obj = chip_utils.get_db_chs_object(protein, species, lab, cite, input_shape)
    # the '' is the commetAPI
    configuration_df = chip_utils.prepare_chs_ibis_df(n_exp,chs_obj,'', positive_fasta, negative_fasta, version,id_prefix)
    X ,y = chip_utils.get_train_test_data(positive_fasta, negative_fasta, False)
    return configuration_df, X, y, chs_obj
   

