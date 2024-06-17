import os
from Bio import SeqIO
import pandas as pd
from io import StringIO
import subprocess
from .run_general import  get_reversed_record,fasta_from_seq_record,save_prediction_df, ORIGINAL, ORIGINAL_WINDOW_SIZE, DEBUG, N_PROCESS
import concurrent
from predictionArchiver_pkg.predictionArchiver import PredictionSaver

TMP_SAVE_PATH = '/tmp/deepbind/'

DEEPBIND_PATH = '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models/original_db/deepbind'

# DEBUG = True
def save_model_list(model_df):
    """
    prepare a list of model ids and save it in a file for deepbind run
    """
  
    save_dir = os.path.join(TMP_SAVE_PATH, 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # print(f'created {save_dir}')

    save_path = os.path.join(save_dir, model_df.source.unique()[0] + '.ids')
    with open(save_path, 'w') as f:
        f.write('\n'.join(model_df['id'].values))
    return save_path



def parse_tsv_string(tsv_string):
    # Split the string into lines
    lines = tsv_string.strip().split('\n')
    # Split each line by tabs
    data = [line.split('\t') for line in lines]
    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

def process_original_results(output_result_path, fasta_file_path):
    # make the index from the fasta 
    seqs = SeqIO.parse(fasta_file_path, "fasta")
    seqs = [str(record.seq) for record in seqs]
    
    # print(f'length of seqs: {len(seqs)}')
    try:
        df = pd.read_csv(output_result_path, sep='\t')
    except pd.errors.EmptyDataError:
        raise Exception("No columns to parse from file ", output_result_path)
    # df = parse_tsv_string(stdout)
    df.insert(0, 'seq', seqs)
    df = df.set_index('seq')
    return df




def run_original_command(fasta_file_path, model_ids_path):
    output_file_path = f"/tmp/toozig/original_pred/{os.path.basename(fasta_file_path)}_{os.path.basename(model_ids_path)}"
    output_file_path = output_file_path.replace('(','').replace(')','').replace(':','_' ) + ".tsv"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    command = f"{DEEPBIND_PATH} '{model_ids_path}' '{fasta_file_path}' > '{output_file_path}'"
    # print('\n\n\n\n\n\n')
    # print(command)
    # print('\n\n\n\n\n\n')
    os.popen(command).read()

    return output_file_path

def get_original_model_prediction(fasta_file_path, model_ids_path):
    output_path = run_original_command(fasta_file_path, model_ids_path)
    df = process_original_results(output_path, fasta_file_path)
    return df


def run_original_on_sequence(record,window, shift, model_ids_path):
    saver = PredictionSaver()
    # here the archive put inside the P2 path
    result_path = saver.get_P2_saving_path(ORIGINAL, str(record.id), window, shift)
    if os.path.exists(result_path):
        return saver.json()
    fasta_file_path = fasta_from_seq_record(record, window, shift)
    result_df = get_original_model_prediction(fasta_file_path, model_ids_path)

    saver.save_prediction_df(result_df, result_path)
    return saver.json()



def main_original(fasta_file,model_df, shift):
    if len(model_df) == 0:
        return {'model_type':ORIGINAL,'path_list': []}
    window = ORIGINAL_WINDOW_SIZE
    saver = PredictionSaver()
    saver_df = saver.get_model_df()
    saver_df.loc[model_df.index, 'input_shape'] = ORIGINAL_WINDOW_SIZE
    all_results = []
    seq_records =SeqIO.parse(fasta_file, "fasta")
    model_ids_path = save_model_list(model_df)
    if DEBUG:
        record = next(seq_records)
        return run_original_on_sequence(record, window, shift, model_df)
    with concurrent.futures.ProcessPoolExecutor(N_PROCESS) as executor:
        results = [executor.submit(run_original_on_sequence, record, window, shift, model_ids_path) for record in seq_records] 
        # check if needed
        # seq_records =SeqIO.parse(fasta_file, "fasta")
        # results += [executor.submit(run_original_on_sequence, get_reversed_record(record) , window, shift, model_df) for record in seq_records]
        for f in concurrent.futures.as_completed(results):
            cur_result = f.result()
            saver = saver.merge_json(cur_result)
    print(f'len of all results before original: {len(all_results)}')
    print(f'len of saver p2 dict before{len(saver.get_dict("P2").files)}')
    for k,v in saver.get_dict('P2').items():
        all_results += [vi.path for vi in v]
    print(f'len of all results after original: {len(all_results)}')
    print(f'len of saver p2 dict after{len(saver.get_dict("P2").files )}')
    return {'model_type':ORIGINAL,'path_list' : all_results}
