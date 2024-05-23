import os
import pandas as pd
import subprocess
from ...db_train_utils.db_train_utils import oneHot_encode
from Bio import SeqIO
import numpy as np
from random import shuffle

TMP_DIR = '/tmp/toozig/'
BG_BED = 'background_bed'
SHUFFLE = 'Shuffle'

def negative_bed(positive_bed, background_bed, padding, n_samples,genome='hg38', **kwargs):
    # Define the output file name
    name = positive_bed.split('.')[0]
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


def shuffle_string(s):
    s_list = list(s)
    shuffle(s_list)
    return ''.join(s_list)


def negative_shuffle(path_to_fasta, **kwargs):
    """
    Creates a negative (shuffled sequences) version of a FASTA file.

    Args:
        path_to_fasta (str): Path to the input FASTA file.

    Returns:
        str: Path to the output FASTA file with shuffled sequences.

    """
    output_path = path_to_fasta.replace(path_to_fasta.split('.')[-1], '_negative.' + path_to_fasta.split('.')[-1])
    records = SeqIO.parse(path_to_fasta, 'fasta')
    with open(output_path, 'w') as output:
        for record in records:
            output.write(f'>{record.id}_negative\n')
            output.write(f'{shuffle_string(str(record.seq))}\n')
    return output_path  


NEGATIVE_TYPE_DICT = {SHUFFLE: negative_shuffle,
                      BG_BED: negative_bed}



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




def get_train_test_data(positive_data, negative_data):
    x_pos, y_pos = get_chip_processed_data(positive_data, True)
    x_neg, y_neg = get_chip_processed_data(negative_data, False)
    X = np.concatenate((x_pos, x_neg))
    y =  np.concatenate((y_pos, y_neg))
    return X, y

def get_chs_exp_details(chip_seq_path,padding, negative_type, positive_bed, negative_bed, positive_fasta, negative_fasta,background_bed, bg_cite):
    exp_details = {'data' : chip_seq_path,
                   'padding': padding,
                   'negative_type': negative_type,
                    'positive_bed': positive_bed, 
                    'negative_bed': negative_bed, 
                    'positive_fasta': positive_fasta, 
                    'negative_fasta': negative_fasta
                    }
    if background_bed is not None:
        exp_details['background_bed'] = background_bed  
        exp_details['bg_cite'] = bg_cite
    return exp_details

