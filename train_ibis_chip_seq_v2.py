
# from db_train_utils.train_chip_seq.chip_seq_main import run_chip_seq
from db_train_utils.train_chip_seq.chip_seq_utils import get_db_chs_object
from db_train_utils.train_chip_seq.ibis_chip_seq_main import run_ibis_chip_seq
import ibis_utils.ibis_utils as iu
from Bio import SeqIO

TMP_DIR = '/tmp/toozig/'


def get_input_shape(fasta_path):
    # read one sequence to get the shape
    fasta_sequences = SeqIO.parse(open(fasta_path), 'fasta')
    seq = next(fasta_sequences)
    return len(seq.seq), 4



def chip_seq_IBIS(
    chip_seq_path,
        protein,
        species,
        experiment, 
        lab, cite,
         n_exp,
         commetAPIKey='',
         version='v2'):
    name_col = 'name'
    center_col = 'abs_summit'
    centered_bed = iu.get_centered_bed(chip_seq_path, TMP_DIR,center_col , name_col)
    positive_data = iu.get_fasta_from_bed(centered_bed, TMP_DIR, genome='hg38')
    negative_data = iu.create_negative_from_fasta(positive_data)
    print(f'bed path: {centered_bed}\npositive data: {positive_data}\nnegative data: {negative_data}')
   
    input_shape = get_input_shape(positive_data)
    print(f'bed path: {centered_bed}\npositive data: {positive_data}\nnegative data: {negative_data}')
    db_chs_obj = get_db_chs_object(protein, species, experiment, lab, cite, input_shape)
    # print('running chip seq')
    run_ibis_chip_seq(positive_data, negative_data, n_exp, commetAPIKey, db_chs_obj, version)


import sys
if __name__ == '__main__':
    chip_seq_path = sys.argv[1]
    version = sys.argv[2]
    prefix = sys.argv[3]
    protein = chip_seq_path.split('/')[-2]
    cite = chip_seq_path.split('/')[-1].split('.pe')[0]

    species = 'Homo Sapiens'
    experiment ='chip seq'
    lab = 'ibis_2024'
    n_exp = 10
    chip_seq_IBIS(chip_seq_path,protein=protein,species=species, experiment=experiment,
                  lab=lab,cite=cite + f'_{version}_{prefix}',n_exp=n_exp)


    
    # path = sys.argv[1]
    # n_exp = 10
    # commet_api = ''
    # print(f'starting {path}')
    # run_main(path)
    # print(f'finished {path}')
