from db_train_utils.train_chip_seq.chip_seq_main import run_chip_seq
from db_train_utils.train_chip_seq.chip_seq_utils import get_db_chs_object
import ibis_utils as iu
from Bio import SeqIO
import sys
POS = 1
NEG = 2
TMP_DIR = '/tmp/toozig/'


def get_input_shape(fasta_path):
    # read one sequence to get the shape
    fasta_sequences = SeqIO.parse(open(fasta_path), 'fasta')
    seq = next(fasta_sequences)
    return len(seq.seq), 4


def peak_to_fasta(peak_path,tmp_dir,center_col,name_col):
    centered_bed = iu.get_centered_bed(peak_path, tmp_dir,center_col , name_col)
    fasta_path = iu.get_fasta_from_bed(centered_bed, tmp_dir, genome='hg38')
    return fasta_path

def GHST_IBIS(
    pos_path,
    neg_path,
        protein,
        species,
        experiment, 
        lab, cite,
         n_exp,
         commetAPIKey=''):
    name_col = 'name'
    center_col = 'abs_summit'
    positive_data = peak_to_fasta(pos_path,TMP_DIR,center_col,name_col)
    negative_data = peak_to_fasta(neg_path,TMP_DIR,center_col,name_col)
    input_shape = get_input_shape(positive_data)
    db_chs_obj = get_db_chs_object(protein, species, experiment, lab, cite, cite, input_shape)
    # print('running chip seq')
    run_chip_seq(positive_data, negative_data, n_exp, commetAPIKey, db_chs_obj)



if __name__ == '__main__':
    GHST_pos_path = sys.argv[POS]
    GHST_neg_path = sys.argv[NEG]
    protein = GHST_pos_path.split('/')[-2]
    cite = GHST_pos_path.split('/')[-1].split('.pe')[0]
    species = 'Homo Sapiens'
    experiment ='GHTS'
    lab = 'ibis_2024'
    n_exp = 10
    GHST_IBIS(GHST_pos_path,GHST_neg_path,protein=protein,species=species, experiment=experiment,
                  lab=lab,cite=cite,n_exp=n_exp)


    
