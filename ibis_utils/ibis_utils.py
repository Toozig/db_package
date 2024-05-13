import os
import pandas as pd
from ..global_db_args import MODEL_TABLE
import subprocess
from Bio import SeqIO
from random import shuffle
import os
import pandas as pd
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

SUBMISSION_DIR = os.path.join(script_dir, 'Leaderboard_aaa_submissions')
CATEGORY_FILE = os.path.join(SUBMISSION_DIR, "%s_aaa_random.tsv")

HG38_ZIP_PATH =  '/dsi/gonen-lab/shared_files/WGS_on_DSD/data/refrence_genome/hg38/hg38.fa.gz'
MM10_ZIP_PATH =  '/dsi/gonen-lab/shared_files/WGS_on_DSD/data/refrence_genome/mm10/GCA_000001635.8_GRCm38.p6_genomic.fna.gz'
HG38_PATH = '/tmp/toozig/hg38.fa'
MM10_PATH = '/dsi/gonen-lab/shared_files/WGS_on_DSD/data/refrence_genome/mm10/mm10.fa'

GENOME_DICT = {'hg38': HG38_PATH,
               'mm10': MM10_PATH}
ZIPPED_GENOME_DICT = {'hg38': HG38_ZIP_PATH,
                      'mm10': MM10_ZIP_PATH}




def bed_to_fasta(fatsa_path, bed_path, output_path, name_flag='-name', **kwargs):
    """
    Converts a BED file to a FASTA file using bedtools.

    Args:
        fatsa_path (str): Path to the input FASTA file.
        bed_path (str): Path to the input BED file.
        output_path (str): Path where the output FASTA file will be saved.
        name_flag (str, optional): Flag for bedtools. Defaults to '-name'.
        **kwargs: Additional flags for bedtools.
    """
    flags = list(kwargs.keys())
    # Define the command as a list of arguments
    command = ["bedtools", "getfasta", "-name", "-fi", fatsa_path, "-bed", bed_path, "-fo", output_path] + flags
    print('\n\n\n\n-------------------------------')
    print('running command:', ' '.join(command))
    print('\n\n\n\n-------------------------------')
    subprocess.run(command, check=True)
    return output_path



def __open_peak_file(bed_path,):
    header = pd.read_csv(bed_path, sep='\t', nrows=1)
    header = header.columns.str.strip()
    bed_df = pd.read_csv(bed_path, sep='\t', comment='#')
    bed_df.columns  = header
    return bed_df

def __save_bed_file(bed_df, bed_path, output_dir, start, end, name_col,padding, prefix=''):
    # use the nname col as id else generate by index and bed file name
    alt_name = bed_path.split('/')[-1].split('.')[0] + '_' + bed_df.index.astype(str)
    names = alt_name if not len(name_col) > 0 else bed_df[name_col]
    new_bed_df = {'chr': bed_df.iloc[:,0], 'start': start, 'end': end, 'name': names}
    new_bed_df = pd.DataFrame(new_bed_df)
    output_name = bed_path.split('/')[-1].split('.')[0] + prefix +f'_{padding}bp.bed'
    output_path = os.path.join(output_dir, output_name)
    new_bed_df.to_csv(output_path, sep='\t', index=False, header=False)
    # filter out na and empty names
    new_bed_df = new_bed_df.dropna()
    return output_path

def get_bed(peak_file, output_dir,padding,name_col=''):
    peak_df = __open_peak_file(peak_file)
    start = peak_df['START']
    end = peak_df['END']
    return __save_bed_file(peak_df, peak_file, output_dir, start, end, name_col, padding,'')

def get_centered_bed(bed_path,output_dir,center_col = '',name_col='', padding=100):
    """
    create a bed file the center of the peak as middle of the peak
    the length of the peak is 2*padding
    if center_col is not empty the center will be (END - START / 2)
    """
    bed_df = __open_peak_file(bed_path)
    center = bed_df[center_col] if len(center_col) > 0 else (bed_df['start'] + bed_df['end']) // 2
    start = center.astype(int) - padding
    end = center + padding
    return __save_bed_file(bed_df, bed_path, output_dir, start, end, name_col,padding, '_centered')



def get_fasta_from_bed(bed_path, output_dir, padding,genome='hg38'):
    """
    return a path to fasta file from a bed file, got HG38 genome
    """

    padding =str(padding)  +'bp_'  
    output_name = padding + bed_path.split('/')[-1].replace('.bed', '.fa')
    output_path = os.path.join(output_dir, output_name)
    genome_fasta = GENOME_DICT.get(genome, None)

    if genome_fasta is None:
        raise ValueError(f'Genome {genome} not supported')

    bed_to_fasta(genome_fasta, bed_path, output_path)
    return output_path


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

def get_exp_proteins(exp_name):
    category_file = CATEGORY_FILE % exp_name
    df = pd.read_csv(category_file, sep='\t')
    return df.columns.tolist()


def get_model_df(exp_name):
    model_table = pd.read_csv(MODEL_TABLE, sep='\t')
    relevant_proteins = get_exp_proteins(exp_name)
    model_table = model_table.loc[model_table['protein'].isin(relevant_proteins)]
    model_table = model_table.loc[model_table.experiment_details.str.contains('ibis')]
    return model_table