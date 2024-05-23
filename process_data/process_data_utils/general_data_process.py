import os
import pandas as pd
import subprocess
from Bio import SeqIO
import json
from random import shuffle
import os
import pandas as pd
# Get the directory of the current script
import os
import shutil
import numpy as np


script_dir = os.path.dirname(os.path.realpath(__file__))

SUBMISSION_DIR = os.path.join(script_dir, 'Leaderboard_aaa_submissions')
CATEGORY_FILE = os.path.join(SUBMISSION_DIR, "%s_aaa_random.tsv")
DATA_DIR = '/home/dsi/toozig/gonen-lab/users/toozig/projects/deepBind_pipeline/streamlit_app/pages/db_package/process_data/data'
DATA_DB = f'{DATA_DIR}/data_db.json'
HG38_ZIP_PATH =  '/dsi/gonen-lab/shared_files/WGS_on_DSD/data/refrence_genome/hg38/hg38.fa.gz'
MM10_ZIP_PATH =  '/dsi/gonen-lab/shared_files/WGS_on_DSD/data/refrence_genome/mm10/GCA_000001635.8_GRCm38.p6_genomic.fna.gz'
HG38_PATH = '/tmp/toozig/hg38.fa'
MM10_PATH = '/dsi/gonen-lab/shared_files/WGS_on_DSD/data/refrence_genome/mm10/mm10.fa'


GENOME_DICT = {'hg38': HG38_PATH,
               'mm10': MM10_PATH}
ZIPPED_GENOME_DICT = {'hg38': HG38_ZIP_PATH,
                      'mm10': MM10_ZIP_PATH}

SPECIES_LIST = ['Homo sapiens', 'Mus musculus']
SPECIES_TO_GENOME = {'Homo sapiens': 'hg38',
               'Mus musculus': 'mm10'}


def species_to_genome(species):
    return SPECIES_TO_GENOME[species]


def copy_file_to_folder(file_path, folder_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    # Create the directory if it doesn't exist
    dir_path = os.path.join(folder_path)
    os.makedirs(dir_path, exist_ok=True)

    # Copy the file
    destination_path = shutil.copy(file_path, dir_path)

    return destination_path

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


def open_file(path, header,sep='\t' ,n_rows=None):

    df = pd.read_csv(path, sep=sep, header=header, nrows=n_rows)
    df.columns = df.columns.astype(str)
    return df

def make_reademe(X,y, data_path, data_id, project_dir):
    """
    create a readme file for the data
    """
    readme_path = f'{project_dir}/{data_id}_readme.txt'
    with open(readme_path, 'w') as file:
        file.write(f'Data id: {data_id}\n')
        file.write(f'X shape: {X.shape}\n')
        file.write(f'y shape: {y.shape}\n')
        file.write(f'original Data path: {data_path}\n')
        file.write(f'X path: {project_dir}/{data_id}_X.npy\n')
        file.write(f'y path: {project_dir}/{data_id}_y.npy\n')
    return readme_path
 
def save_data(X, y, project_dir, data_id, data_path):
    # copy the data to the project dir
    data_new_path = copy_file_to_folder(data_path, project_dir)
    x_path = f'{project_dir}/{data_id}_X.npy'
    y_path = f'{project_dir}/{data_id}_y.npy'
    np.save(x_path, X)
    np.save(y_path, y)
    make_reademe(X,y, data_new_path, data_id, project_dir)
    return x_path, y_path, data_new_path

def __open_peak_file(bed_path):
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
    print(output_name)
    print(output_dir)
    output_path = os.path.join(output_dir, output_name)
    print(output_path)
    # check if the directory exists, create if not

    os.makedirs(output_dir, exist_ok=True)
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
    output_path = output_path.replace(' ', '_')
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



def generate_id(exp_type, protein, species,cite, more=''):
    """
    generate a unique id for the experiment
    """
    data_id =  f'{exp_type}_{protein}_{species}_{cite}'
    data_id += f'_{more}' if len(more) > 0 else ''
    num = len([True for i in os.listdir(DATA_DIR) if data_id in i])
    return f'{data_id}_{num}'



def get_summary(data_id, protein, species,n_samples, X_path, y_path, cite, exp_data, exp_type, comments=''):
    summary = {'id': data_id, 
            'protein': protein, 
            'species': species, 
            'n_samples': n_samples,
            'X_path': X_path,
            'y_path': y_path,
            'cite': cite, 
            'exp_type': exp_type, 
            'experiment_details': exp_data,
            'comments': comments}
    return summary


def save_exp_summary(exp_id,summary, project_dir):
    """
    save the summary of the experiment to a file
    """
    summary['added_by'] = os.getlogin()
    summary['date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    # load the main data base
    data_db = json.load(open(DATA_DB, 'r'))
    data_db[exp_id] = summary
    with open(DATA_DB, 'w') as file:
        # save the json such that each line is an entry
        json.dump(data_db, file, indent=4)
        # json.dump(data_db, file)
    # project_dir = f'{DATA_DIR}/{exp_id}'
    # os.makedirs(project_dir, exist_ok=True)
    # same the summary json as a readme file
    with open(f'{project_dir}/README.txt', 'w') as file:
        file.write(json.dumps(summary, indent=4))
    return summary

def get_data_dir():
    return DATA_DIR