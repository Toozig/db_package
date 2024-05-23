
import json
import os
from .run_RT import get_RT_prediction
from .run_general import get_model_df
import pandas as pd
import plotly.express as px
import random
import string
from Bio import  Entrez, SeqIO
# from .run_RT import get_RT_predictions

P3_PATH = '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/DB_predictions/merged_mATAC_hATAC_0507/P3/'
ZSCORE_PATH = '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/DB_z_score_calc/zscore_dict/merged_mATAC_hATAC_0507_zscore1.json'
ZSCORE_DICT = json.load(open(ZSCORE_PATH))
ALL_VARS_PATH = '/dsi/gonen-lab/shared_files/family_viewer_data/all_vars.csv'
DSD_TF_LIST =  "SRY, SOX9, SOX8, SOX10, DMRT1, GATA4, SF1, NR5A1, WT1, FOXL2, RUNX1, LHX9, EMX2, TCF3, TCF12, LEF1, ESR1, ESR2, AR".replace(' ','').upper().split(',')
DSD_TF_LIST = [i.strip() for i in DSD_TF_LIST]
MODEL_DF_PATH = '/home/dsi/toozig/gonen-lab/users/toozig/projects/deepBind_pipeline/deepBind_run/models/model_table.tsv'

#### table cols ARGS
SEGMENT_ID_COL = 14
LENGTH_COL = 18
TAD_COL =11


CHROM = 'CHROM'
START = 'from'
END = 'to'

def get_protein_df(protein_list):
    model_df = pd.read_csv(MODEL_DF_PATH, sep='\t', index_col=0)
    return model_df[model_df.protein.isin(protein_list)]

def get_protein_dict():
    model_df = pd.read_csv(MODEL_DF_PATH, sep='\t', index_col=0)
    protein_dict = {'DSD TF': {i:i for i in  model_df.protein if i.upper() in DSD_TF_LIST},
                    'Non DSD TF': {i:i for i in  model_df.protein if i.upper() not in DSD_TF_LIST}}
    return protein_dict

def calc_zscore(row):
    model_dict = ZSCORE_DICT[row.name.split('_w')[0]]
    return (row - model_dict['mean']) / model_dict['sd']


def get_model_name(models_id):
    model_df = pd.read_csv(MODEL_DF_PATH, sep='\t')
    model_name  = [i.split('_w')[0] for i in models_id]
    protein = [model_df.loc[model_df['id'] == i, 'protein'].values[0] for i in model_name]
    experiment = [model_df.loc[model_df['id'] == i, 'experiment'].values[0][:2] for i in model_name]
    species = [model_df.loc[model_df['id'] == i, 'species'].values[0][:1] for i in model_name] 
    return [protein[i].upper() + '.' + species[i] + '.' + experiment[i] for i in range(len(model_name))]


def get_plot(score_df, title):
    fig = px.imshow(score_df.T, color_continuous_scale='RdBu_r', origin='lower',
                     labels=dict(x="location", y="TF", color="Z-score")
                     )
    fig.update_layout(title=title)
    return fig



def generate_random_email():
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
    letters = string.ascii_lowercase
    email = ''.join(random.choice(letters) for i in range(10)) + '@' + random.choice(domains)
    return email


def get_seq(chrom,start, end):
    Entrez.email = generate_random_email()
    chrom_dict = {'X':'23', 'Y':'24'}
    cur_chrom = chrom.replace('chr','').upper()
    cur_chrom = chrom_dict[cur_chrom] if cur_chrom in chrom_dict else  cur_chrom
    chrom_id  = "NC_000001"[:-len(cur_chrom)] + cur_chrom
    # return
    handle = Entrez.efetch(db="nucleotide",
                           id=chrom_id,
                           rettype="fasta",
                           strand=1,
                           seq_start=start,
                           seq_stop=end)
    record = SeqIO.read(handle, "fasta")
    handle.close()
    return str(record.seq)




def __sum_names(names_col):

    names = set()
    for i in names_col:
        if type(i) != str:
            continue
        names = names |  set(i.split(';'))
    return len(names)

def get_peak_data():
    """
    gets the main data for the peaks
    """
    all_vars_df = pd.read_csv(ALL_VARS_PATH)
    cols = all_vars_df.columns

    columns = ['INTERVAL_ID', 'CHROM']
    columns += cols[SEGMENT_ID_COL:LENGTH_COL].tolist() + cols[[TAD_COL]].tolist() + ['distance_from_nearest_DSD_TSS']
    columns = [i for i in columns if i in cols and i != 'median_DP']
    segment_df = all_vars_df[columns].copy().drop_duplicates()
    groupd = all_vars_df.groupby('INTERVAL_ID')
    n_probands = groupd['probands_names'].apply(__sum_names)
    n_healthy = groupd['healthy_names'].apply(__sum_names)
    segment_df['n_probands'] = segment_df['INTERVAL_ID'].map(n_probands)
    segment_df['n_healthy'] = segment_df['INTERVAL_ID'].map(n_healthy)
    return segment_df[~segment_df[[CHROM, START,END]].duplicated()]



def get_score_df(chrom,start,end, protein_list, shift, to_norm, seq_id=''):
    peak_seq = get_seq(chrom,start,end)
    score_df = get_RT_prediction(protein_list, peak_seq, shift, seq_id)
    if to_norm:
        score_df = score_df.apply(calc_zscore, axis=0)
    score_df.columns = get_model_name(score_df.columns)
    score_df = score_df.sort_index(axis=1)
    return score_df


def get_app_plot(chrom,start,end,to_norm, protein_list, shift,seq_id):
    # print(f'peak_row: {peak_row}, to_norm: {to_norm}, protein_list: {protein_list}, shift: {shift}')
    if seq_id == 'manual_peak':
        seq_id = ''
    score_df = get_score_df(chrom,start,end, protein_list, shift, to_norm, seq_id)
    score_df.index = score_df.index.astype(int) + start
    fig = get_plot(score_df)
    return  fig

def get_app_model_df(protein_list):
    model_df, _ = get_model_df(protein_list)
    return model_df