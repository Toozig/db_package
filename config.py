# db_package/config.py

import os
import sys

# Path to the db_package directory
PACKAGE_ROOT = "/localdata/idob/deepBind_pipeline/db_package/"

# Add the PACKAGE_ROOT to the Python path
if PACKAGE_ROOT not in sys.path:
    sys.path.append(PACKAGE_ROOT)

# Path to the files
MODEL_TABLE_PATH = os.path.join(PACKAGE_ROOT, 'deepBind_run/models/model_table.tsv')
ZSCORE_JSON_PATH = os.path.join(PACKAGE_ROOT, 'DB_z_score_calc/zscore_dict/merged_mATAC_hATAC_0507_zscore1.json')
MAX_PRED_DIR = os.path.join(PACKAGE_ROOT, 'run_db/cached_max_predictions')
MODEL_PATH = os.path.join(PACKAGE_ROOT, 'deepBind_run/models/IB_models')
DATA_DIR = os.path.join(PACKAGE_ROOT, 'process_data/data')
TMP_DIR ='/tmp/toozig/'
# other configs:
DSD_TF_LIST =  "SRY, SOX9, SOX8, SOX10, DMRT1, GATA4, SF1, NR5A1, WT1, FOXL2, RUNX1, LHX9, EMX2, TCF3, TCF12, LEF1, ESR1, ESR2, AR".replace(' ','').upper().split(',')
DSD_TF_LIST = [i.strip() for i in DSD_TF_LIST]
HG38_FASTA = os.path.join(PACKAGE_ROOT, '/home/dsi/toozig/yaron_lab_dir/deepBind_pipeline/streamlit_app/hg38.fa')
MM10_FASTA = os.path.join(PACKAGE_ROOT, '/home/dsi/toozig/yaron_lab_dir/deepBind_pipeline/streamlit_app/mm10.fa')

def get_hg38_fasta():
    return HG38_FASTA

def get_mm10_fasta():
    return MM10_FASTA


def get_data_dir():
    return DATA_DIR

def get_tmp_dir():
    return TMP_DIR

def get_model_table_path():
    return MODEL_TABLE_PATH

def get_zscore_json_path():
    return ZSCORE_JSON_PATH

def get_dsd_tf_list():
    return DSD_TF_LIST

def get_max_pred_dir():
    return MAX_PRED_DIR

def get_IB_model_path():
    return MODEL_PATH