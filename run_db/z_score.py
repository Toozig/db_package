import json
import os

import sys
if 'config' not in  sys.modules:
    sys.path.append(os.path.dirname(__file__))
    from  .. import  config 
else:
    import config 


ZSCORE_JSON = config.get_zscore_json_path()
from .IB_function import get_IB_model_prediction2

def calc_IB_zscore_params(model_id, X):
    """
    calculate the data prediction mean and std
    """
    print('X shape', X.shape)
    prediction = get_IB_model_prediction2(model_id, X)
    print("result shape- ",prediction.shape)
    mean = float(prediction.mean())
    std = float(prediction.std())
    with open(ZSCORE_JSON, 'r') as f:
        zscore_dict = json.load(f)
    zscore_dict[model_id] = {'mean': mean, 'std': std, 'n_samples': X.shape[0]}
    tmpfile = ZSCORE_JSON + '.tmp'
    with open(tmpfile, 'w') as f:
        json.dump(zscore_dict, f, indent=4)
    #remove old file and rename the tmp
    os.remove(ZSCORE_JSON)
    os.rename(tmpfile, ZSCORE_JSON)
    return {model_id: {'mean': mean, 'std': std}}


def calc_zscore(model_id, y_pred):
    """
    calculate the zscore of the prediction
    """
    with open(ZSCORE_JSON, 'r') as f:
        zscore_dict = json.load(f)
    mean = zscore_dict[model_id]['mean']
    std = zscore_dict[model_id]['std']
    zscore = (y_pred - mean) / std
    return zscore