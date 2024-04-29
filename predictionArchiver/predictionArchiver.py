from pydantic import BaseModel, validator
from typing import List, Dict
import os
import json
import pandas as pd
ARCHIVE_DIR = 'DB_archive'
SAVE_DIR = '/dsi/gonen-lab/users/toozig/projects/deepBind_pipeline/DB_predictions'
ORIGINAL = 'db_original'
GENERATED = 'IB_generated'
MAX_ID_LEN = 100



class PredictionFile(BaseModel):
    file_id: str
    path: str
    phase: str
    window: int
    shift: int
    exist: bool
 

    def __eq__(self, other):
        if isinstance(other, PredictionFile):
            return all([
                self.file_id == other.file_id,
                self.path == other.path,
                self.phase == other.phase,
                self.window == other.window,
                self.shift == other.shift,
                self.exist == other.exist
            ])
        return False

    def __hash__(self):
        return hash((self.file_id, self.path, self.phase, self.window, self.shift, self.exist))

class PhaseDict(BaseModel):
    phase: str
    files: Dict[str, List[PredictionFile]]  


    def drop_duplicates(self):
        for key, value in self.files.items():
            self.files[key] = list(set(value))
        return self


    @classmethod
    def from_json(self, json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        return self.from_dict(json_data)
        

    @classmethod
    def from_dict(cls, d):
        phase = d['phase']
        files = {k: [PredictionFile(**v) for v in vs] for k, vs in d['files'].items()}
        return cls(phase=phase, files=files)


    def add_item(self, key, value):
        if isinstance(value, dict):
            value = PredictionFile(**value)
        if key in self.files:
            file_list = self.files[key]
            if value in file_list:
                return
            self.files[key].append(value)
        else:
            self.files[key] = [value]  

    def items(self):
        return self.files.items()

    def __setitem__(self, key, value):
        self.files[key] = value

    def __getitem__(self, key):
        return self.files.get(key)

    def __contains__(self, item):
        return item in self.files

    def merge(self, other):
        for key, value in other.items():
            if len(value) and type(value[0]) == dict:
                value = [PredictionFile(**v) for v in value]
            if key in self.files:
                self.files[key].extend(value)
            else:
                self.files[key] = value


import ast
def str_to_dict(x):
    return ast.literal_eval(x)

import json


class PredictionSaver:
    _instance = None

    def __new__(cls,name=''):
        if cls._instance is None:
            cls.__init_instance(name)
        return cls._instance


    def get_records_by_id(self, id):
        """
        returns all the records that have the same id
        """
        record_list = []
        for p_dict in [self.P1_dict, self.P2_dict, self.P3_dict]:
            if id in p_dict.files:
                record_list.extend(p_dict.files[id])
        return [i.dict() for i in record_list]
    
    @classmethod
    def __get_archive_dir(cls, name):
        return os.path.join(SAVE_DIR, name, 'metadata')


    @classmethod
    def __init_instance(cls, name):
        parsed_name = os.path.basename(name).split('.')[0]
        project_dir = cls.__get_archive_dir(parsed_name)
        json_file = os.path.join(project_dir, 'all_metadata.json')
        if os.path.exists(json_file):
            print(f'loading instance from:\n {json_file}\n')
            return cls.from_json(json_file)
        else:
            print('creating new instance')
            cls._instance = super(PredictionSaver, cls).__new__(cls)
            cls.name = name
            cls.non_exist_models = []
            cls.model_df = pd.DataFrame()
            cls.P1_dict = PhaseDict(phase='P1', files={})
            cls.P2_dict = PhaseDict(phase='P2', files={})
            cls.P3_dict = PhaseDict(phase='P3', files={})
            return cls._instance


    @classmethod
    def from_json(cls, json_input):
        #incase the input is string of json or path to json
        if isinstance(json_input, dict):
            json_data = json_input
        else:
            json_data = cls.__open_json(json_input)

        instance = super(PredictionSaver, cls).__new__(cls)
        instance.name = json_data['name']
        instance.non_exist_models = json_data['non_exist_models']
        instance.model_df = pd.DataFrame(json_data['model_df'])
        instance.P1_dict = cls.__init_p_dict('P1',json_data['P1_dict']) 
        instance.P2_dict = cls.__init_p_dict('P2',json_data['P2_dict'])
        instance.P3_dict = cls.__init_p_dict('P3',json_data['P3_dict'])
        cls._instance = instance
        return instance

    @classmethod
    def from_instace(cls, instance):
        cls._instance = instance
        return instance

    @classmethod
    def __init_p_dict(self,p,dict_str):
        p_dict = json.loads(dict_str)
        obj = PhaseDict.from_dict(p_dict)
        return obj


    @classmethod
    def __open_json(cls,json_path):
        #chekc if the input is path
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        else:
            json_data = json.loads(json_data)
        return json_data

    def get_save_dir(self):
        return os.path.join(SAVE_DIR, self.name)

        
    def __get_to_update(self, p_instance):
        path = f"{SAVE_DIR}/{self.name}/{p_instance.phase}"
        exists_list =os.listdir(path)
        to_update = []
        for f in exists_list:
            full_path = os.path.join(path, f)
            seq_id = f.replace('.csv', '')
            if seq_id in p_instance:
                for file in p_instance[seq_id]:
                    if file.path == full_path:
                        break
            else:
                to_update.append(full_path)
        return to_update

    def __update_exist(self,p ,to_update, window, shift):
        for file in to_update:
            seq_id = file.split('/')[-1].replace('.csv', '')
            cur_dict = {'file_id': seq_id, 'path': file,
            'phase': p, 'window' : window, 'shift': shift, 'exist' : True}
            self.add_item(p,seq_id, cur_dict)

    def __update_exist_helper(self, p_instance, window, shift):
        to_update = self.__get_to_update(p_instance)
        print(f'updating {p_instance.phase} with {len(to_update)} files')
        self.__update_exist(p_instance.phase, to_update, window, shift)

    def update_exists_file(self, p, window =-1, shift =-1):
        if p == 'P1':
            p_instance = self.P1_dict
        elif p == 'P2':
            p_instance = self.P2_dict
        elif p == 'P3':
            p_instance = self.P3_dict
        if p == 'P2' or p == 'P1':
            if window == -1 or shift == -1:
                raise ValueError('window and shift must be provided for P1 and P2')
        self.__update_exist_helper(p_instance, window, shift)
      

        def drop_duplicates(self):
            self.P1_dict = self.P1_dict.drop_duplicates()
            self.P2_dict = self.P2_dict.drop_duplicates()
            self.P3_dict = self.P3_dict.drop_duplicates()
            return self

    def get_p_instance(self, P):
        if P == 'P1':
            return self.P1_dict
        elif P == 'P2':
            return self.P2_dict
        elif P == 'P3':
            return self.P3_dict


    def add_item(self, P, key, value):
        """
        add item to a dict
        """
        cur_dict = self.get_dict(P)
        cur_dict.add_item(key, value)
    
    def merge_json(self, saver_json):
        try:
            self.P1_dict.merge(self.__init_p_dict('P1',saver_json['P1_dict']) )
            self.P2_dict.merge(self.__init_p_dict('P2',saver_json['P2_dict']))
            self.P3_dict.merge(self.__init_p_dict('P3',saver_json['P3_dict']))
        except Exception as e:

            print('could not merge json')
            print(saver_json)
            raise e
        return self

    def json(self):
        return {'name': self.name, 
        'non_exist_models': self.non_exist_models, 
        'model_df': self.model_df.to_dict(), 
        'P1_dict': self.P1_dict.json(), 
        'P2_dict': self.P2_dict.json(), 
        'P3_dict': self.P3_dict.json()}

    def get_dict(self,P):
        if P == 'P1':
            return self.P1_dict
        elif P == 'P2':
            return self.P2_dict
        elif P == 'P3':
            return self.P3_dict

    def get_prediction_file_by_path(self,P,path):
        cur_dict = self.get_dict(P)
        for key, value in cur_dict.items():
            for file in value:
                if file.path.split('/')[-1] == path.split('/')[-1]:
                    return file
        
    def update_dict(P,k,v):
        cur_dict = self.get_dict(P)
        cur_dict[k] = v



    def save_data(self):
        print('saving data')
        dir_path = self.__get_archive_dir(self.name) 
        os.makedirs(dir_path, exist_ok=True)
        # save the model_df
        model_df_path = os.path.join(dir_path, 'model_df.tsv')
        self.model_df.to_csv(model_df_path, sep='\t')
        # save the non_exist_models
        non_exist_models_path = os.path.join(dir_path, 'non_exist_models.txt')
        with open(non_exist_models_path, 'w') as f:
            f.write('\n'.join(self.non_exist_models))
        # save the P1_dict
        P1_dict_path = os.path.join(dir_path, 'P1_dict.json')
        with open(P1_dict_path, 'w') as f:
            json.dump(self.P1_dict.json(), f)
        # save the P2_dict 
        P2_dict_path = os.path.join(dir_path, 'P2_dict.json')
        with open(P2_dict_path, 'w') as f:
            json.dump(self.P2_dict.json(), f)
        # save the P3_dict
        P3_dict_path = os.path.join(dir_path, 'P3_dict.json')
        with open(P3_dict_path, 'w') as f:
            json.dump(self.P3_dict.json(), f)
        all_metadata = os.path.join(dir_path, 'all_metadata.json')
        with open(all_metadata, 'w') as f:
            json.dump(self.json(), f)
        print(f'data saved at {dir_path}')
            

    def set_model_df(self, model_df):
        self.model_df = model_df
    
    def get_model_df(self):
        return self.model_df

    def set_non_exist_models(self, non_exist_models):
        self.non_exist_models = non_exist_models

    def __update_dict(self, P, key, value):
        cur_dict = self.get_dict(P)
        if key in cur_dict:
            cur_dict[key].append(value)
        else:
            cur_dict[key] = [value]
        
    def get_P1_saving_path(self,model_type,model_id ,seq_id,window_size,shift):
        save_dir = f'{SAVE_DIR}/{self.name}/P1/{model_type}/{model_id}'
        save_dir += f'/{seq_id[:MAX_ID_LEN]}.tsv'
        file_pred = PredictionFile( path=save_dir, phase='P1', file_id=model_id, window=window_size, shift=shift, exist=True)
        #save_dir, 'P1', str(model_id), window_size, shift)
        # print(f'P1 saving path: {save_dir}')
        self.__update_dict('P1', seq_id, file_pred)
        return save_dir


    def get_P2_saving_path(self,model_type,seq_id,window_size,shift):
        save_path = f'{SAVE_DIR}/{self.name}/P2/{model_type}/'
        save_path += f'{seq_id[:MAX_ID_LEN]}.tsv'
        file_pred = PredictionFile(path=save_path, phase='P2', file_id=seq_id, window=window_size, shift=shift, exist=True)
        # file_pred = PredictionFile(save_path, 'P2', str(seq_id), window_size, shift)
        self.__update_dict('P2', seq_id, file_pred)

        return save_path

 
    def get_P3_saving_path(self, P2_path, to_save=True):
        P2_file_pred = self.get_prediction_file_by_path('P2',P2_path)
        
        seq_name = P2_file_pred.file_id
        path = f'{SAVE_DIR}/{self.name}/P3/{seq_name[:MAX_ID_LEN]}.csv'
        file_pred = PredictionFile(path=path, phase='P3', file_id=seq_name, window=-1, shift=-1, exist=True)
        if to_save:
            self.__update_dict('P3', seq_name, file_pred)
        return path

    def save_P1_model(self,df,model_id, seq_id, window_size, shift):
        save_path = self.get_P1_saving_path(GENERATED, model_id, seq_id, window_size, shift, )
        self.save_prediction_df(df, save_path)
        return save_path

    def save_prediction_df(self,df, save_path):
        # checkes if the folder exists
        if any(df.columns.str.contains('unnamed')):
            df = df.drop(columns=df.columns[df.columns.str.contains('unnamed')])
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            # add to the file
            cur_df = pd.read_csv(save_path, sep='\t', index_col=0)
            df = cur_df.combine_first(df)
        df.to_csv(save_path, sep='\t')
        # print(f'saved at {save_path}')
