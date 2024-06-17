from pydantic import BaseModel
import sys
import pandas as pd
import os
if 'config' not in  sys.modules:

    sys.path.append(os.path.dirname(__file__))
    from  .. import  config 
else:
    import config 


ID_COL = 0
MODEL_TABLE = config.get_model_table_path()
class DeepBindData(BaseModel):
    protein : str
    species: str
    cite : str


class DeepbindModel():

    @classmethod
    def generate_id(cls, cite):
        """
        generate an id for the model.
        it won't be saved until generating the model object
        """
        model_df = pd.read_csv(MODEL_TABLE, sep='\t')
        model_id = f'{cite}.{model_df[model_df.columns[ID_COL]].str.contains(cite).sum()}'
        return model_id

    def __init__(self,protein, species,experiment, experiment_details, cite, input_shape,model_id= '', source_path=None):
        self.cite = cite
        self.model_id = self.generate_id(self.cite) if model_id == '' else model_id
        self.protein = protein
        self.species = species
        self.experiment = experiment
        self.experiment_details = experiment_details
        self.source_path = source_path
        self.experiment_details.update({'input_shape': input_shape})
    
    def save_model_to_table(self):
        """
        save the model to the model table
        """
        #check if the file exists in the table
        model_df = pd.read_csv(MODEL_TABLE, sep='\t')
        if self.model_id in model_df[model_df.columns[ID_COL]].values:
            self.model_id = self.generate_id(self.cite)
            print(f'model with id {self.model_id} already exists in the model table')
            print(f'new model id generated: {self.model_id}')
        new_model_df = pd.DataFrame([self.get_db_data()])
        new_model_df['source'] = 'IB_generated'
        new_model_df.to_csv(MODEL_TABLE, sep='\t', mode='a', header=False, index=False)
        print(f'model with id {self.model_id} saved to the model table')

    def get_db_data(self) -> dict:
        """
        get the cite source of the file
        """
        return {
            'model_id': self.model_id,
            'protein': self.protein,
            'species': self.species,
            'experiment': self.experiment,
            'experiment_details': self.experiment_details,
            'cite': self.cite,
            'source': self.source_path
        }
    
    def get_id(self):
        return self.model_id
    
    def get_experiment_details(self):
        return self.experiment_details
    
    def get_protein(self):
        return self.protein