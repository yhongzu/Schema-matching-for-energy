import numpy as np
import pandas as pd
import json
import sys
sys.path.append((r"."))
from Data_Reading.matrys_data_model import Data_model
from Data_Reading.lsp_dataset import Lsp_dataset
from Data_Preprocessing.data_preprocessing import Data_preprocessing
from Methodology.matching import Dataset_level_matching,Attribute_level_matching

class Data_Harmonization(object):
    def __init__(self,data_model='matrys_data_model'):
        '''
        data_model choosing
        '''
        # self.dp = Data_preprocessing()
        dm = Data_model()
        lds = Lsp_dataset()
        # {lsp1:df_file,lsp2:df_file...}
        self.lsp_data = lds.load_lsp_dataset()
        # {category1:{entity1:{df_file},entity2:{df_file}...}...}
        self.matrys_data = dm.load_data_model(data_model=data_model)

    def datasets_generation(self):
        '''
        output={name1:df_dataset,name2:df_dataset...}<dict>
        '''
        lsp_datasets = {}
        for df_datasets in self.lsp_data.values():
            # symspelled_lsp = self.dp.df_preprocessing(df_lsp)
            for d_name in df_datasets.columns:
                df_dataset = pd.DataFrame(df_datasets[d_name])
                df_dataset.columns = [d_name]
                df_dataset.dropna(axis='index', how='all', inplace=True)
                # attributes_list =[i for i in attributes_list if i is not None]
                lsp_datasets[d_name] = df_dataset
        return lsp_datasets
    def entities_generation(self):
        '''
        output:(entities = {name1:df_entity,name2:df_entity...},
                descriptions = {name1:df_description,name2:df_description...})
        '''
        data_model_entities = {}
        data_model_descriptions = {}
        for entities in self.matrys_data.values():
            for e_name,df_entity in entities.items(): 
                temp_attris =  pd.DataFrame(df_entity['Entity Attributes'])
                temp_attris.columns = ['Entity Attributes']     
                data_model_entities[e_name] = temp_attris
                temp_descriptions = pd.DataFrame(df_entity['Attribute description'])
                temp_descriptions.columns = ['Attribute description']
                data_model_descriptions[e_name] = temp_descriptions
        return data_model_entities,data_model_descriptions


    def dataset_level_result(self,k=3,method='edit_distance'):
        lsp_datasets = self.datasets_generation()
        matrys_entities,matrys_description = self.entities_generation()
        dlm = Dataset_level_matching()
        top_entities,top_pairs_score = dlm.dataset_top(
                                                       datasets=lsp_datasets,
                                                       entities=matrys_entities,
                                                       k=k,
                                                       method=method
                                                       )

        result = {}
        for c_name,entities in self.matrys_data.items():
            category = {}
            for e_name in entities.keys():
                entity = {}
                for lsp_name,df_datasets in self.lsp_data.items():
                    temp = []
                    for tup,score in top_pairs_score.items():
                        if tup[1] == e_name and tup[0] in df_datasets.columns:
                            temp.append(tup[0])
                    if len(temp)>0:
                        entity[lsp_name] = temp
                    else:
                        entity[lsp_name] = None
                category[e_name] = entity
            result[c_name] = category
        with open(r'Data_Harmonization\json_files\dlm_{0}_top{1}_harmonization.json'.format(method,k),"w") as f:
            f.write(json.dumps(result,indent=4))
        return result
    # load the dataset_level result
    def load_top_result(self,k=3,method='edit_distance'):
        f = open(r'Data_Harmonization\json_files\dlm_{0}_top{1}_harmonization.json'.format(method,k),'r')
        result = json.load(f)
        return result


    def attribute_level_result(self,method='edit_distance'):
        lsp_datasets = self.datasets_generation()
        matrys_entities,matrys_description = self.entities_generation()
        
        dlm = Dataset_level_matching()
        top_entities,top_pairs_value = dlm.load_top_result(k=3)

        alm = Attribute_level_matching()
        paris_mappping,pair_score = alm.dataset2entity(
                                    datasets=lsp_datasets,
                                    entities=matrys_entities,
                                    top_entities=top_entities,
                                    method=method
                                    )
        result = {}
        # for stroting as json file
        result_json = {}
        for c_name,entities in self.matrys_data.items():
            category = {}
            category_json = {}
            for e_name in entities.keys():
                entity = {}
                entity_json = {}
                for lsp_name,df_datasets in self.lsp_data.items():
                    temp = []
                    for tup,k in paris_mappping.items():
                        if tup[1] == e_name and tup[0] in df_datasets.columns:
                            temp.append(k)
                    if len(temp)>0:
                        temp = pd.concat(temp,axis=1,join='outer')
                        # de-duplicates in columns
                        temp = temp.loc[:,temp.columns.duplicated()==False]
                        temp_json = json.loads(temp.to_json())
                        entity[lsp_name] = temp
                        entity_json[lsp_name] = temp_json
                    else:
                        entity[lsp_name] = None
                        entity_json[lsp_name] = None
                category[e_name] = entity
                category_json[e_name] = entity_json
            result[c_name] = category
            result_json[c_name] = category_json
        with open(r'Data_Harmonization\json_files\alm_{0}_mapping_harmonization.json'.format(method),"w") as f:
            f.write(json.dumps(result_json,indent=4))
        return result
    def load_mapping_result(self,method='edit_distance'):
        f = open(r'Data_Harmonization\json_files\alm_{0}_mapping_harmonization.json'.format(method),'r')
        result = json.load(f)
        return result


if __name__ == "__main__":
    dh = Data_Harmonization()
    # r =dh.dataset_level_result(method='sbert')
    r =dh.attribute_level_result(method='edit_distance')
    
    print(r)