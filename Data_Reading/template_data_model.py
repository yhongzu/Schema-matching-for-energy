
import pandas as pd
import json
import sys
sys.path.append((r"."))
from Data_Reading.matrys_data_model import Data_model
from Data_Reading.lsp_dataset import Lsp_dataset  

class Template_data_model(object):

    def __init__(self):
        ld = Lsp_dataset ()
        self.lsp_data = ld.load_lsp_dataset()
        

    def template_data_model(self):
        read_path = r"MATRYCS Data\MATRYCS_data_model_v11.xlsx"
        categories = pd.read_excel(read_path, sheet_name=None)
        template_data_model = {}
        # just for saving as json_file
        template_data_model_json = {}
        for c_name in categories:
            df_c = pd.read_excel(read_path, sheet_name=c_name)
            df_c['Entity'].fillna(axis=0,method='ffill',inplace=True)
            # how many entities in a category 
            entities=[]
            # Because the first index doesn't have value
            for i in range(1,len(df_c.index)):
                e_name = df_c.loc[i,'Entity']   
                entities.append(e_name)
            entities = list({}.fromkeys(entities).keys())
            category = {}
            category_json = {}
            for e_name in entities:
                df_e = []
                # Because the first index doesn't have value
                # Read the data in each category by entity
                for i in range(1,len(df_c.index)):
                    if df_c.loc[i,'Entity'] == e_name:
                        df_e.append(df_c.loc[i,df_c.columns])         
                df_e = pd.DataFrame(df_e)
                df_e.columns = df_c.columns
                df_e = df_e.reset_index(drop=True)
                

                E = df_e[['Entity Attributes','Attribute description']]
                entity = {}
                entity_json = {}
                for lsp_name in self.lsp_data.keys():
                    index_num = list(df_e.columns).index(lsp_name)
                    temp_lsp = list(df_e.iloc[:,index_num+2])
                    temp_datasets = []
                    # change str to dict
                    for i in range(len(temp_lsp)):
                        if type(temp_lsp[i]) !=type(None):
                            try:
                                temp_lsp[i] = eval(temp_lsp[i])
                                temp_datasets.append(list(temp_lsp[i].keys()))
                            except KeyError:
                                continue
                            except TypeError:
                                continue
                    temp_datasets = [i for k in temp_datasets for i in k]
                    temp_datasets = list({}.fromkeys(temp_datasets).keys())
                    
                    df_datasets = []
                    if len(temp_datasets)>0:
                        for d_name in temp_datasets:
                            temp_d = []
                            for i in range(len(temp_lsp)):
                                try:
                                    attri = temp_lsp[i][d_name]
                                    temp_d.append(attri)
                                except:
                                    temp_d.append(None)
                            dataframe = pd.DataFrame(index=range(len(temp_d)),columns=[d_name])
                            for i in dataframe.index:
                                dataframe.loc[i,d_name] = temp_d[i]
                            # temp_d = pd.DataFrame(temp_d)
                            # temp_d.rename(columns={0:d_name},inplace=True)
                            df_datasets.append(dataframe)
                    
                    if len(df_datasets)>0:
                        df_datasets = pd.concat(df_datasets,axis=1,join='outer')
                        df_temp = []
                        df_temp.append(E)
                        df_temp.append(df_datasets)
                        df_temp = pd.concat(df_temp,axis=1,join='outer')
                        df_temp.dropna(axis='index', how='all', inplace=True)
                        df_temp = df_temp.reset_index(drop=True)
                        entity[lsp_name] = df_temp
                        entity_json[lsp_name] = json.loads(df_temp.to_json())
                    else:
                        entity[lsp_name] = None
                        entity_json[lsp_name] = None
                category[e_name] = entity
                category_json[e_name] =  entity_json
            template_data_model[c_name] = category
            template_data_model_json[c_name] = category_json
        file_name = 'template_data_model'
        with open(r'Data_Reading\template_json_files\{0}.json'.format(file_name),"w") as f:
            f.write(json.dumps(template_data_model_json,indent=4))
        return template_data_model

    

if __name__ == '__main__':
    tdm = Template_data_model()
    t = tdm.template_data_model()
    print(t)
    pass
