from html import entities
from logging import error
import pandas as pd
import sys
sys.path.append((r"."))
from Data_Reading.template_data_model import Template_data_model
from Data_Reading.lsp_dataset import Lsp_dataset
from Data_Reading.matrys_data_model import Data_model
from Methodology.matching import Dataset_level_matching,Attribute_level_matching
class Evaluation(object):
    def __init__(self):
        tdm = Template_data_model()
        lds = Lsp_dataset()
        dm = Data_model()
        # the template file
        self.t_data_model = tdm.template_data_model()
        # {lsp1:df_file,lsp2:df_file...}
        self.lsp_data = lds.load_lsp_dataset()
        # {category1:{entity1:{df_e},entity2:{df_file}...}...}
        self.matrys_data = dm.load_data_model(data_model='matrys_data_model')


    def _datasets_generation(self):
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

    def _matrys_entities(self):
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

    def _dataset_level_template(self):
        '''
        Based on the template file, which is from the matrys data.

        '''
        dl_template = {}
        for c_name,entities in self.t_data_model.items():
            for e_name,entity in entities.items():
                temp = []
                for lsp_name,df_lsp in entity.items():
                    if type(df_lsp) ==type(pd.DataFrame()):   
                        temp.append(df_lsp.columns)
                if len(temp)>0:
                    temp = [j for i in temp for j in i]
                    temp = list({}.fromkeys(temp).keys())
                else:
                    temp = None
                dl_template[e_name] = temp
        return dl_template
        
    def _attribute_level_template(self):
        '''
        Based on the template file, which is from the matrys data.
        
        '''
        al_template = {}
        for c_name,entities in self.t_data_model.items():
            for e_name,entity in entities.items():
                df_temp = []
                for lsp_name, df_datasets in entity.items():
                    if type(df_datasets) ==type(pd.DataFrame()):  
                        df_temp.append(df_datasets)
                if len(df_temp)>0:
                    df_temp = pd.concat(df_temp,axis=1,join='outer')
                    df_temp = df_temp.loc[:,df_temp.columns.duplicated()==False]
                else:
                    df_temp = None
                al_template[e_name] = df_temp
        return al_template

    def dataset_level_evaluation(self,top_pairs_score=None,top_entities=None):
        '''
        MRR(Mean Reciprocal Rank)
        '''
        datasets = self._datasets_generation()
        datasets_num = len(datasets)
        data_model_entities,data_model_descriptions = self._matrys_entities()
        entities_num = len(data_model_entities)

         # the number of Attributes from all entities
        entities_attributes_sum = 0
        for e_name,df_entity in data_model_entities.items():
            entities_attributes_sum+=len(df_entity.index)
        # the number of Attributes from all datasets      
        datasets_attributes_sum = 0
        datasets_attri_num_dict = {}
        for d_name,df_dataset in datasets.items():
            datasets_attri_num_dict[d_name] = len(df_dataset.index)
            datasets_attributes_sum+=len(df_dataset.index)

        # F1 score
        TP_FN = 0
        TP_FP = 0
        TP = 0
        for d_name, k_entities in top_entities.items():
            if len(k_entities)>0:
                TP_FP+=1
                
        # MRR
        Q = len(top_entities)
        R_sum = 0
        for e_name,df_datasets in self._dataset_level_template().items():
            if type(df_datasets)!=type(None):
                # 2 means entity attributes and description two columns
                TP_FN+=(len(df_datasets)-2)
            for d_name, k_entities in top_entities.items():
                temp_r = 0
                if type(df_datasets)!=type(None) and d_name in df_datasets:
                    # As long as one of the three matches correctly, the match is correct
                    if e_name in k_entities:
                        TP+=1
                    for i in range(len(k_entities)):
                        if k_entities[i] == e_name:
                            rank = i+1
                            temp_r = 1/rank
                R_sum += temp_r
        MRR = round(R_sum/Q,3)

        
        try:
            r = round(TP/TP_FN,3)
        except ZeroDivisionError as e:
            print(e)
            r = 0
        try:
            p = round(TP/TP_FP,3)
        except ZeroDivisionError:
            p = 0
        try:
            f1 = round((2*p*r)/(p+r),3)
        except ZeroDivisionError:
            f1=0

        result = {
            'R_sum':R_sum,
            'Q':Q,
            'MRR':MRR,
            'r':r,
            'p':p,
            'f1':f1,
            'datasets_num':datasets_num,
            'entities_num':entities_num,
            'datasets_attributes_sum':datasets_attributes_sum,
            'entities_attributes_sum':entities_attributes_sum,
            'correct_top-k':TP,
            }
        return result


    def _singe_mapping_evaluation(self,df_template=None,df_mapping=None):
        '''
        input:df_template
              df_mapping
        output:
              r,p,f1,
        '''
        goal = df_mapping.columns
        df_t = df_template.loc[:,goal]
        temp = [df_t,df_mapping] 
        df_temp = pd.concat(temp,axis=1,join='outer')
        TP= 0
        # TP+FP
        TP_FP = 0
        # TP+FN
        TP_FN = 0
        for i in df_temp.index:
            # true_attri = [a,b,c]
            true_attri = df_temp.iloc[i,1]
            #  true_attri = [a,b,c]   
            pred_attri = df_temp.iloc[i,3]
            if type(true_attri)!=type(None):
                for x in true_attri:
                    TP_FN+=1
                if type(pred_attri)!=type(None) :
                    for x in pred_attri:
                        TP_FP+=1
                        if type(true_attri)!=type(None) and x in true_attri:
                            TP+=1
        
        try:
            r = round(TP/TP_FN,3)
        except ZeroDivisionError as e:
            print(e)
            r = 0
        try:
            p = round(TP/TP_FP,3)
        except ZeroDivisionError:
            p = 0
        try:
            f1 = round((2*p*r)/(p+r),3)
        except ZeroDivisionError:
            f1=0
        
        return r,p,f1,TP_FN,TP_FP,TP


    def attribute_level_evaluation(self,top_entities=None,pairs_mapping=None):
        # the sum of datasets and entities
        data_model_entities,data_model_descriptions =self._matrys_entities()
        entities_num = len(data_model_entities)
        datasets = self._datasets_generation()
        datasets_num = len(datasets)

        # the number of Attributes from all entities
        entities_attributes_sum = 0
        for e_name,df_entity in data_model_entities.items():
            entities_attributes_sum+=len(df_entity.index)
        # the number of Attributes from all datasets      
        datasets_attributes_sum = 0
        datasets_attri_num_dict = {}
        for d_name,df_dataset in datasets.items():
            datasets_attri_num_dict[d_name] = len(df_dataset.index)
            datasets_attributes_sum+=len(df_dataset.index)
        
        # f1 score attribute-level
        attr_level_tup = []
        for e_name,df_datasets in self._dataset_level_template().items():
            for d_name, k_entities in top_entities.items():
                if type(df_datasets)!=type(None) and d_name in df_datasets:
                    # As long as one of the three matches correctly, the match is correct
                    if e_name in k_entities:
                        attr_level_tup.append((d_name,e_name))
        correct_top_k = len(attr_level_tup)

        attr_level_attributes_sum = 0
        for d_name,df_dataset in datasets.items():
            for tup in attr_level_tup:
                if d_name == tup[0]:
                    attr_level_attributes_sum+=len(df_dataset.index)

        TP_FN_attri_level= 0
        TP_FP_attri_level = 0
        for e_name,df_template in self._attribute_level_template().items():
            for tup in attr_level_tup:
                if type(df_template)!=type(None):
                    if tup[1] == e_name and tup[0] in df_template.columns:
                        for i in df_template.index:
                            true_attri = df_template.loc[i,tup[0]]
                            if type(true_attri)!=type(None):
                                for x in true_attri:
                                    TP_FN_attri_level+=1

        for pred_tup,df_mapping in pairs_mapping.items():
            for tup in attr_level_tup:
                # tup:(d_name,e_name)
                if type(df_mapping) == type(pd.DataFrame()):
                    if tup[0] == pred_tup[0]:
                        for i in df_mapping.index:
                            pred_attri = df_mapping.loc[i,tup[0]]
                            if type(pred_attri)!=type(None):
                                for x in pred_attri:
                                    TP_FP_attri_level += 1

        # f1 score entire
        TP_FN_entire= 0
        TP_FP_entire = 0
        TP = 0
        for e_name,df_template in self._attribute_level_template().items():
            if type(df_template)!=type(None):
                temp = list(df_template.columns)
                temp.remove('Entity Attributes')
                temp.remove('Attribute description')
                num = len(df_template.index)
                for d_name in temp:
                    for i in range(num):
                        true_attri = df_template.loc[i,d_name]
                        if type(true_attri)!=type(None):
                            for x in true_attri:
                                TP_FN_entire+=1
        for tup,df_mapping in pairs_mapping.items():
            # tup:(d_name,e_name)
            if type(df_mapping) == type(pd.DataFrame()):
                for i in df_mapping.index:
                    pred_attri = df_mapping.loc[i,tup[0]]
                    if type(pred_attri)!=type(None):
                        for x in pred_attri:
                            TP_FP_entire += 1

        # f1-weighted
        correct_matching_dataset_attri_sum = 0
        correct_matching_dataset_num = 0
        for e_name,df_template in self._attribute_level_template().items():
            for tup,df_mapping in pairs_mapping.items():
                if type(df_template)!=type(None):
                    if tup[1] == e_name and tup[0] in df_template.columns:
                        correct_matching_dataset_num+=1
                        r,p,f1,TP_FN,TP_FP,tp = self._singe_mapping_evaluation(
                                                                df_template=df_template,
                                                                df_mapping=df_mapping
                                                                )
                        
                        TP+=tp
                        dataset_attri_num = datasets_attri_num_dict[tup[0]]
                        correct_matching_dataset_attri_sum+=dataset_attri_num
                        # print(tup)
                        # print(r,p,f1,TP_FN,TP_FP,TP,dataset_attri_num)
                        # ====================================================
                        # only correct matching dataset
                        
        try:
            r_entire = round(TP/TP_FN_entire,3)
            r_attr = round(TP/TP_FN_attri_level,3)

        except ZeroDivisionError:
            r_entire = 0
            r_attr =0

        try:
            p_entire = round(TP/TP_FP_entire,3)
            p_attr = round(TP/TP_FP_attri_level,3)

        except ZeroDivisionError:
            p_entire = 0
            p_attr = 0

        try:
            f1_entire = round((2*p_entire*r_entire)/(p_entire+r_entire),3)
            f1_attr = round((2*p_attr*r_attr)/(p_attr+r_attr),3)

        except ZeroDivisionError:
            f1_entire = 0
            f1_attr = 0




        result = {}
        result['TP']=TP
        result['TP_FP_entire']=TP_FP_entire
        result['TP_FN_entire']=TP_FN_entire
        result['TP_FN_attri_level']=TP_FN_attri_level
        result['TP_FP_attri_level']=TP_FP_attri_level

        result['r_entire']=r_entire
        result['p_entire']=p_entire
        result['f1_entire']=f1_entire

        result['r_attri-level']=r_attr
        result['p_attri-level']=p_attr
        result['f1_attri-level']=f1_attr

        result['correct_top_k']=correct_top_k
        result['correct_matching_dataset_num']=correct_matching_dataset_num
        result['attr-level_attributes_sum']=attr_level_attributes_sum
        result['correct_matching_attributes_num']=TP

        return result
                       

if __name__ == '__main__':
    e = Evaluation()
    # dlm = Dataset_level_matching()
    # alm = Attribute_level_matching()
    # print(e._dataset_level_template())
    print(e._attribute_level_template())
    # top_entities,top_pairs_score = dlm.load_top_result(method='edit_distance',k=1)
    # result = e.dataset_level_evaluation(top_pairs_score=top_pairs_score,top_entities = top_entities)
    # pairs_mapping,pairs_score = alm.load_mapping_result(k=3,method='word2vec')
    # print(pairs_mapping)
    # result = e.attribute_level_evaluation(pairs_mapping=pairs_mapping)
    # print(pairs_mapping)
    # print(result)
    pass
