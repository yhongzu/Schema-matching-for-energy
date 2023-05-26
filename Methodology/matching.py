import numpy as np
import json
import sys
sys.path.append((r"."))
from Data_Preprocessing.data_preprocessing import Data_preprocessing
from Data_Reading.matrys_data_model import Data_model
from Data_Reading.lsp_dataset import Lsp_dataset


class Datasets_Generation(object):

    def __init__(self,data_model='matrys_data_model'):
        lds = Lsp_dataset()
        dm = Data_model()
        # {lsp1:df_datasets,lsp2:df_datasets...}
        self.lsp_data = lds.load_lsp_dataset()
        # {category1:{entity1:{df_e},entity2:{df_file}...}...}
        self.matrys_data = dm.load_data_model(data_model=data_model)
    
    def lsp_datasets(self):
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

    def matrys_entities(self):
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


from Matcher.String_based_methods.string_based_methods import Edit_distance,Jaccard
from Matcher.Konwledge_based_methods.word_net import Word_net
from Matcher.Konwledge_based_methods.wiki_cons import Wiki_cons
from Matcher.Corpus_based_methods.wordembedding import Word2vec,Glove,Fast_text,Bert,Sbert


class Dataset_level_matching(object):

    def __init__(self):
        self.dp = Data_preprocessing()
        pass
    
    # The matchers can be used during Dataset_level 
    def _load_matcher(self,method='edit_distance'):
        '''
        method=<'edit_distance','sbert'>
        '''
        if method == 'edit_distance':
            matcher = Edit_distance()
        elif method == 'jaccard':
            matcher = Jaccard()
        elif method == 'word_net_pathsim':
            matcher = Word_net(art='path_sim')
        elif method == 'wiki_cons':
            matcher = Wiki_cons(art='con_sim4')
        elif method == 'word2vec':
            matcher = Word2vec()
        elif method == 'glove-300':
            matcher = Glove(model_name='glove-wiki-gigaword-300')
        elif method == 'fasttext-300':
            matcher = Fast_text()
        elif method == 'bert-base-uncased':
            matcher = Bert(model_name='bert-base-uncased')
        elif method == 'sbert':
            matcher = Sbert(model_name='sentence-transformers/all-MiniLM-L6-v2')
        return matcher

    def _dataset_similarity(self,d_name=None,dataset=None,e_name=None,entity=None,matcher=None):
        '''
        The similarity calculation process between a dataaset and an entity
        example:
            dataset<DataFrame> = d_name
                                attri1
                                attri2
                                ...
            entity<DataFrame> = Entity Attributes 
                                attri1             
                                attri2      
                                ...                 
        '''
        theshold = 0.1
        attri_matrix = {}
        dataset_phases = list(dataset.iloc[:,0])
        entity_phases = list(entity.iloc[:,0])      
        
        # scores matrix
        scores = matcher.sim(sentences_a=dataset_phases,
                                    sentences_b=entity_phases)
        for i in range(len(dataset_phases)):
            for j in range(len(entity_phases)):
                score = round(float(scores[i][j]),3)
                if score>theshold:
                    attri_matrix[(dataset_phases[i],entity_phases[j])] = score
                else:
                    attri_matrix[(dataset_phases[i],entity_phases[j])] = 0


        # Listed from largest to smallest   
        am_sorted = sorted(attri_matrix.items(),key=lambda x:x[1],reverse=True)
        # the form of tup:((i,j),similarity)
        max_tup = {}
        dataset_picked_tup  = []
        # entity_picked_tup  = []
        # An attri in an entity may match several attributes in a dataset, but an attribute in a dataset may not match several attributes in an entity
        for tup in am_sorted:
            # if tup[0][0] not in dataset_picked_tup and tup[0][1] not in entity_picked_tup:
            if tup[0][0] not in dataset_picked_tup:
                max_tup[tup[0]] = tup[1]
                dataset_picked_tup.append(tup[0][0])
                # entity_picked_tup.append(tup[0][1])
                list(set(dataset_picked_tup))
                # list(set(entity_picked_tup))
        max_value = list(max_tup.values())

        # add the name_similarity between the dataset and entity
        name_similarity = matcher._(str_a=d_name,str_b=e_name)
        max_value.append(name_similarity)

        # the mean value of all of the max_value
        mean = np.mean(np.array(max_value))
        return round(mean,3)


    def dataset_top(self,datasets=None,entities=None,k=3,method='edit_distance'): 
        '''
        top-k strategy
        input:
            datasets = {name1:df_dataset,name2:df_dataset...}
                the form of df_dataset:
                    df_dataset<DataFrame> = d_name
                                         attri1
                                         attri2
                        ...
                
            entities = {name1:df_entity,name2:df_entity...}
                the form of df_entity:
                    df_entity<DataFrame> = Entity Attributes  
                                        attri1    
                                        attri2  
                                        ...  

        output:
            top_entities = {d_name1:[e_name1,e_name2,e_name3],...}
            top_pairs_value = {(d_name,e_name1):s_value1,(d_name,e_name2):s_value2,(d_name,e_name3):s_value3,...} 
        ''' 

        matcher = self._load_matcher(method=method)
        # Threshold setting
        threshold = 0.1
        top_entities = {}
        # just for saving in json file
        top_pairs_score = {}
        for d_name,df_dataset in datasets.items():
            temp = {}
            for e_name,df_entity in entities.items():
                # the similarity score between two datesets
                score = self._dataset_similarity(
                                                d_name = d_name,
                                                dataset=df_dataset, 
                                                e_name = e_name,
                                                entity = df_entity,
                                                matcher= matcher,                                               
                                                )
                if score > threshold:
                    temp[(d_name,e_name)] = score
                else:
                    temp[(d_name,e_name)] = 0
            temp = sorted(temp.items(),key=lambda x:x[1],reverse=True)
            # [((d_name,e_name1),s_value1),((d_name,e_name2),s_value2),((d_name,e_name3),s_value3)]
            
            top_k = temp[0:k]
            
            k_entities = []
            for i in top_k:
                # for json file the type of the key must be str 
                # {(d_name,e_name1):s_value1,(d_name,e_name2):s_value2,(d_name,e_name3):s_value3,...}
                if i[1]>threshold:
                    top_pairs_score[i[0]] = i[1]
                    k_entities.append(i[0][1])
            # top_dict = {d_name1:[e_name1,e_name2,e_name3],...}
            if len(k_entities)>0:
                top_entities[d_name] =  k_entities  

        with open(r'Methodology\json_files\dlm_{0}_top{1}_pairs_score.json'.format(method,k),"w") as f:
            top_score = {}
            for i,j in top_pairs_score.items():
                top_score[str(i)] = j
            f.write(json.dumps(top_score,indent=4))
        with open(r'Methodology\json_files\dlm_{0}_top{1}_entities.json'.format(method,k),"w") as f:
            f.write(json.dumps(top_entities,indent=4))
        return top_entities,top_pairs_score

    def load_top_result(self,method='edit_distance',k=3):
        '''
        example
        top_entities = {d_name1:[e_name1,e_name2,e_name3],...}
        top_pairs_value = {(d_name,e_name1):s_value1,(d_name,e_name2):s_value2,(d_name,e_name3):s_value3,...}
        '''
        fp1 = open(r'Methodology\json_files\dlm_{0}_top{1}_entities.json'.format(method,k),"r")
        top_entities = json.load(fp1)
        fp2 = open(r'Methodology\json_files\dlm_{0}_top{1}_pairs_score.json'.format(method,k),"r")
        f2 = json.load(fp2)
        top_pairs_score = {}
        for tup,v in f2.items():
            top_pairs_score[eval(tup)] = v
        return top_entities,top_pairs_score


import pandas as pd
from Matcher.String_based_methods.string_based_methods import Edit_distance,Jaccard
from Matcher.Konwledge_based_methods.word_net import Word_net
from Matcher.Konwledge_based_methods.wiki_cons import Wiki_cons
from Matcher.Corpus_based_methods.wordembedding import Word2vec,Glove,Fast_text,Bert,Sbert
from Active_Learning.ALmatcher import ALmatcher
class Attribute_level_matching(object):

    def __init__(self):
        self.dp = Data_preprocessing()

    def _load_matcher(self,method='edit_distance'):

        '''
        The matchers during the attribute level
        method=<'edit_distance','jaccard','dice','word_net_pathsim','wiki_cons','word2vec'
                    'glove-300','fasttext-300','bert-base-uncased','sbert','ALmatcher'>
        '''
        if method == 'edit_distance':
            matcher = Edit_distance()
        elif method == 'jaccard':
            matcher = Jaccard()
        elif method == 'word_net_pathsim':
            matcher = Word_net(art='path_sim')
        elif method == 'wiki_cons':
            matcher = Wiki_cons(art='con_sim4')
        elif method == 'word2vec':
            matcher = Word2vec()
        elif method == 'glove-300':
            matcher = Glove(model_name='glove-wiki-gigaword-300')
        elif method == 'fasttext-300':
            matcher = Fast_text()
        elif method == 'bert-base-uncased':
            matcher = Bert(model_name='bert-base-uncased')
        elif method == 'sbert':
            matcher = Sbert(model_name='sentence-transformers/all-MiniLM-L6-v2')
        elif method == 'ALmatcher':
            matcher = ALmatcher()
        return matcher


    def _dataset_mapping(self,d_name=None,dataset=None,e_name=None,entity=None,matcher=None):
        '''
        example:
        input:
            dataset<DataFrame> = d_name
                                attri1
                                attri2
                                ...
            entity<DataFrame> = Entity Attributes or (Attributes description)
                                attri1             
                                attri2      
                                ...
        output:df_mapping
                (
                Entity Attributes or (Attributes description)  d_name
                attri1                                          NAN
                attri2                                          attri1
                attri3                                          attri2
                ... 
                , dataset_similarity
                )
                                
        '''
        dataset_att_num = len(dataset.index)
        threshold = 0.2
        attri_matrix = {}
        dataset_phases = list(dataset.iloc[:,0])
        entity_phases = list(entity.iloc[:,0])
        cosine_scores = matcher.sim(sentences_a=dataset_phases,
                                    sentences_b=entity_phases)
        for i in range(len(dataset_phases)):
            for j in range(len(entity_phases)):
                score = round(float(cosine_scores[i][j]),3)
                if  score>=threshold:               
                    attri_matrix[(dataset_phases[i],entity_phases[j])] = score
                else:
                    attri_matrix[(dataset_phases[i],entity_phases[j])] = 0


        # Listed from largest to smallest   
        am_sorted = sorted(attri_matrix.items(),key=lambda x:x[1],reverse=True)
        max_tup = {}
        dataset_picked_tup  = []
        # entity_picked_tup  = []
        # An attri in an entity may match several attributes in a dataset, but an attribute in a dataset may not match several attributes in an entity
        for tup in am_sorted:
            # the form of tup:((d,e),similarity)
            # if tup[0][0] not in dataset_picked_tup and tup[0][1] not in entity_picked_tup:
            if tup[0][0] not in dataset_picked_tup:
                max_tup[tup[0]] = tup[1]
                dataset_picked_tup.append(tup[0][0])
                # entity_picked_tup.append(tup[0][1])
                list(set(dataset_picked_tup))
                # list(set(entity_picked_tup))
        max_value = list(max_tup.values())
        # Use dataset name as an attribute
        name_similarity = matcher._(str_a=d_name, str_b=e_name)
        max_value.append(name_similarity)
        # the mean value of all of the max_value
        mean = np.mean(np.array(max_value))


        '''Attribute mapping Process'''
        # the name column of entity
        column_e = entity.columns[0]
        df_mapping = pd.DataFrame(index=range(len(entity.index)),columns=[column_e,d_name])
        # the Attribute_level mapping Table
        for i in df_mapping.index:
            df_mapping.loc[i,column_e] = entity.iloc[i,0]
        for i in df_mapping.index:
            temp = []
            for tup,score in max_tup.items():
                # Matching only if score is greater than the threshold
                if score >=threshold:
                    if tup[1] == df_mapping.loc[i,column_e]:
                        temp.append(tup[0])
            if len(temp)>0:
                df_mapping.loc[i,d_name] = temp
        return df_mapping,round(float(mean),3)

    def dataset2entity(self,datasets=None,entities=None,k=None,top_entities=None,method='edit_distance'):
        '''
        input:
        datasets = {name1:df_dataset,name2:df_dataset...}
                the form of df_dataset:
                    df_dataset<DataFrame> = d_name
                                         attri1
                                         attri2
                        ...
                
            entities = {name1:df_entity,name2:df_entity...}
                the form of df_entity:
                    df_entity<DataFrame> = Entity Attributes  
                                        attri1    
                                        attri2  
                                        ... 
            top_entities = {d_name1:[e_name1,e_name2,e_name3],...}
        '''
        matcher = self._load_matcher(method=method)
        threshold = 0.2
        pairs_mapping = {}
        pairs_score = {}
        for d_name, topk_entities in top_entities.items():
            temp = {}
            for e_name in topk_entities:
                # {(d_name,e_name):(df_mapping,s_value),...}
                r= self._dataset_mapping(
                                        d_name=d_name,
                                        dataset=datasets[d_name],
                                        e_name=e_name,
                                        entity=entities[e_name],
                                        matcher=matcher
                                        )
                if r[1] > threshold:
                    temp[(d_name,e_name)] = r
            temp = sorted(temp.items(),key=lambda x:x[1][1],reverse=True)
            # ((d_name,e_name),(df_mapping,s_value))
            if len(temp) > 0:
                pair = temp[0]
                pairs_mapping[pair[0]] = pair[1][0]
                pairs_score[pair[0]] = pair[1][1]

        # save json files
        with open(r'Methodology\json_files\alm_top{0}_{1}_mapping_result.json'.format(k, method),"w") as f:
            temp_mapping = {}
            for i,j in pairs_mapping.items():
                temp_mapping[str(i)] = json.loads(j.to_json())
            f.write(json.dumps(temp_mapping,indent=4))
        with open(r'Methodology\json_files\alm_top{0}_{1}_pairs_score.json'.format(k, method),"w") as f:
            temp_pairs = {}
            for i, j in pairs_score.items():
                temp_pairs[str(i)] = j
            f.write(json.dumps(temp_pairs,indent=4))
        return pairs_mapping, pairs_score

    def load_mapping_result(self,k=3,method='edit_distance'):
        '''
        example
        mapping_result = {(d_name,e_name1):df_mapping,...}
        pairs_score = {(d_name,e_name1):s_value1,(d_name,e_name2):s_value2,(d_name,e_name3):s_value3,...}
        '''
        fp1 = open(r'Methodology\json_files\alm_top{0}_{1}_mapping_result.json'.format(k,method),"r")
        f1 = json.load(fp1)
        pairs_mapping = {}
        for tup, mapping in f1.items():
            df_temp = {}
            for i ,j in mapping.items():
                temp = {}
                for x in j.keys():
                    temp[eval(x)] = j.get(x)
                df_temp[i] = temp
            df_file = pd.DataFrame(df_temp)
            pairs_mapping[eval(tup)] = df_file
        fp2 = open(r'Methodology\json_files\alm_top{0}_{1}_pairs_score.json'.format(k,method),"r")
        f2 = json.load(fp2)
        pairs_score = {}
        for tup,v in f2.items():
            pairs_score[eval(tup)] = v
        return pairs_mapping,pairs_score
      
                
if __name__ == "__main__":
    
    dg = Datasets_Generation()
    lds = dg.lsp_datasets()
    # print(lds)
    dme,dmd = dg.matrys_entities()

    # dlm = Dataset_level_matching()
    # top_entities,top_pairs_score=dlm.dataset_top(
    #                                             datasets=lds,
    #                                             entities=dme,
    #                                             k=1,
    #                                             method='edit_distance'
    #                                             )
    # top3_entities,top3_pairs_score = dlm.load_top_result(method='edit_distance',k=3)
    # print()

    alm = Attribute_level_matching()
    # matcher = alm._load_matcher(method='fasttext-300')
    # r = alm._dataset_mapping(d_name = '2020_Data_Sample',
    #                          dataset = lds['2020_Data_Sample'],
    #                          e_name='Questionnaire',
    #                          entity=dme['Questionnaire'],
    #                          matcher=matcher,
    #                          )
    # print(r)

    # pairs_mapping, pairs_score = alm.dataset2entity(
    #                             datasets=lds,
    #                             entities=dme,
    #                             top_entities=top3_entities,
    #                             k = 3,
    #                             method = 'edit_distance'
    #                             )
    pairs_mapping,pairs_score = alm.load_mapping_result(method='jaccard',k=1)
    print(pairs_mapping)
    # print(pairs_score)

    
    

            