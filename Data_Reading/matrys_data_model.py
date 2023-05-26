import pandas as pd 
import json
import nltk
from keybert import KeyBERT
import wikipedia
# Read data set from matrys data model,every entity is one data set 
class Data_model(object):
    def __init__(self):
        # Data stored path
        self.read_path = r"MATRYCS Data\MATRYCS_data_model_v11.xlsx"
        # Data sets stored path
        self.save_path = r'Data_Reading\json_files'

    def _read_matrys_data(self):
        data_model = {}
        catagetries = pd.read_excel(self.read_path, sheet_name=None)
        for c_name in catagetries:
            df_c = pd.read_excel(self.read_path, sheet_name=c_name)
            df_c = pd.DataFrame(df_c[['Entity','Entity Attributes','Attribute description']])
            df_c.dropna(axis='index', how='all', inplace=True)
            df_c['Entity'].fillna(axis=0,method='ffill',inplace=True)    
            # reset the index of df_c from 0 beginning
            df_c = df_c.reset_index(drop=True)
            # how many entities in a category
            entities=[]
            for i in range(len(df_c.index)): 
                e_name = df_c.loc[i,'Entity']   
                entities.append(e_name)
            entities = list({}.fromkeys(entities).keys())
            # each entity of a category as a dataset
            category = {}
            for e_name in entities:
                df_e = []
                for i in range(len(df_c.index)):
                    if df_c.loc[i,'Entity'] == e_name:
                        df_e.append(df_c.loc[i,['Entity Attributes','Attribute description']])
                df_e = pd.DataFrame(df_e)
                df_e =df_e.reset_index(drop=True)
                df_e.columns = ['Entity Attributes','Attribute description']
                category[e_name] = df_e
            data_model[c_name] = category
        file_name = 'matrys_data_model'
        self._save_matrys_file(data_model, self.save_path,file_name)
        # with pd.ExcelWriter(r"Files_read\{0}.xlsx".format(file_name)) as writer:
        #     for name, sheet_file in data_model.items():
        #         sheet_file.to_excel(writer,sheet_name=name,index=False)
        return data_model


    def _save_matrys_file(self,file,save_path,file_name):
        file_json = {}
        for c_name,entities in file.items():
            category = {}
            for e_name,df_e in entities.items():
                e = df_e.to_json()
                e = json.loads(e) 
                category[e_name] = e
            file_json[c_name] = category
        with open(save_path+r"\{0}.json".format(file_name),"w") as f:
            f.write(json.dumps(file_json,indent=4))


    def load_data_model(self,data_model='matrys_data_model'):
        '''
        data_model:
            'matrys_data_model' means Loading data model with incomplete description
            'original' means Loading a data model with a complete description but not summarized
            'summarized' means Loading a data model with a complete description and already summarized
        '''
        load_path = self.save_path+r"\{}.json".format(data_model)
        fpath = open(load_path,'r')
        file = json.load(fpath)
        data_model = {}
        for c_name,entities in file.items():
            category = {}
            for e_name,e in entities.items():
                df_e = {}
                for i, j in e.items():
                    temp = {}
                    for k in j.keys():
                        temp[eval(k)] = j.get(k)
                    df_e[i] = temp
                df_e = pd.DataFrame(df_e)
                category[e_name] = df_e
            data_model[c_name] = category
        return data_model  


# matrys_data_model description summary
class _Summary(object):
    def __init__(self):
        dm = Data_model()
        self.matrys_data = dm.load_data_model()
        self.kmodel = KeyBERT('distilbert-base-nli-mean-tokens')
        self.save_path = r"Data_Reading\json_files"
    def _method(self,sentence, num):
        keyword = self.kmodel.extract_keywords(sentence, keyphrase_ngram_range=(1, num))
        try:
            t = keyword[0][0]
        except IndexError :
            t = sentence
        return t
    def _handle_m(self,s):
        s = s.replace('_',' ') 
        s = s.replace('-',' ')
        if s == 'uVIndexMax':
            s = 'uv Index Max'
            return s
        if s == 'consumptionORD':
            s = 'consumption ORD'
            return s 
        if ' ' in s:
            return s
        if 'CO2' in s:
            s = s.replace("CO2",'Cotwo')
        if len(s)<2:
            return s   
        else:     
            for i, element in enumerate(s):
                if i == 0:
                    if s[i] == element.upper() and s[i+1] == s[i+1].upper():
                        return s
                elif i > 0 and i < len(s)-1:
                    if s[i] == element.upper() and (s[i-1] == s[i-1].upper() or  s[i+1] == s[i+1].upper()):
                        return s
            s_list = list(s)
            temp = 0
            for i, element in enumerate(s):
                if i != 0 and element == element.upper():               
                    s_list.insert(i+temp,' ')
                    temp +=1   
            s = "".join(s_list)
            if 'Cotwo' in s:
                s = s.replace('Cotwo','CO2')
            return s


    def _description(self,art='original'):
        '''
        art:
            original means Descriptive information complemented by Wikipedia is not summarized.
            summarized means Descriptive information complemented by Wikipedia is  summarized.
        '''
        data_model = {}
        # c is category
        for c_name, entities in self.matrys_data.items():
            category = {}
            for e_name,df_e in entities.items():
                for i in df_e.index:
                    handled_attri = self._handle_m(df_e.loc[i,'Entity Attributes'])
                    attri_list = nltk.word_tokenize(handled_attri)
                    if df_e.loc[i,'Attribute description'] == None:
                        try:
                            page_attri = wikipedia.WikipediaPage(handled_attri)
                            gloss_attri = page_attri.summary
                        except:
                            gloss_attri = handled_attri
                        df_e.loc[i,'Attribute description'] = gloss_attri
                    # the length of the attributes phases
                    if art=='summarized':
                        num = len(attri_list)
                        df_e.loc[i,'Attribute description'] = self._method(df_e.loc[i,'Attribute description'],num)
                category[e_name] = df_e
            data_model[c_name] = category
        if art=='summarized':
            file_name = art
        elif art=='original':
            file_name = art
        self._save_matrys_file(data_model, self.save_path,file_name)
        return data_model  


    def _save_matrys_file(self,file,save_path,file_name):
        file_json = {}
        for c_name,entities in file.items():
            category = {}
            for e_name,df_e in entities.items():
                e = df_e.to_json()
                e = json.loads(e) 
                category[e_name] = e
            file_json[c_name] = category
        with open(save_path+r"\{0}.json".format(file_name),"w") as f:
            f.write(json.dumps(file_json,indent=4))

if __name__ == "__main__":
    dm =Data_model()
    matrys_data_model = dm._read_matrys_data()
    print(matrys_data_model)
    # print(matrys_data_model)
    # s = _Summary()
    # sd = s._description(art='summaried')
    # print(sd)

