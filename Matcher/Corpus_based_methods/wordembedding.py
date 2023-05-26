from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import gensim.downloader as api
class Model(object):

    def _dataset_info(self):
        info = api.info() 
        corpora_name = info['corpora'].keys()
        models_name = list(info['models'].keys())
        info = {'corpus':corpora_name,'models':models_name}
        return info

    def _api_model_save(self,model_name=None):
        '''
        models:
        ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 
        'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 
        'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', 
        '__testing_word2vec-matrix-synopsis']
        '''
        save_path = r'D:\path\Wordembedding_models'+r'\{}'.format(model_name)
        model = api.load(model_name)
        model.save(save_path)

    def model_load(self,model_name=None):
        '''
        models:
        ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 
        'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 
        'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200','glove.42B.300d'
        '__testing_word2vec-matrix-synopsis']
        '''
        load_path = r'D:\path\Wordembedding_models'+r'\{}'.format(model_name)
        model = KeyedVectors.load(load_path)
        return model


    # # only for the model 'glove.42.300d'
    # def _model_created(self,glove_name):
    #     '''
    #     glove_name:
    #     glove.42B.300d
    #     '''
    #     dir_path = r'E:\Guanchen_Pan\MasterThesis\Wordembedding_models'
    #     glove_input_file = dir_path + r'\{}.txt'.format(glove_name)
    #     word2vec_output_file = dir_path + r'\{}.word2vec'.format(glove_name)
    #     glove2word2vec(glove_input_file, word2vec_output_file)
        
    
    # def _model_save(self,glove_name):
    #     '''
    #     glove_name:
    #     glove.42B.300d
    #     '''
    #     dir_path = r'E:\Guanchen_Pan\MasterThesis\Wordembedding_models'
    #     load_fpath = dir_path + r'\{}.word2vec'.format(glove_name)
    #     save_fpath = dir_path + r'\{}'.format(glove_name)
    #     model = KeyedVectors.load_word2vec_format(load_fpath, binary=False)
    #     model.save(save_fpath)
    #     return model

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import gensim.downloader as api
import sys
sys.path.append((r"."))
from Data_Preprocessing.data_preprocessing import Data_preprocessing

class Word2vec(object):
    def __init__(self):
        self.dp = Data_preprocessing()
        m = Model()
        self.model = m.model_load(model_name='word2vec-google-news-300')
    def _(self,str_a=None,str_b=None):
        str_a = self.dp.phrase_preprocessing(phrase=str_a,art='lsp') # lsp data
        str_b = self.dp.phrase_preprocessing(phrase=str_b,art='matrys') # matrys data
        k=1
        if k==1:
            try:
                result = self.model.similarity(str_a.lower(),str_b.lower())
            except KeyError:
                k = 2
        if k ==2:
            try:
                temp = []
                tokens_a = self.dp.tokenize(str_a.lower())
                tokens_b = self.dp.tokenize(str_b.lower())
                for i in tokens_a:
                    for j in tokens_b:
                        try:                     
                            r = self.model.similarity(i, j)
                        except:
                            r = 0
                            k = 3
                        temp.append(r)
                temp_array = np.array(temp)
                result = np.mean(temp_array)
            except:
                result = 0
                k = 3
        if k == 3:
            try:
                str_b = self.dp.phrase_symspell(str_b)
                temp = []
                tokens_a = self.dp.tokenize(str_a.lower())
                tokens_b = self.dp.tokenize(str_b.lower())
                for i in tokens_a:
                    for j in tokens_b:
                        try:                     
                            r = self.model.similarity(i, j)
                        except:
                            r = 0
                        temp.append(r)
                temp_array = np.array(temp)
                temp_result = np.mean(temp_array)
                result = max(temp_result,result)
            except:
                result = 0  
        return result

    def sim(self,sentences_a=[],sentences_b=[]):
        cosine_scores = []
        for i in sentences_a:
            temp = []
            for j in sentences_b:    
                s = self._(str_a=i,str_b=j)
                temp.append(s)
            cosine_scores.append(temp)
        cosine_scores = torch.tensor(cosine_scores)
        return cosine_scores


class Glove(object):

    def __init__(self,model_name='glove-wiki-gigaword-50'):
        '''
        glove-wiki-gigaword-300
        glove-wiki-gigaword-50

        '''
        m = Model()
        self.dp = Data_preprocessing()
        if model_name == 'glove-wiki-gigaword-50':
            self.model = m.model_load(model_name=model_name)
        elif model_name == 'glove-wiki-gigaword-300':
            self.model = m.model_load(model_name=model_name)

    def _(self,str_a=None,str_b=None):
        str_a = self.dp.phrase_preprocessing(phrase=str_a,art='lsp') # lsp data
        str_b = self.dp.phrase_preprocessing(phrase=str_b,art='matrys') # matrys data
        k=1
        if k==1:
            try:
                result = self.model.similarity(str_a.lower(),str_b.lower())
            except KeyError:
                k = 2
        if k ==2:
            try:
                temp = []
                tokens_a = self.dp.tokenize(str_a.lower())
                tokens_b = self.dp.tokenize(str_b.lower())
                for i in tokens_a:
                    for j in tokens_b:
                        try:                     
                            r = self.model.similarity(i, j)
                        except:
                            r = 0
                            k = 3
                        temp.append(r)
                temp_array = np.array(temp)
                result = np.mean(temp_array)
            except:
                result = 0
                k = 3
        if k == 3:
            try:
                str_l = self.dp.phrase_symspell(str_l)
                temp = []
                tokens_a = self.dp.tokenize(str_a.lower())
                tokens_b = self.dp.tokenize(str_b.lower())
                for i in tokens_a:
                    for j in tokens_b:
                        try:                     
                            r = self.model.similarity(i, j)
                        except:
                            r = 0
                        temp.append(r)
                temp_array = np.array(temp)
                temp_result = np.mean(temp_array)
                result = max(temp_result,result)
            except:
                result = 0   
        return result
    
    def sim(self,sentences_a=[],sentences_b=[]):
        cosine_scores = []
        for i in sentences_a:
            temp = []
            for j in sentences_b:    
                s = self._(str_a=i,str_b=j)
                temp.append(s)
            cosine_scores.append(temp)
        cosine_scores = torch.tensor(cosine_scores)
        return cosine_scores
        

class Fast_text(object):

    def __init__(self):
        '''
        'fasttext-wiki-news-subwords-300'
        '''
        m = Model()
        self.dp = Data_preprocessing()
        self.model = m.model_load(model_name='fasttext-wiki-news-subwords-300')
        

    def _(self,str_a=None,str_b=None):
        str_a = self.dp.phrase_preprocessing(phrase=str_a,art='lsp') # lsp data
        str_b = self.dp.phrase_preprocessing(phrase=str_b,art='matrys') # matrys data
        k=1
        if k==1:
            try:
                result = self.model.similarity(str_a.lower(),str_b.lower())
            except KeyError:
                k = 2
        if k ==2:
            try:
                temp = []
                tokens_a = self.dp.tokenize(str_a.lower())
                tokens_b = self.dp.tokenize(str_b.lower())
                for i in tokens_a:
                    for j in tokens_b:
                        try:                     
                            r = self.model.similarity(i, j)
                        except:
                            r = 0
                            k = 3
                        temp.append(r)
                temp_array = np.array(temp)
                result = np.mean(temp_array)
            except:
                result = 0
                k = 3
        if k == 3:
            try:
                str_l = self.dp.phrase_symspell(str_l)
                temp = []
                tokens_a = self.dp.tokenize(str_a.lower())
                tokens_b = self.dp.tokenize(str_b.lower())
                for i in tokens_a:
                    for j in tokens_b:
                        try:                     
                            r = self.model.similarity(i, j)
                        except:
                            r = 0
                        temp.append(r)
                temp_array = np.array(temp)
                temp_result = np.mean(temp_array)
                result = max(temp_result,result)
            except:
                result = 0   
        return result
    def sim(self,sentences_a=[],sentences_b=[]):
        cosine_scores = []
        for i in sentences_a:
            temp = []
            for j in sentences_b:    
                s = self._(str_a=i,str_b=j)
                temp.append(s)
            cosine_scores.append(temp)
        cosine_scores = torch.tensor(cosine_scores)
        return cosine_scores


from transformers import logging
logging.set_verbosity_error()
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class Bert(object):
    def __init__(self, model_name = 'bert-base-uncased'):
        '''
        'bert-base-uncased'
        '''
        self.dp = Data_preprocessing()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,output_hidden_states=True)

    def get_embeddings(self,text):
        tokens=self.tokenizer.encode_plus(text,
                              max_length=20,
                              padding='max_length',
                              truncation=True
                              )
        output=self.model(torch.tensor(tokens.input_ids).unsqueeze(0),
                    attention_mask=torch.tensor(tokens.attention_mask).unsqueeze(0)).hidden_states[-1]
        return torch.mean(output,axis=1).detach().numpy()

    #calculate similarity
    def _(self,str_a=None,str_b=None):
        if str_b == None:
            return 0
        str_a = self.dp.lsp_attribute(str_a) # lsp data
        str_b = self.dp.matrys_attribute(str_b) # matrys data
        out_a=self.get_embeddings(str_a)#create embeddings of text
        out_b=self.get_embeddings(str_b)#create embeddings of text
        s = cosine_similarity(out_a,out_b)[0][0]
        return round(float(s),3)

    def sim(self,sentences_a=[],sentences_b=[]):
        cosine_scores = []
        for i in sentences_a:
            temp = []
            for j in sentences_b:    
                s = self._(str_a=i,str_b=j)
                temp.append(s)
            cosine_scores.append(temp)
        cosine_scores = torch.tensor(cosine_scores)
        return cosine_scores
        
    


class Sbert(object):
    def __init__(self, model_name = 'sentence-transformers/all-MiniLM-L6-v2'):
        '''
        'sentence-transformers/all-MiniLM-L6-v2'
        'sentence-transformers/bert-base-nli-mean-tokens'
        '''
        self.dp = Data_preprocessing()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        

    def _embedding(self,sentences):
        tokens = {'input_ids':[], 'attention_mask':[]}
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                                sent,
                                add_special_tokens=True,
                                max_length=20,
                                truncation=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt'
                            )
            tokens['input_ids'].append(encoded_dict['input_ids'])
            tokens['attention_mask'].append(encoded_dict['attention_mask'])
        tokens['input_ids'] = torch.cat(tokens['input_ids'], dim=0)
        tokens['attention_mask'] = torch.cat(tokens['attention_mask'], dim=0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state
        attention = tokens['attention_mask']
        mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
        mask_embeddings = embeddings * mask
        summed = torch.sum(mask_embeddings,1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled

    def _(self,str_a=None,str_b=None):
        if str_b == None:
            return 0
        str_a = self.dp.lsp_attribute(str_a) # lsp data
        str_b = self.dp.matrys_attribute(str_b) # matrys data
        phrases = [str_a.lower(),str_b.lower()]
        embedding = self._embedding(phrases)
        s = torch.cosine_similarity(embedding[0],embedding[1],dim=0)
        return round(float(s),3)


    def sim(self,sentences_a=[],sentences_b=[]):
        X = []
        for a in sentences_a:
            str_a = self.dp.lsp_attribute(a) # lsp data

            X.append(str_a.lower())
        Y = []
        for b in sentences_b:
            str_b = self.dp.matrys_attribute(b) # matrys
            Y.append(str_b.lower())
        embedding1 = self._embedding(X)
        embedding2 = self._embedding(Y)
        cosine_scores = []
        for i in embedding1:
            temp = []
            for j in embedding2:
                s = torch.cosine_similarity(i,j,dim=0)
                temp.append(s)
            cosine_scores.append(temp)
        cosine_scores = torch.tensor(cosine_scores)
        return cosine_scores

        

if __name__ == '__main__':

    # a = ['The cat sits outside',
    #          'A man is playing guitar',
    #          'The new movie is awesome']
    # b = ['The dog plays in the garden',
    #           'A woman watches TV',
    #           'The new movie is so great']
    a = [
        'dateCreated',
        'DATE',
        'LOCATION',
        'value',
        'power',
        'ENERGY_SOURCE',
        ]
    b = [
    'dateCreated',
    'DATE',
    'LOCATION',
    'value',
    'power',
    'ENERGY_SOURCE',
    ]
    # wv = Word2vec()
    # score = wv.sim(sentences_a=a,sentences_b=b)

    # g = Glove()
    # score = g.sim(sentences_a=a,sentences_b=b)

    # ft = Fast_text()
    # score = ft.sim(sentences_a=a,sentences_b=b)

    sbert = Bert(model_name = 'bert-base-uncased')
    score = sbert.sim(sentences_a=a,sentences_b=b)
    # score = sbert._(str_a='DATE',str_b="dateCreated")
    print(score)

    sbert = Sbert(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    score = sbert.sim(sentences_a=a,sentences_b=b)
    # score = sbert._(str_a='DATE',str_b="dateCreated")
    
    # score = bert.sim(sentences_a=a,sentences_b=b)
    print(score)
    pass