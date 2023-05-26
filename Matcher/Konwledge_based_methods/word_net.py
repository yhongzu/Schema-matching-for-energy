from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
import nltk
import torch
import numpy as np
import sys
sys.path.append((r"."))
from Data_Preprocessing.data_preprocessing import Data_preprocessing
from Data_Reading.matrys_data_model import Data_model
from Data_Reading.lsp_dataset import Lsp_dataset
class Word_net(object):

    def __init__(self,art='path_sim'):
        self.dp = Data_preprocessing()
        if art == 'path_sim':
            self.art = art
        elif art == 'wup_sim':
            self.art = art
        

    def _path_sim(self,str_a=None,str_b=None):

        synsets1 = wordnet.synsets(str_a)
        synsets2 = wordnet.synsets(str_b)
        path_sim = 0
        for tempword1 in synsets1:
            for tempword2 in synsets2:
                if tempword1.pos() == tempword2.pos():
                    try:
                        path_sim = max(path_sim,tempword1.path_similarity(tempword2))
                    except Exception as e:
                        print(tempword1, tempword2)
                        print("path: "+str(e))
        return round(path_sim,3)

    def _wup_sim(self,str_a=None,str_b=None):
        synsets1 = wordnet.synsets(str_a)
        synsets2 = wordnet.synsets(str_b)
        wup_sim = 0
        for tempword1 in synsets1:
            for tempword2 in synsets2:
                if tempword1.pos() == tempword2.pos():
                    try:
                        wup_sim = max(wup_sim,tempword1.wup_similarity(tempword2))
                    except Exception as e:
                        print(tempword1, tempword2)
                        print("wup: "+str(e))
        return round(wup_sim,3)


    def _(self,str_a=None,str_b=None):
        str_a = self.dp.phrase_preprocessing(phrase=str_a,art='lsp') # lsp data
        str_b = self.dp.phrase_preprocessing(phrase=str_b,art='matrys') # matrys data
        wn_temp = []
        list_a = nltk.word_tokenize(str_a.lower())
        list_b = nltk.word_tokenize(str_b.lower())
        for stemm_a  in list_a:
            temp = []
            for stemm_b in list_b:
                if self.art=='path_sim':
                    wn_result = self._path_sim(str_a=stemm_a,str_b=stemm_b)
                    temp.append(wn_result)
                elif self.art=='wup_sim':
                    wn_result = self._path_sim(str_a=stemm_a,str_b=stemm_b)
                    wn_temp.append(wn_result)
            wn_temp.append(max(temp))
        wn_array = np.array(wn_temp)
        wn_value = np.mean(wn_array)
        return round(wn_value,3)

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


'''
    def wordnetSim(str_a,str_b):
        brown_ic = wordnet_ic.ic('ic-brown.dat')
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
        synsets1=wn.synsets(str_a)
        synsets2=wn.synsets(str_b)

        path_sim = -1
        wup_sim = -1

        res_sim = -1
        jcn_sim = -1
        lin_sim = -1
        for tmpword1 in synsets1:
            for tmpword2 in synsets2:
                print(tmpword1.pos(),tmpword2.pos())
                if tmpword1.pos() == tmpword2.pos():
                    try:
                        path_sim=max(path_sim,tmpword1.path_similarity(tmpword2))
                        
                    except Exception as e:
                        print(tmpword1, tmpword2)
                        print("path: "+str(e))

                    try:
                        wup_sim=max(wup_sim,tmpword1.wup_similarity(tmpword2))
                    except Exception as e:
                        print (tmpword1, tmpword2)
                        print( "wup: "+str(e))
                    try:
                        res_sim=max(res_sim,tmpword1.res_similarity(tmpword2,brown_ic))
                    except Exception as e:
                        print( tmpword1, tmpword2)
                        print ("res: "+str(e))

                    try:
                        jcn_sim=max(jcn_sim,tmpword1.jcn_similarity(tmpword2,brown_ic))
                    except Exception as e:
                        print (tmpword1, tmpword2)
                        print ("jcn: "+str(e))

                    try:
                        lin_sim=max(lin_sim,tmpword1.lin_similarity(tmpword2,semcor_ic))
                    except Exception as e:
                        print (tmpword1, tmpword2)
                        print ("lin: "+str(e))

        path_result=(str_a, str_b, path_sim)
        wup_result=(str_a, str_b, wup_sim)
        res_result=(str_a, str_b, res_sim)
        jcn_result=(str_a, str_b, jcn_sim)
        lin_result=(str_a, str_b, lin_sim)

        results=[path_result, wup_result, res_result, jcn_result, lin_result]
        return path_sim
    '''
    
if __name__ == '__main__':
    wn = Word_net(art='path_sim')
    a = ['cat dog',
             'A man is playing guitar',
             'The new movie is awesome']
    b = ['dog cat',
              'A man is playing guitar',
              'The new movie is so great']
    a = wn.sim(sentences_a=a,sentences_b=b)
    print(a)
    pass
    

    
