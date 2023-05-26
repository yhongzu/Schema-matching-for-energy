
from nltk.metrics import edit_distance as ed
import sys
import torch
sys.path.append((r"."))
from Data_Preprocessing.data_preprocessing import Data_preprocessing

class Edit_distance(object):
    '''
        Edit_Distance_levenshtein
        s1 = "abc", s2 = "def"
        Formula:
                | 0                                           i = 0, j = 0
                | j                                           i = 0, j > 0
        d[i,j] = | i                                           i > 0, j = 0
                | min(d[i,j-1]+1, d[i-1,j]+1, d[i-1,j-1])    s1(i) = s2(j)
                | min(d[i,j-1]+1, d[i-1,j]+1, d[i-1,j-1]+1)  s1(i) ≠ s2(j)
        定义二维数组[4][4]:
             d e f             d e f
          |x|x|x|x|        |0|1|2|3|
        a |x|x|x|x|  =>  a |1|1|2|3|  => 编辑距离d = [4][4] = 3
        b |x|x|x|x|      b |2|2|2|3|
        c |x|x|x|x|      c |3|3|3|3|
    '''
    def __init__(self):
        self.dp = Data_preprocessing()

    def _(self,str_a=None,str_b=None):
        str_a = self.dp.phrase_preprocessing(phrase=str_a,art='lsp')
        str_b = self.dp.phrase_preprocessing(phrase=str_b,art='matrys')
        a = len(str_a)
        b = len(str_b)
        levD = ed(str_a.lower(), str_b.lower())
        return round((1-(levD/(max(a,b)))),3)
    def sim(self,sentences_a=[],sentences_b=[]):
        '''
        sentence_a and sentence_b represent the a entire dataset and entity.
        '''
        cosine_scores = []
        for i in sentences_a:
            temp = []
            for j in sentences_b:    
                s = self._(str_a=i,str_b=j)
                temp.append(s)
            cosine_scores.append(temp)
        cosine_scores = torch.tensor(cosine_scores)
        return cosine_scores

# only for String
class Jaccard(object):
    def __init__(self):
        self.dp = Data_preprocessing()
    
    def _(self,str_a=None,str_b=None):
        str_a = self.dp.phrase_preprocessing(phrase=str_a,art='lsp')
        str_b = self.dp.phrase_preprocessing(phrase=str_b,art='matrys')
        list_a = list(str_a.lower())
        list_b = list(str_b.lower())
        a = len(list_a)
        b = len(list_b)
        intersection = []
        for i in list_a:
            for j in list_b:
                if i == j:
                    intersection.append(j)     
                    list_b.remove(j)
                    break
        union = a+b-len(intersection)
        return round(len(intersection)/union,3)
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


if __name__ == '__main__':
    j = Jaccard()
    a = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome']
    b = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']
    c = j.sim(sentences_a=a,sentences_b=b)
    print(c)

    pass