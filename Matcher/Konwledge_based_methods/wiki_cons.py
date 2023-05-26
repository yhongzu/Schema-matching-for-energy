import wikipedia
import nltk
import torch
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import sys
sys.path.append((r"."))
from Data_Preprocessing.data_preprocessing import Data_preprocessing
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def stemming(tokens_list):
    # stemming
    tagged_sent = nltk.pos_tag(tokens_list)
    lemmas = []
    wnl = WordNetLemmatizer()
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas.append(wnl.lemmatize(tag[0],pos=wordnet_pos))
    return lemmas

class Wiki_cons(object):
    def __init__(self,art='con_sim4'):
        self.dp = Data_preprocessing()
        if art == 'con_sim4':
            self.art = art

            
    def _(self,str_a=None,str_b=None):
        str_a = self.dp.phrase_preprocessing(phrase=str_a,art='lsp') # lsp data
        str_b = self.dp.phrase_preprocessing(phrase=str_b,art='matrys') # matrys data
        self._features(str_a=str_a,str_b=str_b)
        if self.art=='con_sim1':
            sim = self._cons_sim1()
        if self.art=='con_sim2':
            sim = self._cons_sim2()
        if self.art=='con_sim4':
            sim = self._cons_sim4()
        return round(sim,3)

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


    def _features(self,str_a=None,str_b=None):
        # Con str_a
        try:
            self.synonyms_a = set(wikipedia.search(str_a))
        except:
            self.synonyms_a = set(nltk.word_tokenize(str_a))

        try:
            page_a = wikipedia.WikipediaPage(str_a)
            self.anchors_a = set(page_a.links)
            self.categories_a = set(page_a.categories)
            gloss_a = page_a.summary
            tokens_a = nltk.word_tokenize(gloss_a)
            self.gloss_a = set(stemming(tokens_a))
        except wikipedia.exceptions.DisambiguationError as e:
            self.anchors_a = set(e.options)
            self.categories_a = set(e.options)
            self.gloss_a = set(e.options)
        except KeyError as e:
            self.anchors_a =set(nltk.word_tokenize(str_a))
            self.categories_a = set(nltk.word_tokenize(str_a))
            self.gloss_a = set(nltk.word_tokenize(str_a))
        except wikipedia.exceptions.PageError as e:
            self.anchors_a = set([])
            self.categories_a = set([])
            self.gloss_a = set([])


        #Con str_b 
        try:     
            self.synonyms_b = set(wikipedia.search(str_b))
        except:
            self.synonyms_b = set(nltk.word_tokenize(str_b))
        try:
            page_b = wikipedia.WikipediaPage(str_b)
            self.anchors_b = set(page_b.links)
            self.categories_b = set(page_b.categories)
            gloss_b = page_b.summary
            tokens_b = nltk.word_tokenize(gloss_b)
            self.gloss_b = set(stemming(tokens_b))
        except wikipedia.exceptions.DisambiguationError as e:
            self.anchors_b = set(e.options)
            self.categories_b = set(e.options)
            self.gloss_b = set(e.options)
        except KeyError as e:
            self.anchors_b = set(nltk.word_tokenize(str_b))
            self.categories_b = set(nltk.word_tokenize(str_b))
            self.gloss_b = set(nltk.word_tokenize(str_b))
        except wikipedia.exceptions.PageError as e:
            self.anchors_b = set([])
            self.categories_b = set([])
            self.gloss_b = set([])
        
    
    def _x_sim(self,set_a,set_b):
        intersections = len(set_a.intersection(set_b))
        if intersections == 0:
            return 0
        else:
            unions = len(set_a.union(set_b))
            return round(intersections/unions,3)
        
    def _re_sim(self,set_a,set_b):
        intersections = len(set_a.intersection(set_b))
        if intersections == 0:
            return 0
        else:
            a2b_diff = len(set_a.difference(set_b))
            b2a_diff = len(set_b.difference(set_a))
            a = 0.5 # default
            return round(intersections/(intersections+a*a2b_diff+(1-a)*b2a_diff),3)
    

    def _cons_sim1(self):
        # X-sim
        s_synonyms = self._x_sim(self.synonyms_a,self.synonyms_b)
        s_gloss = self._x_sim(self.gloss_a,self.gloss_b)
        s_anchors = self._x_sim(self.anchors_a,self.anchors_b)
        s_categories = self._x_sim(self.categories_a,self.categories_b)
        sim = (s_synonyms+s_gloss+s_anchors+s_categories)/4
        return round(sim,3)
    def _cons_sim2(self):
        # RE-sim
        s_synonyms = self._re_sim(self.synonyms_a,self.synonyms_b)
        s_gloss = self._re_sim(self.gloss_a,self.gloss_b)
        s_anchors = self._re_sim(self.anchors_a,self.anchors_b)
        s_categories = self._re_sim(self.categories_a,self.categories_b)
        sim = (s_synonyms+s_gloss+s_anchors+s_categories)/4
        return round(sim,3)
    def _cons_sim4(self):
        s_synonyms = self._re_sim(self.synonyms_a,self.synonyms_b)
        s_gloss = self._re_sim(self.gloss_a,self.gloss_b)
        s_anchors = self._re_sim(self.anchors_a,self.anchors_b)
        s_categories = self._re_sim(self.categories_a,self.categories_b)
        sim = max(s_synonyms,s_gloss,s_anchors,s_categories)
        return round(sim,3)


if __name__ == "__main__":
    wiki = Wiki_cons()
    a = ['cat',
             'man',
             'movie']
    b = ['dog',
              'woman',
              'movie']
    a = wiki.sim(sentences_a=a,sentences_b=b)
    print(a)
    pass
    
