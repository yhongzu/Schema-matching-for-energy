import pkg_resources
import nltk
import sys
from symspellpy.symspellpy import SymSpell
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import copy
sys.path.append(r".")
from Data_Reading.lsp_dataset import Lsp_dataset
from Data_Reading.matrys_data_model import Data_model


class Data_preprocessing(object):
    def __init__(self):
        # Spelling Correction
        self.sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


    # get pos
    def _get_wordnet_pos(self,tag):
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
        
    
    # tokenize
    def tokenize(self,phrase):      
        tokens = nltk.word_tokenize(phrase)
        return tokens


    # stemming
    def stemming(self,tokens_list):    
        tagged_sent = nltk.pos_tag(tokens_list)
        lemmas = []
        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = self._get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas.append(wnl.lemmatize(tag[0],pos=wordnet_pos))
        result = ' '.join(lemmas)
        return result

    def phrase_symspell(self,str):
        # the words without space to phrase
        phrase = self.sym_spell.word_segmentation(str)
        correct_phrase = phrase.corrected_string
        return correct_phrase

    
    def phrase_preprocessing(self,art='lsp',phrase=None):
        '''
        art('lsp','matrys')
        '''
        if art=='lsp':
            temp = self.lsp_attribute(phrase)
        elif art == 'matrys':
            temp = self.matrys_attribute(phrase)
        temp = self.tokenize(temp.lower())
        temp = self.stemming(temp)
        return temp


    # lsp_Attributes preprocessing
    def lsp_attribute(self,s):
        s = s.replace('_',' ') 
        s = s.replace('-',' ')
        s = s.replace('(',' ')
        s = s.replace(')',' ')
        if ' ' in s:
            return s
        elif len(s)<2:
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
            return s


    # matrys_attributes preprocessing
    def matrys_attribute(self,s):
        s = s.replace('_',' ') 
        s = s.replace('-',' ')
        s = s.replace('(',' ')
        s = s.replace(')',' ')
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


    

if __name__ == "__main__":
    dp = Data_preprocessing()
    # t = dp.phrase_preprocessing(art='matrys',phrase="Organization_Name")
    print(dp.lsp_attribute('USO_EDIFICIO'))
    # print(t)

    