import sys
import pandas as pd
import numpy as np
sys.path.append((r"."))
from Experiment.evaluation import Evaluation
from Methodology.matching import Datasets_Generation,Dataset_level_matching,Attribute_level_matching

class Main(object):

    # Dataset level: After Experiments implementation, Then the result should be evaluated
    def output_dlm_evaluatation(self,method='sbert',k_values=[]):
        e = Evaluation()
        dlm = Dataset_level_matching()
        output= []
        for k in k_values:
            top_entities,top_pairs_score = dlm.load_top_result(method=method,k=k)
            result = e.dataset_level_evaluation(top_pairs_score=top_pairs_score,top_entities=top_entities)
            df_result = pd.DataFrame.from_dict(result,orient='index',columns=[r'k={0}'.format(k)])
            output.append(df_result)
        df_output = pd.concat(output,axis=1,join='outer')
        with pd.ExcelWriter(r'Experiment\main_output_files\dlm_output_{0}.xlsx'.format(method)) as writer:
            df_output.to_excel(writer,index=True)
        return df_output

    # Attribute level: After Experiments implementation, Then the result should be evaluated
    def output_alm_evaluation(self,k=3,methods=[]):
        e = Evaluation()
        alm = Attribute_level_matching()
        dlm = Dataset_level_matching()
        output = []
        top_entities,top_pairs_score = dlm.load_top_result(method='sbert',k=k)
        for method in methods:
            pairs_mapping,pairs_score = alm.load_mapping_result(k=k,method=method)
            # output single method mapping result file
            with pd.ExcelWriter(r'Experiment\single_method_mapping_files\{0}_mapping_result.xlsx'.format(method)) as writer:
                l = [tup[1] for tup in pairs_mapping.keys()]
                l = list({}.fromkeys(l).keys())
                for i in l:
                    df = []
                    for tup, mapping_file in pairs_mapping.items():
                        if tup[1] == i:
                            df.append(mapping_file)
                    df = pd.concat(df,axis=1,join='outer')
                    df = df.loc[:,df.columns.duplicated()==False]
                    df.to_excel(writer,sheet_name=i,index=False)

            e_result = e.attribute_level_evaluation(top_entities=top_entities,pairs_mapping=pairs_mapping)
            df_e_result = pd.DataFrame.from_dict(e_result,orient='index',columns=[method])
            output.append(df_e_result)

        df_output = pd.concat(output,axis=1,join='outer')
        with pd.ExcelWriter(r'Experiment\main_output_files\alm_top{0}_output_{1}.xlsx'.format(k,methods)) as writer:
            df_output.to_excel(writer,index=True)
        return df_output

    # experiments implentation for dataset level matching
    def implementation_dlm(self,methods=['edit_distance'],k_values=[3]):

        dg = Datasets_Generation()
        lds = dg.lsp_datasets()
        dme,dmd = dg.matrys_entities()
        dlm = Dataset_level_matching()
        for m in methods:
            for k in k_values:
                print(r'dlm_{0}_top{1} starts.'.format(m,k))
                start = perf_counter()
                top_entities,top_pairs_score=dlm.dataset_top(
                                                            datasets=lds,
                                                            entities=dme,
                                                            k=k,
                                                            method=m
                                                            )
                finish = perf_counter()
                t = (finish-start)/60
                info = r'dlm_{0}_top{1} completes and Time consuming (min) :{2} min.'.format(m,k,t)
                print(info)
                fh = open(r"Experiment\time_info_files\dlm_{0}_top{1}.txt".format(m,k), 'w')
                fh.write(info)
                fh.close()

    # Experiment implentation for Attribute level matching
    def implmentation_alm(self,methods=['jaccard'],k=3):
        '''method=<'edit_distance','jaccard','dice','word_net_pathsim','wiki_cons','word2vec',
                   'glove-50','fasttext-300','bert-base-uncased','sbert','ALmatcher'>
        '''
        dg = Datasets_Generation()
        lds = dg.lsp_datasets()
        dme,dmd = dg.matrys_entities()
        dlm = Dataset_level_matching()
        alm = Attribute_level_matching()
        for method in methods:
            print(r'alm_top{0}_{1}_ starts.'.format(k,method))
            start = perf_counter()
            # Loading topk results based on some method of calculation
            top3_entities,top3_pairs_score = dlm.load_top_result(method='sbert',k=k)

            pairs_mapping, pairs_score = alm.dataset2entity(
                                                            datasets=lds,
                                                            entities=dme,
                                                            top_entities=top3_entities,
                                                            k = k,
                                                            method = method,
                                                            )
            finish = perf_counter()
            t = (finish-start)/60                                         
            print(r'alm_top{0}_{1}_ completes and Time consuming (min) :{2} min.'.format(k,method,t))


from time import perf_counter
if __name__ == '__main__':

    start = perf_counter()
    main = Main()
    # It is possible to test the time consumption of different methods

    # Dataset level experiment
    main.implementation_dlm(methods=[
        'edit_distance',
        # 'jaccard',
        # 'word_net_pathsim',
        # 'word2vec',
        # 'glove-300',
        # 'fasttext-300',
        # 'sbert',
        # 'bert-base-uncased',
        # 'wiki_cons',
    ],k_values=[1,3])

    # Attribute level experiment
    main.implmentation_alm(
        k=3,
        methods=[
        'edit_distance',
        # 'jaccard',
        # 'word_net_pathsim',
        # 'word2vec',
        # 'glove-300',
        # 'fasttext-300',
        # 'sbert',
        # 'bert-base-uncased',
        # 'wiki_cons',
        # 'ALmatcher',
    ])

    # Dataset level Evalution
    dlm_e = main.output_dlm_evaluatation(method='sbert',k_values=[1,3])
    print(dlm_e)


    # Attribute level evaluation
    alm_e = main.output_alm_evaluation(
        k=3,
        methods=[
        'edit_distance',
        # 'jaccard',
        # 'word_net_pathsim',
        # 'wiki_cons',
        # 'word2vec',
        # 'glove-300',
        # 'fasttext-300',
        # 'bert-base-uncased',
        # 'sbert',
        # 'ALmatcher',
    ])
    print(alm_e)

    finish = perf_counter()
    t = (finish-start)/60
    print('time(min): {0}'.format(t))


