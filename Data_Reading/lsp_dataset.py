import pandas as pd 
import os
import json

class Lsp_dataset(object):
    def __init__(self):
        # Data stored path
        self.folders_path = r"E:\master_yang_test\masterthesis_pan-main\New+_Project\MATRYCS Data"
        # Data sets stored path
        self.save_path = r'E:\master_yang_test\masterthesis_pan-main\New+_Project\Data_Reading\json_files'
        dir_list = []
        for dir_name in os.listdir(self.folders_path):
            if '.xlsx' in dir_name:
                continue
            dir_list.append(dir_name)
        dir_list.sort(key=lambda x: eval(x[3:x.index('_')]), reverse=False)
        self.dir_dict = {}
        for i in range(1,len(dir_list)+1):
            self.dir_dict[i] = dir_list[i-1]


    def load_lsp_dataset(self):
        load_path = self.save_path+r"\{}.json".format('lsp_datasets')
        fpath = open(load_path,'r')
        file_data = json.load(fpath)
        file_dict = {}
        for name,file in file_data.items():
            df_temp = {}
            for i ,j in file.items():
                temp = {}
                for k in j.keys():
                    temp[eval(k)] = j.get(k)
                df_temp[i] = temp
            df_file = pd.DataFrame(df_temp)
            file_dict[name] = df_file
        return file_dict

    def _save_lsp_file(self,file,save_path,file_name):
        file_json = {}
        for k,f in file.items():
            df_file = f.to_json()
            temp = json.loads(df_file) 
            file_json[k] = temp
        
        with open(save_path+r"\{0}.json".format(file_name),"w") as f:
            f.write(json.dumps(file_json,indent=4))

    
    def _read_lsp(self):
        lsp = {}
        for i ,folder_name in self.dir_dict.items():
            key_name = r'LSP{0}'.format(i)
            read_path = self.folders_path+r"\{0}".format(folder_name)
            df_list = []
            for d_name in os.listdir(read_path):
                d_path = read_path+r"\{0}".format(d_name)
                files_list = os.listdir(d_path)
                if len(files_list)>0:
                    file_name = files_list[0]
                    fp = d_path+r"\{0}".format(file_name)
                    file = pd.read_excel(fp)
                    file.dropna(axis='index',how='all',inplace=True)        
                    # file.dropna(axis='columns',how='all',inplace=True)
                    list_columns = list(file)
                    df_temp = pd.DataFrame(list_columns)
                    dataset_name = r"{0}".format(d_name)
                    df_temp.columns = [dataset_name]
                    df_list.append(df_temp)
            if len(df_list)>0:
                lsp[key_name] = pd.concat(df_list,axis=1,join='outer')
        file_name = 'lsp_datasets'
        self._save_lsp_file(lsp, self.save_path,file_name)
        return lsp




    # Read Lsp data sets
    # LSP1_BTC
    def _read_lsp1(self):
        read_path = self.folders_path+r"\{0}".format(self.dir_dict[1])
        # save_path = self.save_path+r"\{0}".format(self.dir_dict[1])
        df_list = []
        for d_name in os.listdir(read_path):
            d_path = read_path+r"\{0}".format(d_name)
            files_list = os.listdir(d_path)
            # Because all of the files in one lsp dataset folder have the same form, so only pick one file:files_list[0]
            file_name = files_list[0]
            fp = d_path+r"\{0}".format(file_name)
            file = pd.read_excel(fp)
            file = file.loc[:,'DATE':'VALUE']
            # file.dropna(axis='index',how='all',inplace=True)        
            # file.dropna(axis='columns',how='all',inplace=True)
            list_columns = list(file) 
            temp = []
            a = "Unnamed"
            for i in list_columns:
                if a not in str(i):
                    temp.append(str(i))
            if len(temp)>0:
                df_temp = pd.DataFrame(temp)
                dataset_name = r"{0}".format(d_name)
                df_temp.columns = [dataset_name]
                df_list.append(df_temp)
        lsp1 = pd.concat(df_list,axis=1,join='outer')
        return lsp1


    # LSP2_Fasda
    def _read_lsp2(self):
        read_path = self.folders_path+r"\{0}".format(self.dir_dict[2])
        df_list = []
        for d_name in os.listdir(read_path):
            d_path = read_path+r"\{0}".format(d_name)
            files_list = os.listdir(d_path)
            # Because all of the files in one lsp dataset folder have the same form, so only pick one file:files_list[0]
            file_name = files_list[0]
            fp = d_path+r"\{0}".format(file_name)
            file = pd.read_csv(fp)
            file.dropna(axis='index',how='all',inplace=True)        
            file.dropna(axis='columns',how='all',inplace=True)
            list_columns = list(file)
            temp = []
            a = "Unnamed"
            for i in list_columns:
                if a not in str(i):
                    temp.append(str(i))
            if len(temp)>0:
                df_temp = pd.DataFrame(temp)
                dataset_name = r"{0}".format(d_name)
                df_temp.columns = [dataset_name]
                df_list.append(df_temp)
        lsp2 = pd.concat(df_list,axis=1,join='outer')
        return lsp2

    # LSP3_VEOLIA
    def _read_lsp3(self):
        pass

    # LSP4_ASM
    def _read_lsp4(self):
        pass

    # LSP5_Coopernico_Data gathering
    def _read_lsp5(self):
        read_path = read_path = self.folders_path+r"\{0}".format(self.dir_dict[5])
        df_list =[]
        for d_name in os.listdir(read_path):
            d_path = read_path+r"\{}".format(d_name)
            files_list = os.listdir(d_path)
            # Because all of the files in one lsp dataset folder have the same form, so only pick one file:files_list[0]
            file_name = files_list[0]
            fp = d_path+r"\{0}".format(file_name)
            if d_name == "05_Listagem Energia Consumida":
                file = pd.read_excel(fp,skiprows=4)
                file_columns = list(file)
                temp = []
                a = "Unnamed"
                for i in file_columns:
                    if a not in str(i):
                        temp.append(str(i))
                file = pd.DataFrame(temp)
                file.columns = [d_name]
                df_list.append(file)
            elif d_name == "15_billing information":
                file = pd.read_excel(fp,skiprows=5)
                file_columns = list(file)
                temp = []
                a = "Unnamed"
                for i in file_columns:
                    if a not in str(i):
                        temp.append(str(i))
                file = pd.DataFrame(temp)
                file.columns = [d_name]
                df_list.append(file)
            elif d_name == "15_Lista de Pontos de Entrega":
                file = pd.read_excel(fp,skiprows=4)
                file_columns = list(file)
                temp = []
                a = "Unnamed"
                for i in file_columns:
                    if a not in str(i):
                        temp.append(str(i))
                file = pd.DataFrame(temp)
                file.columns = [d_name]
                df_list.append(file)
            elif d_name == "15_Price List":
                file = pd.read_excel(fp,skiprows=5)
                file_columns = list(file)
                temp = []
                a = "Unnamed"
                for i in file_columns:
                    if a not in str(i):
                        temp.append(str(i))
                file = pd.DataFrame(temp)
                file.columns = [d_name]
                df_list.append(file)          
            elif file_name == "07_single project month production example":
                file = pd.read_csv(fp,skipinitialspace=True)
                file_columns = list(file)
                temp = []
                for i in file_columns:
                    if 'Unnamed' not in str(i):
                        temp.append(i)
                file = pd.DataFrame(temp)
                file.columns = [d_name]
                df_list.append(file)
            
        lsp5 = pd.concat(df_list,axis=1,join='outer')
        return lsp5

    # LSP6_VEOLIA
    def _read_lsp6(self):
        pass

    # LSP7_ICLEI
    def _read_lsp7(self):
        read_path = self.folders_path+r"\{0}".format(self.dir_dict[7])
        df_list = []
        for d_name in os.listdir(read_path):
            d_path = read_path+r"\{}".format(d_name)
            files_list = os.listdir(d_path)
            # Because all of the files in one lsp dataset folder have the same form, so only pick one file:files_list[0]
            file_name = files_list[0]
            fp = d_path+r"\{}".format(file_name)
            file = pd.read_csv(fp)
            file_columns = list(file)
            file = pd.DataFrame(file_columns)
            file.columns = [d_name]
            df_list.append(file)
        lsp7 = pd.concat(df_list,axis=1,join='outer')
        return lsp7

    # LSP8_GDYNIA
    def _read_lsp8(self):
        pass


    # LSP9_EREN
    def _read_lsp9(self):
        pass

    # LSP10_LEIF
    def _read_lsp10(self):
        read_path = read_path = read_path = read_path = self.folders_path+r"\{0}".format(self.dir_dict[10])
        df_list = []
        for d_name in os.listdir(read_path):
            d_path = read_path+r"\{}".format(d_name)
            files_list = os.listdir(d_path)
            file_name = files_list[0]
            # Because all of the files in one lsp dataset folder have the same form, so only pick one file:files_list[0]
            fp = d_path+r"\{}".format(file_name)
            file = pd.read_excel(fp)
            file.dropna(axis='index', how='all', inplace=True)
            file.dropna(axis='columns', how='all', inplace=True)
            file_columns = list(file)
            temp = []
            a = "Unnamed"
            for i in file_columns:
                if a not in str(i):
                    temp.append(str(i))
            if len(temp)>0:
                file = pd.DataFrame(temp)
                file.columns = [d_name]
                df_list.append(file)
        lsp10 = pd.concat(df_list,axis=1,join='outer')
        return lsp10

    # LSP11_HOUSING EUROPE
    def read_lsp11(self):
        pass

    def read_lsp_data(self):
        df_lsp1 = self._read_lsp1()
        df_lsp2 = self._read_lsp2()
        # df_lsp3 = self.read_lsp3()
        df_lsp5 = self._read_lsp5()
        df_lsp7 = self._read_lsp7()
        df_lsp10 = self._read_lsp10()
        lsp = {
            "LSP1":df_lsp1,
            "LSP2":df_lsp2,
            "LSP5":df_lsp5,
            "LSP7":df_lsp7,
            "LSP10":df_lsp10
        }
        file_name = 'lsp_datasets'
        self._save_lsp_file(lsp, self.save_path,file_name)
        return lsp

    
if __name__ == "__main__":
    lds =Lsp_dataset()
    print(lds.dir_dict)
    lsp_datasets = lds._read_lsp()
    print(lsp_datasets)

    
    # lsp_datasets=lds.read_lsp_data()
    # print(lds.load_lsp_dataset())