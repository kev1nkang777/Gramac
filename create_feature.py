import numpy as np
import os,csv
from pwn import *
import pandas as pd
import networkx as nx
from tqdm import tqdm
import func_timeout
from func_timeout import func_set_timeout
import copy


def get_No_vertices(G):
    return len(G.nodes())

def get_No_edges(G):
    return len(G.edges())

def get_degree(G):
    out_degree = {d[0]:d[1] for d in G.out_degree(G.nodes())}
    in_degree = {d[0]:d[1] for d in G.in_degree(G.nodes())}
    w_out_degree = { i:out_degree[i]/max(sum(out_degree.values()),1) for i in out_degree }
    w_in_degree = { i:in_degree[i]/max(sum(in_degree.values()),1) for i in in_degree }
    oDegree = np.mean([i for i in w_out_degree.values()])
    iDegree = np.mean([i for i in w_in_degree.values()])
    return oDegree, iDegree

def get_No_connected_components(G):
    return len(list(nx.connected_components(G.to_undirected())))

def get_No_loops(G):
    return len(list(nx.simple_cycles(G)))

def get_No_PE(G):
    PE=0
    tmp=[]
    for Edge in G.edges():
        if sorted([Edge[0],Edge[1]]) in tmp:
            PE+=1
            #print(sorted([Edge[0],Edge[1]]))
            continue
        else:
            tmp.append(sorted([Edge[0],Edge[1]]))
    return PE


@func_set_timeout(15)
def Feature_collection(fpath):
    Feature=[]
    G = nx.read_gpickle(fpath)
    Feature.append(get_No_vertices(G))
    Feature.append(get_No_edges(G))
    Degree=get_degree(G)
    Feature.append(Degree[0])
    Feature.append(Degree[1])
    Feature.append(get_No_connected_components(G))
    Feature.append(get_No_loops(G))
    Feature.append(get_No_PE(G))
    return np.array(Feature)

 
def read_label(dataset_path, path):
    df = pd.read_csv(dataset_path)
    label_dict = {'BenignWare':0, 'Mirai':1, 'Tsunami':2, 'Hajime':3, 'Dofloo':4, 'Bashlite':5, 'Xorddos':6, 'Android':7, 'Pnscan':8, 'Unknown':9}

    # 转换标签列为字典
    label_mapping = dict(zip(df['filename'], df['label']))

    file_label_dict = {file_name: label_dict[label_mapping.get(os.path.splitext(file_name)[0], 'Unknown')] for file_name in tqdm(os.listdir(path)) if file_name.endswith('.gpickle')}
    # file_label_dict = {file_name: label_dict[label_mapping.get(os.path.splitext(file_name)[0], 'Unknown')] for file_name in tqdm(os.listdir(malware_path)) if file_name.endswith('.gpickle')}

    return file_label_dict
    
    
    
def create_feature(path, file_label_dict):
    fail_ct = 0
    ben_rows = []
    mal_rows = []

    for dirPath, dirName, fileName in os.walk(path):
        for f in tqdm(fileName):
            fpath = dirPath + f
            name = f.replace('.gpickle', '')
            try:
                Result = Feature_collection(fpath)
                row = [name, file_label_dict[f], *Result]
                if file_label_dict[f] == 0:
                    ben_rows.append(row)
                else:
                    mal_rows.append(row)
            except func_timeout.exceptions.FunctionTimedOut:
                pass
            except Exception as e:
                print(f'Exception occurred: {type(e).__name__}: {str(e)}')
                fail_ct += 1
                pass

    # 開啟檔案並寫入資料
    with open('./Ben_feature_sym.csv', 'a+', newline='') as ben_file:
        ben_writer = csv.writer(ben_file)
        ben_writer.writerows(ben_rows)

    with open('./Mal_feature_sym.csv', 'a+', newline='') as mal_file:
        mal_writer = csv.writer(mal_file)
        mal_writer.writerows(mal_rows)

    print('fail sample =', fail_ct)



if __name__=='__main__':
    path = r'/home/kevin/Gramac/new_train/sym_gpickle/'
    dataset_path = r'/mnt/dataset/dataset.csv'

    # Result = Feature_collection('/home/kevin/Gramac/new_train/ben_sym_gpickle/4481a75aee9c42a2313be7f84c50b3a3c1979aa9a5bfd68931063747f217ad6d.gpickle')
    
    print('creating label_dict......')
    file_label_dict = read_label(dataset_path, path)
    print('label_dict created successfully!!!!!!')
    
    # print(type(file_label_dict['4481a75aee9c42a2313be7f84c50b3a3c1979aa9a5bfd68931063747f217ad6d.gpickle']))
    
    print('creating feature......')
    create_feature(path,file_label_dict)
    print('feature created successfully !!!!')
    

