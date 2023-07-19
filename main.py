

# In[ ]:

import numpy as np
import r2pipe, os, sys
import json
from pwn import *
import pandas as pd
import networkx as nx
import joblib
import copy

from utils import parameter_parser
from torch_tools import TorchTrainer , MLP
from torch.utils.data import DataLoader
import torch



def create_graph(path):
    r2 = r2pipe.open(path)
    r2.cmd('aaaa')
    data = r2.cmd('agCd')
    label = {}
    G = nx.DiGraph()
    for lines in data.split('\n'):
        tmp = []
        for words in lines.split():
            if words[0] == '"':
                words = words.replace('"', '')
            tmp.append(words)
        try:
            if tmp[1][1] == 'l':
                func = tmp[1][7:]
                func = func.replace('"', '')
                label[tmp[0]] = func
        except:
            pass
    for lines in data.split('\n'):
        tmp = []
        for words in lines.split():
            if words[0] == '"':
                words = words.replace('"', '')
            tmp.append(words)
        try:
            if tmp[1] == '->':
                G.add_edge(label[tmp[0]], label[tmp[2]])
        except:
            pass
    r2.quit()
    return G

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
    if np.isnan(oDegree) or np.isnan(iDegree):
        oDegree = 0
        iDegree = 0
    return iDegree,oDegree

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

def Feature_collection(path):
    Feature=[]
    
    try:
        G=create_graph(path)
    except:
        print('fail to constuct the FCG.')
    Feature.append(get_No_vertices(G))
    Feature.append(get_No_edges(G))
    Degree=get_degree(G)
    Feature.append(Degree[0])
    Feature.append(Degree[1])
    Feature.append(get_No_connected_components(G))
    Feature.append(get_No_loops(G))
    Feature.append(get_No_PE(G))
    return np.array(Feature)



def Predict(X,clf,classification):
    if clf != 'mlp':
        if classification:
            model = joblib.load('./classification_model/'+clf+'_model.joblib')
            result = model.predict_proba(X)
        else:
            model = joblib.load('./detection_model/'+clf+'_model.joblib')
            result = model.predict_proba(X)
    else:
        X = torch.tensor(X.astype(np.float32))
        if classification:
            model = MLP(num_features=X.size(1), hidden_channels=64, num_classes=9).to('cpu')
            
            predict_trainer = TorchTrainer(model)
            predict_trainer.load('./classification_model/mlp_cls.pt')
        else:
            model = MLP(num_features=X.size(1), hidden_channels=64, num_classes=2).to('cpu')
            predict_trainer = TorchTrainer(model)
            predict_trainer.load('./detection_model/mlp_md.pt')

        train_data_loader = DataLoader(X, batch_size=1, num_workers=0, drop_last=False, shuffle=False)
        result = predict_trainer.predict(train_data_loader).numpy()
    return result


def main(args):
    # default: fail to predict -> -1
    result = [-1] 
    
    labels = ['BenignWare','Malware']
    if args.classify:
        labels = ['BenignWare', 'Mirai', 'Bashlite', 'Android', 'Tsunami', 'Dofloo', 'Xorddos', 'Hajime', 'Pnscan']
    
    try:
        feature = Feature_collection(args.input_path)
    except:
        print('fail to collect the feature.')
    
    # prediction
    feature = np.array(feature).reshape(1,-1)
    result = Predict(feature,args.model,args.classify)
    # print(labels[result[0]])
    print(result)
    
    return result


if __name__=='__main__':
    args = parameter_parser()
    if os.path.isfile(args.input_path):
        main(args)
    else:
        print('input path error')


# %%
