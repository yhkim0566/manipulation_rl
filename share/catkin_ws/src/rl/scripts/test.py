#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 


import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt




def main():
    path = '/root/share/catkin_ws/src/ur10_teleop_interface/scripts/'
    filename = 'datasets_damp_2500.npy'
    datasets = np.load('./dataset/datasets_damp_2500.npy', encoding='bytes')

    ratio = 0.8
    dataset_size = datasets.shape[0]
    dataset_index = np.linspace(0,dataset_size-1,dataset_size,dtype=int)
    np.random.shuffle(dataset_index)
    print(dataset_index)
    train_index = dataset_index[:int(dataset_size*ratio)]
    eval_index = dataset_index[int(dataset_size*ratio):]
    
    train_data = datasets[train_index]
    eval_data = datasets[eval_index]
    print(train_data[0].keys())
    _train_data = defaultdict(list)
    for i in range(len(train_data)):
        for k in train_data[0].keys():
            _train_data[k.decode('utf-8')].extend(np.asarray(train_data[i][k]))
            
    for k in _train_data.keys():
        print(k)
        print(np.asarray(_train_data[k]).shape)
    
if __name__ == '__main__':
    main()