#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 


import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D




def main():
    datasets = np.load('./result/obstacle_avoidance_naive1.npy', encoding='bytes')
    dataset1 = datasets[0]
    
    datasets = np.load('./result/obstacle_avoidance_deriv1.npy', encoding='bytes')
    dataset2 = datasets[0]
    
    datasets = np.load('./result/obstacle_avoidance_naive5.npy', encoding='bytes')
    dataset3 = datasets[0]
    
    datasets = np.load('./result/obstacle_avoidance_deriv5.npy', encoding='bytes')
    dataset4 = datasets[0]
    
    fig = plt.figure(figsize=(30,20))
    ax1 = fig.add_subplot(1,1,1, projection='3d')
    
    ax1.scatter(-0.5106587048041216, 0.419292690922957, 0.8189151099771221, s = 500)
    ax1.scatter(-0.38178888,  0.53573178,  0.47328898, s = 500)
    ax1.scatter(-0.44622379,  0.47751223,  0.64610205, s= 500)
    ax1.plot(np.asarray(dataset1['real_cur_pos'])[:,0],np.asarray(dataset1['real_cur_pos'])[:,1],np.asarray(dataset1['real_cur_pos'])[:,2], lw=3, label='naive1')
    ax1.plot(np.asarray(dataset2['real_cur_pos'])[:,0],np.asarray(dataset2['real_cur_pos'])[:,1],np.asarray(dataset2['real_cur_pos'])[:,2], lw=3, label='deriv1')
    ax1.plot(np.asarray(dataset3['real_cur_pos'])[:,0],np.asarray(dataset3['real_cur_pos'])[:,1],np.asarray(dataset3['real_cur_pos'])[:,2], lw=3, label='naive5')
    ax1.plot(np.asarray(dataset4['real_cur_pos'])[:,0],np.asarray(dataset4['real_cur_pos'])[:,1],np.asarray(dataset4['real_cur_pos'])[:,2], lw=3, label='deriv5')

    
    
    plt.legend(fontsize=15)
    plt.show()

    #3.75 [0.08551] [-0.35506086]
    #4.65 [0.07716828] [-0.36968701]

    #4.55 [0.08921668] [-0.43942496]
    #4.55 [0.08771234] [-0.42816943]

if __name__ == '__main__':
    main()