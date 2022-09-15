#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 


import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



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
    
    m_index = _train_data['real_m_index']        
    print(np.min(np.asarray(m_index)))
    print(np.max(np.asarray(m_index)))
    plt.hist(m_index)
    plt.show()
    
    '''
    fig = plt.figure(figsize=(30,20))
    
    ax1 = fig.add_subplot(3,2,1, projection='3d')
    ax1.title.set_text('trajectory 1~50')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([0,1])
    for i in range(50):
        dataset = datasets[i]
        ax1.plot3D(np.asarray(datasets[i][b'real_cur_pos'])[:,0], np.asarray(datasets[i][b'real_cur_pos'])[:,1], np.asarray(datasets[i][b'real_cur_pos'])[:,2],lw=2)
    
    
    ax2 = fig.add_subplot(3,2,2, projection='3d')
    ax2.title.set_text('trajectory 1~10')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim([-1,1])
    ax2.set_ylim([-1,1])
    ax2.set_zlim([0,1])
    for i in range(0,10):
        dataset = datasets[i]
        ax2.plot3D(np.asarray(datasets[i][b'real_cur_pos'])[:,0], np.asarray(datasets[i][b'real_cur_pos'])[:,1], np.asarray(datasets[i][b'real_cur_pos'])[:,2],lw=2)
    
    ax3 = fig.add_subplot(3,2,3, projection='3d')
    ax3.title.set_text('trajectory 11~20')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_xlim([-1,1])
    ax3.set_ylim([-1,1])
    ax3.set_zlim([0,1])
    for i in range(10,20):   
        dataset = datasets[i]
        ax3.plot3D(np.asarray(datasets[i][b'real_cur_pos'])[:,0], np.asarray(datasets[i][b'real_cur_pos'])[:,1], np.asarray(datasets[i][b'real_cur_pos'])[:,2],lw=2)
    
    ax4 = fig.add_subplot(3,2,4, projection='3d')
    ax4.title.set_text('trajectory 21~30')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.set_xlim([-1,1])
    ax4.set_ylim([-1,1])
    ax4.set_zlim([0,1])
    for i in range(20,30):
        dataset = datasets[i]
        ax4.plot3D(np.asarray(datasets[i][b'real_cur_pos'])[:,0], np.asarray(datasets[i][b'real_cur_pos'])[:,1], np.asarray(datasets[i][b'real_cur_pos'])[:,2],lw=2)
    
    ax5 = fig.add_subplot(3,2,5, projection='3d')
    ax5.title.set_text('trajectory 31~40')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')
    ax5.set_xlim([-1,1])
    ax5.set_ylim([-1,1])
    ax5.set_zlim([0,1])
    for i in range(30,40):
        dataset = datasets[i]
        ax5.plot3D(np.asarray(datasets[i][b'real_cur_pos'])[:,0], np.asarray(datasets[i][b'real_cur_pos'])[:,1], np.asarray(datasets[i][b'real_cur_pos'])[:,2],lw=2)
    
    ax6 = fig.add_subplot(3,2,6, projection='3d')
    ax6.title.set_text('trajectory 41~50')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('z')
    ax6.set_xlim([-1,1])
    ax6.set_ylim([-1,1])
    ax6.set_zlim([0,1])
    for i in range(40,50):
        dataset = datasets[i]
        ax6.plot3D(np.asarray(datasets[i][b'real_cur_pos'])[:,0], np.asarray(datasets[i][b'real_cur_pos'])[:,1], np.asarray(datasets[i][b'real_cur_pos'])[:,2],lw=2)
        
    plt.legend(fontsize=15)
    plt.show()
    fig.savefig('./trajectories.png')
    fig.savefig('./trajectories.pdf',dpi=600)
    '''
    '''

    ''' 
if __name__ == '__main__':
    main()
    