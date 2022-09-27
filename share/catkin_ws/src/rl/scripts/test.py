#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 


import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


## real m index
###naive
# try1
# horizon1 0.229 -> 0.230 --> 0.234   85.2 -> 84.8 -> 85.0
# horizon3 0.225 -> 0.227 --> 0.230   86.4 -> 85.8 -> 85.4
# horizon5 0.219 -> 0.224 --> 0.234   89.0 -> 87.8 -> 85.0
# horizon7 0.214 -> 0.222 --> 0.224   94.4 -> 90.8 -> 91.0

# try2
# horizon1 0.112 -> 0.118 --> 0.125   67.2 -> 65.2 -> 64.6
# horizon3 0.111 -> 0.117 --> 0.123   69.0 -> 67.6 -> 67.0
# horizon5 0.109 -> 0.116 --> 0.125   70.4 -> 68.6 -> 64.6
# horizon7 0.107 -> 0.114 --> 0.123   73.4 -> 71.0 -> 68.8 

# try3
# horizon1 0.158 -> 0.179 --> 0.226   193.0 -> 195.4 -> 215.4
# horizon3 0.163 -> 0.187 --> 0.235   196.2 -> 200.8 -> 222.4
# horizon5 0.164 -> 0.194 --> 0.226   196.2 -> 204.0 -> 215.4
# horizon7 0.167 -> 0.200 --> 0.241   198.6 -> 208.4 -> 231.2

# try4
# horizon1 0.115 -> 0.115 --> 0.116    53.4 -> 52.6 -> 53.4
# horizon3 0.116 -> 0.117 --> 0.118    55.0 -> 54.2 -> 54.6
# horizon5 0.119 -> 0.119 --> 0.116    56.0 -> 56.0 -> 53.4
# horizon7 0.120 -> 0.119 --> 0.120    57.6 -> 57.2 -> 57.8

# try5
# horizon1 0.136 -> 0.146 --> 0.151    69.8 -> 70.2 -> 71.2
# horizon3 0.135 -> 0.143 --> 0.150    71.2 -> 71.6 -> 72.0
# horizon5 0.136 -> 0.142 --> 0.151    72.2 -> 73.0 -> 74.4
# horizon7 0.145 -> 0.143 --> 0.147    161.0 -> 80.0 -> 75.6


###deriv
# try1
# horizon1  
# horizon3  
# horizon5  
# horizon7  

# try2
# horizon1  0.132 -> 0.136 -> 0.143   63.8 -> 64.2 -> 65.4
# horizon3  0.133 -> 0.137 -> 0.145   65.2 -> 64.8 -> 67.4
# horizon5  0.133 -> 0.136 -> 0.143   64.3 -> 66.0 -> 65.4
# horizon7  0.132 -> 0.134 -> 0.146   66.4 -> 66.0 -> 69.2

# try3
# horizon1  0.162 -> 0.178 -> 0.222   190.0 -> 196.4 -> 212.8
# horizon3  0.162 -> 0.184 -> 0.235   192.8 -> 199.0 -> 223.0
# horizon5  0.161 -> 0.194 -> 0.222   194.0 -> 201.6 -> 212.8
# horizon7  0.160 -> 0.198 -> 0.241   195.2 -> 206.8 -> 228.4

# try4
# horizon1  0.120 -> 0.123 -> 0.124  53.4 -> 53.4 -> 54.0
# horizon3  0.119 -> 0.122 -> 0.123  54.0 -> 54.4 -> 54.8
# horizon5  0.119 -> 0.121 -> 0.124  54.8 -> 55.2 -> 54.0
# horizon7  0.117 -> 0.122 -> 0.124  54.8 -> 55.8 -> 55.8

# try5
# horizon1  0.160 -> 0.166 -> 0.169  66.0 -> 67.4 -> 67.8
# horizon3  0.155 -> 0.163 -> 0.167  66.8 -> 68.2 -> 68.2
# horizon5  0.154 -> 0.158 -> 0.169  67.2 -> 69.4 -> 67.8
# horizon7  0.147 -> 0.157 -> 0.162  69.4 -> 70.2 -> 71.0

#########################################################################################
###naive
# try1
# horizon1 0.226 0.229 0.230  84.6 54.0 87.6
# horizon3 0.224 0.227 0.229  85.6 85.6 85.6
# horizon5 0.216 0.222 0.230  92.0 89.3 84.0
# horizon7 fail fail fail

# try2
# horizon1 0.126 0.129 0.131  63.3 63.6 63.6
# horizon3 0.125 0.128 0.131  64.3 63.6 64.3
# horizon5 0.120 0.123 0.131  69.0 70.0 63.6
# horizon7 0.122 0.126 0.131  66.0 66.3 66.6

# try3
# horizon1 0.170 0.205 0.230  192.6 205.3 224.6  
# horizon3 0.169 0.203 0.228  196.0 205.6 229.0  
# horizon5 0.169 0.203 0.230  197.0 209.6 224.6  
# horizon7 0.168 0.204 0.227  199.6 211.0 233.6  

# try4
# horizon1 0.114 0.114 0.115  52.0 52.3 53.0
# horizon3 0.114 0.113 0.114  53.3 53.0 52.3
# horizon5 0.114 0.115 0.115  54.0 53.0 53.0
# horizon7 0.113 0.115 0.115  54.6 55.0 56.0

# try5
# horizon1 0.135 0.137 0.140  70.3 67.6 67.0
# horizon3 0.135 0.136 0.139  72.6 72.0 69.0
# horizon5 0.132 0.133 0.134  73.3 78.6 67.0
# horizon7 0.130 0.132 0.134  76.6 77.6 75.0

###deriv
# try1
# horizon1 0.231 0.238 0.247  83.6 84.3 84.3
# horizon3 0.230 0.235 0.244  85.0 84.3 86.6
# horizon5 0.226 0.231 0.247  85.3 88.0 84.3
# horizon7 0.224 0.225 0.235  89.6 88.3 89.6

# try2
# horizon1 0.133 0.133 0.134  64.3 64.3 65.3
# horizon3 0.134 0.134 0.134  64.3 66.0 65.3
# horizon5 0.134 0.134 0.134  66.0 65.3 65.0
# horizon7 0.132 0.134 0.134  67.0 67.3 67.6

# try3 (얘만 성능 안좋음)
# horizon1 0.164 0.180 0.218  190.3 194.6 210.0
# horizon3 0.162 0.181 0.215  191.6 197.3 212.3
# horizon5 0.159 0.180 0.218  191.6 197.3 210.0
# horizon7 0.158 0.180 0.215  193.3 200.0 218.3

# try4
# horizon1 0.116 0.120 0.123  52.6 53.6 53.3
# horizon3 0.115 0.119 0.123  53.6 54.0 55.0
# horizon5 0.117 0.119 0.123  54.0 55.0 53.3
# horizon7 0.117 0.118 0.122  53.3 54.0 55.6

# try5
# horizon1 0.153 0.153 0.164  63.6 64.3 65.0
# horizon3 0.152 0.153 0.164  65.6 65.6 66.0
# horizon5 0.151 0.152 0.164  67.6 67.3 65.0
# horizon7 0.149 0.151 0.162  70.3 69.6 70.3



### 
def main():


    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip00_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip05_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip10_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip00_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip05_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip10_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip00_horizon5.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip05_horizon5.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))  
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip10_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
    '''
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip00_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip05_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))  
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip10_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
    '''
     
     
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip00_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip05_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip10_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip00_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip05_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip10_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip00_horizon5.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip05_horizon5.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))  
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip10_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
    
    '''
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip00_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip05_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))  
        
    datasets = np.load('./result/exp2/naive/naive_orient_try1_manip10_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(3):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(real_m_index.shape)
    print(np.mean(np.asarray(mean)))
    '''
    '''
    fig = plt.figure(figsize=(30,20))
    
    ax1 = fig.add_subplot(2,1,1, projection='3d')
    ax1.title.set_text('trajectory 1~50')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    ax1.set_xlim([-0.8,0.8])
    ax1.set_ylim([0.3,0.85])
    ax1.set_zlim([0.3,0.85])


    name2 = 'real_cur_pos'
    
    ax1.plot(np.asarray(datasets[0][name2])[:,0], np.asarray(datasets[0][name2])[:,1], np.asarray(datasets[0][name2])[:,2],lw=2, color='royalblue', label = 'real') 
    ax1.plot(np.asarray(datasets[1][name2])[:,0], np.asarray(datasets[1][name2])[:,1], np.asarray(datasets[1][name2])[:,2],lw=2, color='royalblue', label = 'real') 
    ax1.plot(np.asarray(datasets[2][name2])[:,0], np.asarray(datasets[2][name2])[:,1], np.asarray(datasets[2][name2])[:,2],lw=2, color='royalblue', label = 'real') 
    ax1.plot(np.asarray(datasets[3][name2])[:,0], np.asarray(datasets[3][name2])[:,1], np.asarray(datasets[3][name2])[:,2],lw=2, color='royalblue', label = 'real') 
    ax1.plot(np.asarray(datasets[4][name2])[:,0], np.asarray(datasets[4][name2])[:,1], np.asarray(datasets[4][name2])[:,2],lw=2, color='royalblue', label = 'real') 
    
    
    ax2 = fig.add_subplot(2,1,2, projection='3d')
    ax2.title.set_text('trajectory 1~50')
    ax2.set_xlabel('roll')
    ax2.set_ylabel('pitch')
    ax2.set_zlabel('yaw')
    
    #ax2.set_xlim([-0.8,0.8])
    #ax2.set_ylim([0.3,0.85])
    #ax2.set_zlim([0.3,0.85])
    
    ax2.plot(np.asarray(datasets[0][name2])[:,3], np.asarray(datasets[0][name2])[:,4], np.asarray(datasets[0][name2])[:,5],lw=2, color='royalblue', label = 'real') 
    ax2.plot(np.asarray(datasets[1][name2])[:,3], np.asarray(datasets[1][name2])[:,4], np.asarray(datasets[1][name2])[:,5],lw=2, color='royalblue', label = 'real') 
    ax2.plot(np.asarray(datasets[2][name2])[:,3], np.asarray(datasets[2][name2])[:,4], np.asarray(datasets[2][name2])[:,5],lw=2, color='royalblue', label = 'real') 
    ax2.plot(np.asarray(datasets[3][name2])[:,3], np.asarray(datasets[3][name2])[:,4], np.asarray(datasets[3][name2])[:,5],lw=2, color='royalblue', label = 'real') 
    ax2.plot(np.asarray(datasets[4][name2])[:,3], np.asarray(datasets[4][name2])[:,4], np.asarray(datasets[4][name2])[:,5],lw=2, color='royalblue', label = 'real') 
    plt.show()
    '''
    
    #path = '/root/share/catkin_ws/src/ur10_teleop_interface/scripts/'
    #filename = 'ntraj50_params_ori02_xyz_08_05_in_055_03.npy'
    #datasets = np.load('./dataset/ntraj50_params_ori02_xyz_08_05_in_055_03.npy', encoding='bytes')
    
    '''
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
    '''
    fig = plt.figure(figsize=(30,20))
    
    ax1 = fig.add_subplot(3,2,1, projection='3d')
    ax1.title.set_text('trajectory 1~50')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    ax1.set_xlim([-0.8,0.8])
    ax1.set_ylim([0.3,0.85])
    ax1.set_zlim([0.3,0.85])
    
    #ax1.set_xlim([-3.14,3.14])
    #ax1.set_ylim([-3.14,3.14])
    #ax1.set_zlim([-3.14,3.14])
    index1 = 0
    index2 = 1
    index3 = 2
    name = b'unity_cur_pos'
    name2 = b'real_cur_pos'
    
    ax1.plot(np.asarray(datasets[-1][name])[:,index1], np.asarray(datasets[-1][name])[:,index2], np.asarray(datasets[-1][name])[:,index3],lw=2, color='coral',label='unity')
    ax1.plot(np.asarray(datasets[-1][name2])[:,index1], np.asarray(datasets[-1][name2])[:,index2], np.asarray(datasets[-1][name2])[:,index3],lw=2, color='royalblue', label = 'real') 
    
    for i in range(1,50):
        dataset = datasets[i]
        ax1.plot(np.asarray(datasets[i][name])[:,index1], np.asarray(datasets[i][name])[:,index2], np.asarray(datasets[i][name])[:,index3],lw=2, color='coral')
        ax1.plot(np.asarray(datasets[i][name2])[:,index1], np.asarray(datasets[i][name2])[:,index2], np.asarray(datasets[i][name2])[:,index3],lw=2, color='royalblue')
    ax1.legend()
    
    ax2 = fig.add_subplot(3,2,2, projection='3d')
    ax2.title.set_text('trajectory 1~10')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim([-0.8,0.8])
    ax2.set_ylim([0.3,0.85])
    ax2.set_zlim([0.3,0.85])
    
    for i in range(0,10):
        ax2.plot(np.asarray(datasets[i][name])[:,index1], np.asarray(datasets[i][name])[:,index2], np.asarray(datasets[i][name])[:,index3],lw=2, color='coral')
        ax2.plot(np.asarray(datasets[i][name2])[:,index1], np.asarray(datasets[i][name2])[:,index2], np.asarray(datasets[i][name2])[:,index3],lw=2, color='royalblue')
    
    ax3 = fig.add_subplot(3,2,3, projection='3d')
    ax3.title.set_text('trajectory 11~20')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_xlim([-0.8,0.8])
    ax3.set_ylim([0.3,0.85])
    ax3.set_zlim([0.3,0.85])
    
    for i in range(11,20):   
        ax3.plot(np.asarray(datasets[i][name])[:,index1], np.asarray(datasets[i][name])[:,index2], np.asarray(datasets[i][name])[:,index3],lw=2, color='coral')
        ax3.plot(np.asarray(datasets[i][name2])[:,index1], np.asarray(datasets[i][name2])[:,index2], np.asarray(datasets[i][name2])[:,index3],lw=2, color='royalblue')
        
    ax4 = fig.add_subplot(3,2,4, projection='3d')
    ax4.title.set_text('trajectory 21~30')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.set_xlim([-0.8,0.8])
    ax4.set_ylim([0.3,0.85])
    ax4.set_zlim([0.3,0.85])
    
    for i in range(21,30):
        ax4.plot(np.asarray(datasets[i][name])[:,index1], np.asarray(datasets[i][name])[:,index2], np.asarray(datasets[i][name])[:,index3],lw=2, color='coral')
        ax4.plot(np.asarray(datasets[i][name2])[:,index1], np.asarray(datasets[i][name2])[:,index2], np.asarray(datasets[i][name2])[:,index3],lw=2, color='royalblue')
        
    ax5 = fig.add_subplot(3,2,5, projection='3d')
    ax5.title.set_text('trajectory 31~40')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')
    ax5.set_xlim([-0.8,0.8])
    ax5.set_ylim([0.3,0.85])
    ax5.set_zlim([0.3,0.85])
    
    for i in range(31,40):
        ax5.plot(np.asarray(datasets[i][name])[:,index1], np.asarray(datasets[i][name])[:,index2], np.asarray(datasets[i][name])[:,index3],lw=2, color='coral')
        ax5.plot(np.asarray(datasets[i][name2])[:,index1], np.asarray(datasets[i][name2])[:,index2], np.asarray(datasets[i][name2])[:,index3],lw=2, color='royalblue')
        
    ax6 = fig.add_subplot(3,2,6, projection='3d')
    ax6.title.set_text('trajectory 41~50')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('z')
    ax6.set_xlim([-0.8,0.8])
    ax6.set_ylim([0.3,0.85])
    ax6.set_zlim([0.3,0.85])
    
    for i in range(41,50):
        ax6.plot(np.asarray(datasets[i][name])[:,index1], np.asarray(datasets[i][name])[:,index2], np.asarray(datasets[i][name])[:,index3],lw=2, color='coral')
        ax6.plot(np.asarray(datasets[i][name2])[:,index1], np.asarray(datasets[i][name2])[:,index2], np.asarray(datasets[i][name2])[:,index3],lw=2, color='royalblue')
    

        
    plt.legend(fontsize=15)
    plt.show()
    fig.savefig('./trajectory comparison.png')
    fig.savefig('./trajectory.pdf',dpi=600)
    '''
if __name__ == '__main__':
    main()
    
    

# draw error histogram for each axis
'''
index1 = 0
index2 = 1
index3 = 2
name = b'unity_cur_pos'
name2 = b'real_cur_pos'

error_x = np.array([])
error_y = np.array([])
error_z = np.array([])

for i in range(50):
    error_x = np.hstack([error_x,(np.sqrt((np.asarray(datasets[i][name])[:,index1] - np.asarray(datasets[i][name2])[:,index1])**2))])
    error_y = np.hstack([error_y,(np.sqrt((np.asarray(datasets[i][name])[:,index2] - np.asarray(datasets[i][name2])[:,index2])**2))])
    error_z = np.hstack([error_z,(np.sqrt((np.asarray(datasets[i][name])[:,index3] - np.asarray(datasets[i][name2])[:,index3])**2))])      
plt.hist(np.hstack([error_x,error_y,error_z]),bins=20)
plt.savefig('./error_histogram.pdf')
plt.show()

max_error_x = 0.0
max_error_y = 0.0
max_error_z = 0.0

for i in range(50):
    _max_error_x = np.max(np.sqrt((np.asarray(datasets[i][name])[:,index1] - np.asarray(datasets[i][name2])[:,index1])**2))
    _max_error_y = np.max(np.sqrt((np.asarray(datasets[i][name])[:,index2] - np.asarray(datasets[i][name2])[:,index2])**2))
    _max_error_z = np.max(np.sqrt((np.asarray(datasets[i][name])[:,index3] - np.asarray(datasets[i][name2])[:,index3])**2))
    
    if max_error_x < _max_error_x:
        max_error_x = _max_error_x
        
    if max_error_y < _max_error_y:
        max_error_y = _max_error_y
            
    if max_error_z < _max_error_z:
        max_error_z = _max_error_z
        
print(max_error_x, max_error_y, max_error_z)
'''