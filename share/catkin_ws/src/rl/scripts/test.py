#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 


import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# try1
# horizon1 0.229 -> 0.230 --> 0.234
# horizon3 0.225 -> 0.227 --> 0.230
# horizon5 0.219 -> 0.224 --> 0.234
# horizon7 0.214 -> 0.222 --> 0.224

# try2
# horizon1 0.112 -> 0.118 --> 0.125
# horizon3 0.111 -> 0.117 --> 0.123
# horizon5 0.109 -> 0.116 --> 0.125
# horizon7 0.107 -> 0.114 --> 0.123

# try3
# horizon1 0.158 -> 0.179 --> 0.226
# horizon3 0.163 -> 0.187 --> 0.235
# horizon5 0.164 -> 0.194 --> 0.226
# horizon7 0.167 -> 0.200 --> 0.241


def main():
    
    datasets = np.load('./result/deriv_orient_try3_manip00_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/deriv_orient_try3_manip05_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/deriv_orient_try3_manip10_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/deriv_orient_try3_manip00_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/deriv_orient_try3_manip05_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/deriv_orient_try3_manip10_horizon3.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/deriv_orient_try3_manip00_horizon5.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/deriv_orient_try3_manip05_horizon5.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))  
        
    datasets = np.load('./result/deriv_orient_try3_manip10_horizon1.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
    
    
    datasets = np.load('./result/deriv_orient_try3_manip00_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
    datasets = np.load('./result/deriv_orient_try3_manip05_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))  
        
    datasets = np.load('./result/deriv_orient_try3_manip10_horizon7.npy', encoding='bytes')
    mean = []
    for i in range(5):
        real_m_index = np.asarray(datasets[i]['real_m_index'])
        mean.append(np.mean(real_m_index))
    print(np.mean(np.asarray(mean)))
        
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