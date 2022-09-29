#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 


import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import


# 거리 계산을 위한 함수 필요함
# real cur pos 의 각 trial, episode 별로 현재위치 - 이전위치의 sqrt(dx**2+dy**2+dz**2) 를 더해서   거리를 return 해야 함
# manipulability index는 min, max, mean 값을 표로 나타내기

def get_plot_components(exp, method, filename):
    f = './result/'+exp+'/'+method+'/'+method+'_'+filename+'.npy'
    datasets = np.load(f, encoding='bytes')
    
    
    name = 'real_cur_pos'
    index = np.min([np.asarray(datasets[0][name]).shape[0], np.asarray(datasets[1][name]).shape[0],np.asarray(datasets[2][name]).shape[0]])

    trans_mean_x = np.mean([np.asarray(datasets[0][name])[:index,0], np.asarray(datasets[1][name])[:index,0], np.asarray(datasets[2][name])[:index,0]],0)
    trans_mean_y = np.mean([np.asarray(datasets[0][name])[:index,1], np.asarray(datasets[1][name])[:index,1], np.asarray(datasets[2][name])[:index,1]],0)
    trans_mean_z = np.mean([np.asarray(datasets[0][name])[:index,2], np.asarray(datasets[1][name])[:index,2], np.asarray(datasets[2][name])[:index,2]],0)
    
    trans_std_x = np.std([np.asarray(datasets[0][name])[:index,0], np.asarray(datasets[1][name])[:index,0], np.asarray(datasets[2][name])[:index,0]],0)
    trans_std_y = np.std([np.asarray(datasets[0][name])[:index,1], np.asarray(datasets[1][name])[:index,1], np.asarray(datasets[2][name])[:index,1]],0)
    trans_std_z = np.std([np.asarray(datasets[0][name])[:index,2], np.asarray(datasets[1][name])[:index,2], np.asarray(datasets[2][name])[:index,2]],0)
    
    orient_mean_x = np.mean([np.asarray(datasets[0][name])[:index,3], np.asarray(datasets[1][name])[:index,3], np.asarray(datasets[2][name])[:index,3]],0)
    orient_mean_y = np.mean([np.asarray(datasets[0][name])[:index,4], np.asarray(datasets[1][name])[:index,4], np.asarray(datasets[2][name])[:index,4]],0)
    orient_mean_z = np.mean([np.asarray(datasets[0][name])[:index,5], np.asarray(datasets[1][name])[:index,5], np.asarray(datasets[2][name])[:index,5]],0)
    
    orient_std_x = np.std([np.asarray(datasets[0][name])[:index,3], np.asarray(datasets[1][name])[:index,3], np.asarray(datasets[2][name])[:index,3]],0)
    orient_std_y = np.std([np.asarray(datasets[0][name])[:index,4], np.asarray(datasets[1][name])[:index,4], np.asarray(datasets[2][name])[:index,4]],0)
    orient_std_z = np.std([np.asarray(datasets[0][name])[:index,5], np.asarray(datasets[1][name])[:index,5], np.asarray(datasets[2][name])[:index,5]],0)
    
    name = 'real_m_index'
    real_m_index_mean = np.mean([np.asarray(datasets[0][name][:index]), np.asarray(datasets[1][name][:index]), np.asarray(datasets[2][name][:index])],0)
    real_m_index_std =  np.std([np.asarray(datasets[0][name][:index]), np.asarray(datasets[1][name][:index]), np.asarray(datasets[2][name][:index])],0)
    return index, [trans_mean_x, trans_mean_y, trans_mean_z], [trans_std_x, trans_std_y, trans_std_z], [orient_mean_x, orient_mean_y, orient_mean_z], [orient_std_x, orient_std_y, orient_std_z], real_m_index_mean, real_m_index_std


### 
def main():

    exp = 'exp2'
    method = 'naive'
    
    _index = []
    _trans_mean = []
    _trans_std = []
    _orient_mean = []
    _orient_std = []
    _real_m_index_mean = []
    _real_m_index_std = []
    
    filename = 'orient_try1_manip00_horizon1'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip05_horizon1'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip10_horizon1'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip00_horizon3'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip05_horizon3'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip10_horizon3'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip00_horizon5'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip05_horizon5'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip10_horizon5'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    '''
    filename = 'orient_try1_manip00_horizon7'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip05_horizon7'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    
    filename = 'orient_try1_manip10_horizon7'
    index, trans_mean, trans_std, orient_mean, orient_std, real_m_index_mean, real_m_index_std  = get_plot_components(exp, method, filename)
    _index.append(index)
    _trans_mean.append(trans_mean)
    _trans_std.append(trans_std)
    _orient_mean.append(orient_mean)
    _orient_std.append(orient_std)
    _real_m_index_mean.append(real_m_index_mean)
    _real_m_index_std.append(real_m_index_std)
    '''
    
    # try1
    #init_pose = [0.306, 0.809, 0.544, 0.040, 1.418, 1.512]
    #goal_pose = [-0.378, 0.584, 0.618, 0.188, 1.511, 1.286]

    # try2
    #init_pose = [-0.622, 0.456, 0.337, 0.090, 1.017, 1.588]    
    #goal_pose = [-0.242, 0.366, 0.752, -0.019, 0.737, 1.708]
    
    # try3
    #init_pose = [-0.659, 0.429, 0.448, 0.087, 0.636, 1.632]
    #goal_pose = [0.783, 0.792, 0.669, -0.180, 0.693, 1.547]
    
    # try4
    #init_pose = [0.178, 0.395, 0.376, -0.308, 0.851, 1.561]
    #goal_pose = [0.447, 0.585, 0.738, -0.094, 0.888, 1.448]

    # try1
    #init_pose = [-0.317, 0.593, 0.300, -0.316, 0.912, 1.430]
    #goal_pose = [-0.510, 0.419, 0.818, -0.317, 1.088, 1.613]

    fig = plt.figure(figsize=(30,30))
    fig.suptitle('Naive learning method, experiment 1 \n init_pose = [0.306, 0.809, 0.544, 0.040, 1.418, 1.512] \n goal_pose = [-0.378, 0.584, 0.618, 0.188, 1.511, 1.286]', fontsize=14)

    ax1 = fig.add_subplot(3,3,1, projection='3d')
    ax1.set_title('trajectory (translation)', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    ax1.set_xlim([-0.8,0.8])
    ax1.set_ylim([0.3,0.85])
    ax1.set_zlim([0.3,0.85])


    ax1.plot(_trans_mean[0][0],  _trans_mean[0][1],  _trans_mean[0][2], lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax1.plot(_trans_mean[1][0],  _trans_mean[1][1],  _trans_mean[1][2], lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax1.plot(_trans_mean[2][0],  _trans_mean[2][1],  _trans_mean[2][2], lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax1.plot(_trans_mean[3][0],  _trans_mean[3][1],  _trans_mean[3][2], lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax1.plot(_trans_mean[4][0],  _trans_mean[4][1],  _trans_mean[4][2], lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax1.plot(_trans_mean[5][0],  _trans_mean[5][1],  _trans_mean[5][2], lw=2, label = 'time horizon = 3, manip_coeff = 1.0')  
    ax1.plot(_trans_mean[6][0],  _trans_mean[6][1],  _trans_mean[6][2], lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax1.plot(_trans_mean[7][0],  _trans_mean[7][1],  _trans_mean[7][2], lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax1.plot(_trans_mean[8][0],  _trans_mean[8][1],  _trans_mean[8][2], lw=2, label = 'time horizon = 5, manip_coeff = 1.0')  
    #ax1.plot(_trans_mean[9][0],  _trans_mean[9][1],  _trans_mean[9][2], lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax1.plot(_trans_mean[10][0], _trans_mean[10][1], _trans_mean[10][2],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax1.plot(_trans_mean[11][0], _trans_mean[11][1], _trans_mean[11][2],lw=2, label = 'time horizon = 7, manip_coeff = 1.0') 
    #ax1.legend(prop={'size': 10})
    

    ax2 = fig.add_subplot(3,3,2, projection='3d')
    ax2.set_title('trajectory (orientation)', fontsize=14)
    ax2.set_xlabel('roll')
    ax2.set_ylabel('pitch')
    ax2.set_zlabel('yaw')
    
    ax2.plot(_orient_mean[0][0],  _orient_mean[0][1],  _orient_mean[0][2], lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax2.plot(_orient_mean[1][0],  _orient_mean[1][1],  _orient_mean[1][2], lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax2.plot(_orient_mean[2][0],  _orient_mean[2][1],  _orient_mean[2][2], lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax2.plot(_orient_mean[3][0],  _orient_mean[3][1],  _orient_mean[3][2], lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax2.plot(_orient_mean[4][0],  _orient_mean[4][1],  _orient_mean[4][2], lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax2.plot(_orient_mean[5][0],  _orient_mean[5][1],  _orient_mean[5][2], lw=2, label = 'time horizon = 3, manip_coeff = 1.0') 
    ax2.plot(_orient_mean[6][0],  _orient_mean[6][1],  _orient_mean[6][2], lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax2.plot(_orient_mean[7][0],  _orient_mean[7][1],  _orient_mean[7][2], lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax2.plot(_orient_mean[8][0],  _orient_mean[8][1],  _orient_mean[8][2], lw=2, label = 'time horizon = 5, manip_coeff = 1.0')
    #ax2.plot(_orient_mean[9][0],  _orient_mean[9][1],  _orient_mean[9][2], lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax2.plot(_orient_mean[10][0], _orient_mean[10][1], _orient_mean[10][2],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax2.plot(_orient_mean[11][0], _orient_mean[11][1], _orient_mean[11][2],lw=2, label = 'time horizon = 7, manip_coeff = 1.0')  
    #ax2.legend(prop={'size': 10})

    ax3 = fig.add_subplot(3,3,3)
    ax3.set_title('manipulability index', fontsize=14)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('manipulability index')
    ax3.plot(np.linspace(0,_index[0]-1, _index[0])*0.075, _real_m_index_mean[0],  lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax3.plot(np.linspace(0,_index[1]-1, _index[1])*0.075, _real_m_index_mean[1],  lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax3.plot(np.linspace(0,_index[2]-1, _index[2])*0.075, _real_m_index_mean[2],  lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax3.plot(np.linspace(0,_index[3]-1, _index[3])*0.075, _real_m_index_mean[3],  lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax3.plot(np.linspace(0,_index[4]-1, _index[4])*0.075, _real_m_index_mean[4],  lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax3.plot(np.linspace(0,_index[5]-1, _index[5])*0.075, _real_m_index_mean[5],  lw=2, label = 'time horizon = 3, manip_coeff = 1.0') 
    ax3.plot(np.linspace(0,_index[6]-1, _index[6])*0.075, _real_m_index_mean[6],  lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax3.plot(np.linspace(0,_index[7]-1, _index[7])*0.075, _real_m_index_mean[7],  lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax3.plot(np.linspace(0,_index[8]-1, _index[8])*0.075, _real_m_index_mean[8],  lw=2, label = 'time horizon = 5, manip_coeff = 1.0') 
    #ax3.plot(np.linspace(0,_index[9]-1, _index[9])*0.075, _real_m_index_mean[9] ,lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax3.plot(np.linspace(0,_index[10]-1,_index[10])*0.075,_real_m_index_mean[10],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax3.plot(np.linspace(0,_index[11]-1,_index[11])*0.075,_real_m_index_mean[11],lw=2, label = 'time horizon = 7, manip_coeff = 1.0') 
    
    ax3.fill_between(np.linspace(0,_index[0]-1, _index[0])*0.075,  _real_m_index_mean[0]  - _real_m_index_std[0],  _real_m_index_mean[0]  + _real_m_index_std[0] , alpha=0.4)    
    ax3.fill_between(np.linspace(0,_index[1]-1, _index[1])*0.075,  _real_m_index_mean[1]  - _real_m_index_std[1],  _real_m_index_mean[1]  + _real_m_index_std[1] , alpha=0.4)
    ax3.fill_between(np.linspace(0,_index[2]-1, _index[2])*0.075,  _real_m_index_mean[2]  - _real_m_index_std[2],  _real_m_index_mean[2]  + _real_m_index_std[2] , alpha=0.4)
    ax3.fill_between(np.linspace(0,_index[3]-1, _index[3])*0.075,  _real_m_index_mean[3]  - _real_m_index_std[3],  _real_m_index_mean[3]  + _real_m_index_std[3] , alpha=0.4)
    ax3.fill_between(np.linspace(0,_index[4]-1, _index[4])*0.075,  _real_m_index_mean[4]  - _real_m_index_std[4],  _real_m_index_mean[4]  + _real_m_index_std[4] , alpha=0.4)
    ax3.fill_between(np.linspace(0,_index[5]-1, _index[5])*0.075,  _real_m_index_mean[5]  - _real_m_index_std[5],  _real_m_index_mean[5]  + _real_m_index_std[5] , alpha=0.4)
    ax3.fill_between(np.linspace(0,_index[6]-1, _index[6])*0.075,  _real_m_index_mean[6]  - _real_m_index_std[6],  _real_m_index_mean[6]  + _real_m_index_std[6] , alpha=0.4)
    ax3.fill_between(np.linspace(0,_index[7]-1, _index[7])*0.075,  _real_m_index_mean[7]  - _real_m_index_std[7],  _real_m_index_mean[7]  + _real_m_index_std[7] , alpha=0.4)    
    ax3.fill_between(np.linspace(0,_index[8]-1, _index[8])*0.075,  _real_m_index_mean[8]  - _real_m_index_std[8],  _real_m_index_mean[8]  + _real_m_index_std[8] , alpha=0.4)
    #ax3.fill_between(np.linspace(0,_index[9]-1, _index[9])*0.075,  _real_m_index_mean[9]  - _real_m_index_std[9],  _real_m_index_mean[9]  + _real_m_index_std[9] , alpha=0.4)
    #ax3.fill_between(np.linspace(0,_index[10]-1,_index[10])*0.075, _real_m_index_mean[10] - _real_m_index_std[10], _real_m_index_mean[10] + _real_m_index_std[10], alpha=0.4)    
    #ax3.fill_between(np.linspace(0,_index[11]-1,_index[11])*0.075, _real_m_index_mean[11] - _real_m_index_std[11], _real_m_index_mean[11] + _real_m_index_std[11], alpha=0.4)
    ax3.legend(prop={'size': 10})

    ax4 = fig.add_subplot(3,3,4)
    ax4.set_title('trajectory (x)', fontsize=14)
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('x (m)')
    ax4.plot(np.linspace(0,_index[0]-1, _index[0])*0.075, _trans_mean[0][0], lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax4.plot(np.linspace(0,_index[1]-1, _index[1])*0.075, _trans_mean[1][0], lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax4.plot(np.linspace(0,_index[2]-1, _index[2])*0.075, _trans_mean[2][0], lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax4.plot(np.linspace(0,_index[3]-1, _index[3])*0.075, _trans_mean[3][0], lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax4.plot(np.linspace(0,_index[4]-1, _index[4])*0.075, _trans_mean[4][0], lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax4.plot(np.linspace(0,_index[5]-1, _index[5])*0.075, _trans_mean[5][0], lw=2, label = 'time horizon = 3, manip_coeff = 1.0') 
    ax4.plot(np.linspace(0,_index[6]-1, _index[6])*0.075, _trans_mean[6][0], lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax4.plot(np.linspace(0,_index[7]-1, _index[7])*0.075, _trans_mean[7][0], lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax4.plot(np.linspace(0,_index[8]-1, _index[8])*0.075, _trans_mean[8][0], lw=2, label = 'time horizon = 5, manip_coeff = 1.0') 
    #ax4.plot(np.linspace(0,_index[9]-1, _index[9])*0.075, _trans_mean[9][0], lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax4.plot(np.linspace(0,_index[10]-1,_index[10])*0.075,_trans_mean[10][0],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax4.plot(np.linspace(0,_index[11]-1,_index[11])*0.075,_trans_mean[11][0],lw=2, label = 'time horizon = 7, manip_coeff = 1.0') 
    
    
    ax4.fill_between(np.linspace(0,_index[0]-1, _index[0])*0.075,  _trans_mean[0][0]  - _trans_std[0][0],  _trans_mean[0][0]  + _trans_std[0][0] , alpha=0.4)    
    ax4.fill_between(np.linspace(0,_index[1]-1, _index[1])*0.075,  _trans_mean[1][0]  - _trans_std[1][0],  _trans_mean[1][0]  + _trans_std[1][0] , alpha=0.4)
    ax4.fill_between(np.linspace(0,_index[2]-1, _index[2])*0.075,  _trans_mean[2][0]  - _trans_std[2][0],  _trans_mean[2][0]  + _trans_std[2][0] , alpha=0.4)
    ax4.fill_between(np.linspace(0,_index[3]-1, _index[3])*0.075,  _trans_mean[3][0]  - _trans_std[3][0],  _trans_mean[3][0]  + _trans_std[3][0] , alpha=0.4)
    ax4.fill_between(np.linspace(0,_index[4]-1, _index[4])*0.075,  _trans_mean[4][0]  - _trans_std[4][0],  _trans_mean[4][0]  + _trans_std[4][0] , alpha=0.4)
    ax4.fill_between(np.linspace(0,_index[5]-1, _index[5])*0.075,  _trans_mean[5][0]  - _trans_std[5][0],  _trans_mean[5][0]  + _trans_std[5][0] , alpha=0.4)
    ax4.fill_between(np.linspace(0,_index[6]-1, _index[6])*0.075,  _trans_mean[6][0]  - _trans_std[6][0],  _trans_mean[6][0]  + _trans_std[6][0] , alpha=0.4)
    ax4.fill_between(np.linspace(0,_index[7]-1, _index[7])*0.075,  _trans_mean[7][0]  - _trans_std[7][0],  _trans_mean[7][0]  + _trans_std[7][0] , alpha=0.4)    
    ax4.fill_between(np.linspace(0,_index[8]-1, _index[8])*0.075,  _trans_mean[8][0]  - _trans_std[8][0],  _trans_mean[8][0]  + _trans_std[8][0] , alpha=0.4)
    #ax4.fill_between(np.linspace(0,_index[9]-1, _index[9])*0.075,  _trans_mean[9][0]  - _trans_std[9][0],  _trans_mean[9][0]  + _trans_std[9][0] , alpha=0.4)
    #ax4.fill_between(np.linspace(0,_index[10]-1,_index[10])*0.075, _trans_mean[10][0] - _trans_std[10][0], _trans_mean[10][0] + _trans_std[10][0], alpha=0.4)    
    #ax4.fill_between(np.linspace(0,_index[11]-1,_index[11])*0.075, _trans_mean[11][0] - _trans_std[11][0], _trans_mean[11][0] + _trans_std[11][0], alpha=0.4)
    ax4.legend(prop={'size': 10})
    
    ax5 = fig.add_subplot(3,3,5)
    ax5.set_title('trajectory (y)', fontsize=14)
    ax5.set_xlabel('time (s)')
    ax5.set_ylabel('y (m)')
    ax5.plot(np.linspace(0,_index[0]-1, _index[0])*0.075, _trans_mean[0][1], lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax5.plot(np.linspace(0,_index[1]-1, _index[1])*0.075, _trans_mean[1][1], lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax5.plot(np.linspace(0,_index[2]-1, _index[2])*0.075, _trans_mean[2][1], lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax5.plot(np.linspace(0,_index[3]-1, _index[3])*0.075, _trans_mean[3][1], lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax5.plot(np.linspace(0,_index[4]-1, _index[4])*0.075, _trans_mean[4][1], lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax5.plot(np.linspace(0,_index[5]-1, _index[5])*0.075, _trans_mean[5][1], lw=2, label = 'time horizon = 3, manip_coeff = 1.0') 
    ax5.plot(np.linspace(0,_index[6]-1, _index[6])*0.075, _trans_mean[6][1], lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax5.plot(np.linspace(0,_index[7]-1, _index[7])*0.075, _trans_mean[7][1], lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax5.plot(np.linspace(0,_index[8]-1, _index[8])*0.075, _trans_mean[8][1], lw=2, label = 'time horizon = 5, manip_coeff = 1.0') 
    #ax5.plot(np.linspace(0,_index[9]-1, _index[9])*0.075, _trans_mean[9][1], lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax5.plot(np.linspace(0,_index[10]-1,_index[10])*0.075,_trans_mean[10][1],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax5.plot(np.linspace(0,_index[11]-1,_index[11])*0.075,_trans_mean[11][1],lw=2, label = 'time horizon = 7, manip_coeff = 1.0') 
    
    ax5.fill_between(np.linspace(0,_index[0]-1, _index[0])*0.075,  _trans_mean[0][1]  - _trans_std[0][1],  _trans_mean[0][1]  + _trans_std[0][1] , alpha=0.4)    
    ax5.fill_between(np.linspace(0,_index[1]-1, _index[1])*0.075,  _trans_mean[1][1]  - _trans_std[1][1],  _trans_mean[1][1]  + _trans_std[1][1] , alpha=0.4)
    ax5.fill_between(np.linspace(0,_index[2]-1, _index[2])*0.075,  _trans_mean[2][1]  - _trans_std[2][1],  _trans_mean[2][1]  + _trans_std[2][1] , alpha=0.4)
    ax5.fill_between(np.linspace(0,_index[3]-1, _index[3])*0.075,  _trans_mean[3][1]  - _trans_std[3][1],  _trans_mean[3][1]  + _trans_std[3][1] , alpha=0.4)
    ax5.fill_between(np.linspace(0,_index[4]-1, _index[4])*0.075,  _trans_mean[4][1]  - _trans_std[4][1],  _trans_mean[4][1]  + _trans_std[4][1] , alpha=0.4)
    ax5.fill_between(np.linspace(0,_index[5]-1, _index[5])*0.075,  _trans_mean[5][1]  - _trans_std[5][1],  _trans_mean[5][1]  + _trans_std[5][1] , alpha=0.4)
    ax5.fill_between(np.linspace(0,_index[6]-1, _index[6])*0.075,  _trans_mean[6][1]  - _trans_std[6][1],  _trans_mean[6][1]  + _trans_std[6][1] , alpha=0.4)
    ax5.fill_between(np.linspace(0,_index[7]-1, _index[7])*0.075,  _trans_mean[7][1]  - _trans_std[7][1],  _trans_mean[7][1]  + _trans_std[7][1] , alpha=0.4)    
    ax5.fill_between(np.linspace(0,_index[8]-1, _index[8])*0.075,  _trans_mean[8][1]  - _trans_std[8][1],  _trans_mean[8][1]  + _trans_std[8][1] , alpha=0.4)
    #ax5.fill_between(np.linspace(0,_index[9]-1, _index[9])*0.075,  _trans_mean[9][1]  - _trans_std[9][1],  _trans_mean[9][1]  + _trans_std[9][1] , alpha=0.4)
    #ax5.fill_between(np.linspace(0,_index[10]-1,_index[10])*0.075, _trans_mean[10][1] - _trans_std[10][1], _trans_mean[10][1] + _trans_std[10][1], alpha=0.4)    
    #ax5.fill_between(np.linspace(0,_index[11]-1,_index[11])*0.075, _trans_mean[11][1] - _trans_std[11][1], _trans_mean[11][1] + _trans_std[11][1], alpha=0.4)
    ax5.legend(prop={'size': 10})

    ax6 = fig.add_subplot(3,3,6)
    ax6.set_title('trajectory (z)', fontsize=14)
    ax6.set_xlabel('time (s)')
    ax6.set_ylabel('z (m)')
    ax6.plot(np.linspace(0,_index[0]-1, _index[0])*0.075, _trans_mean[0][2], lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax6.plot(np.linspace(0,_index[1]-1, _index[1])*0.075, _trans_mean[1][2], lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax6.plot(np.linspace(0,_index[2]-1, _index[2])*0.075, _trans_mean[2][2], lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax6.plot(np.linspace(0,_index[3]-1, _index[3])*0.075, _trans_mean[3][2], lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax6.plot(np.linspace(0,_index[4]-1, _index[4])*0.075, _trans_mean[4][2], lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax6.plot(np.linspace(0,_index[5]-1, _index[5])*0.075, _trans_mean[5][2], lw=2, label = 'time horizon = 3, manip_coeff = 1.0') 
    ax6.plot(np.linspace(0,_index[6]-1, _index[6])*0.075, _trans_mean[6][2], lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax6.plot(np.linspace(0,_index[7]-1, _index[7])*0.075, _trans_mean[7][2], lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax6.plot(np.linspace(0,_index[8]-1, _index[8])*0.075, _trans_mean[8][2], lw=2, label = 'time horizon = 5, manip_coeff = 1.0') 
    #ax6.plot(np.linspace(0,_index[9]-1, _index[9])*0.075, _trans_mean[9][2], lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax6.plot(np.linspace(0,_index[10]-1,_index[10])*0.075,_trans_mean[10][2],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax6.plot(np.linspace(0,_index[11]-1,_index[11])*0.075,_trans_mean[11][2],lw=2, label = 'time horizon = 7, manip_coeff = 1.0') 
    
    ax6.fill_between(np.linspace(0,_index[0]-1, _index[0])*0.075,  _trans_mean[0][2]  - _trans_std[0][2],  _trans_mean[0][2]  + _trans_std[0][2] , alpha=0.4)    
    ax6.fill_between(np.linspace(0,_index[1]-1, _index[1])*0.075,  _trans_mean[1][2]  - _trans_std[1][2],  _trans_mean[1][2]  + _trans_std[1][2] , alpha=0.4)
    ax6.fill_between(np.linspace(0,_index[2]-1, _index[2])*0.075,  _trans_mean[2][2]  - _trans_std[2][2],  _trans_mean[2][2]  + _trans_std[2][2] , alpha=0.4)
    ax6.fill_between(np.linspace(0,_index[3]-1, _index[3])*0.075,  _trans_mean[3][2]  - _trans_std[3][2],  _trans_mean[3][2]  + _trans_std[3][2] , alpha=0.4)
    ax6.fill_between(np.linspace(0,_index[4]-1, _index[4])*0.075,  _trans_mean[4][2]  - _trans_std[4][2],  _trans_mean[4][2]  + _trans_std[4][2] , alpha=0.4)
    ax6.fill_between(np.linspace(0,_index[5]-1, _index[5])*0.075,  _trans_mean[5][2]  - _trans_std[5][2],  _trans_mean[5][2]  + _trans_std[5][2] , alpha=0.4)
    ax6.fill_between(np.linspace(0,_index[6]-1, _index[6])*0.075,  _trans_mean[6][2]  - _trans_std[6][2],  _trans_mean[6][2]  + _trans_std[6][2] , alpha=0.4)
    ax6.fill_between(np.linspace(0,_index[7]-1, _index[7])*0.075,  _trans_mean[7][2]  - _trans_std[7][2],  _trans_mean[7][2]  + _trans_std[7][2] , alpha=0.4)    
    ax6.fill_between(np.linspace(0,_index[8]-1, _index[8])*0.075,  _trans_mean[8][2]  - _trans_std[8][2],  _trans_mean[8][2]  + _trans_std[8][2] , alpha=0.4)
    #ax6.fill_between(np.linspace(0,_index[9]-1, _index[9])*0.075,  _trans_mean[9][2]  - _trans_std[9][2],  _trans_mean[9][2]  + _trans_std[9][2] , alpha=0.4)
    #ax6.fill_between(np.linspace(0,_index[10]-1,_index[10])*0.075, _trans_mean[10][2] - _trans_std[10][2], _trans_mean[10][2] + _trans_std[10][2], alpha=0.4)    
    #ax6.fill_between(np.linspace(0,_index[11]-1,_index[11])*0.075, _trans_mean[11][2] - _trans_std[11][2], _trans_mean[11][2] + _trans_std[11][2], alpha=0.4)
    ax6.legend(prop={'size': 10})
    
    ax7 = fig.add_subplot(3,3,7)
    ax7.set_title('trajectory (roll)', fontsize=14)
    ax7.set_xlabel('time (s)')
    ax7.set_ylabel('roll (rad)')
    ax7.plot(np.linspace(0,_index[0]-1, _index[0])*0.075, _orient_mean[0][0], lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax7.plot(np.linspace(0,_index[1]-1, _index[1])*0.075, _orient_mean[1][0], lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax7.plot(np.linspace(0,_index[2]-1, _index[2])*0.075, _orient_mean[2][0], lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax7.plot(np.linspace(0,_index[3]-1, _index[3])*0.075, _orient_mean[3][0], lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax7.plot(np.linspace(0,_index[4]-1, _index[4])*0.075, _orient_mean[4][0], lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax7.plot(np.linspace(0,_index[5]-1, _index[5])*0.075, _orient_mean[5][0], lw=2, label = 'time horizon = 3, manip_coeff = 1.0') 
    ax7.plot(np.linspace(0,_index[6]-1, _index[6])*0.075, _orient_mean[6][0], lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax7.plot(np.linspace(0,_index[7]-1, _index[7])*0.075, _orient_mean[7][0], lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax7.plot(np.linspace(0,_index[8]-1, _index[8])*0.075, _orient_mean[8][0], lw=2, label = 'time horizon = 5, manip_coeff = 1.0') 
    #ax7.plot(np.linspace(0,_index[9]-1, _index[9])*0.075, _orient_mean[9][0], lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax7.plot(np.linspace(0,_index[10]-1,_index[10])*0.075,_orient_mean[10][0],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax7.plot(np.linspace(0,_index[11]-1,_index[11])*0.075,_orient_mean[11][0],lw=2, label = 'time horizon = 7, manip_coeff = 1.0') 
    
    ax7.fill_between(np.linspace(0,_index[0]-1, _index[0])*0.075,  _orient_mean[0][0]  - _orient_std[0][0],  _orient_mean[0][0]  + _orient_std[0][0] , alpha=0.4)    
    ax7.fill_between(np.linspace(0,_index[1]-1, _index[1])*0.075,  _orient_mean[1][0]  - _orient_std[1][0],  _orient_mean[1][0]  + _orient_std[1][0] , alpha=0.4)
    ax7.fill_between(np.linspace(0,_index[2]-1, _index[2])*0.075,  _orient_mean[2][0]  - _orient_std[2][0],  _orient_mean[2][0]  + _orient_std[2][0] , alpha=0.4)
    ax7.fill_between(np.linspace(0,_index[3]-1, _index[3])*0.075,  _orient_mean[3][0]  - _orient_std[3][0],  _orient_mean[3][0]  + _orient_std[3][0] , alpha=0.4)
    ax7.fill_between(np.linspace(0,_index[4]-1, _index[4])*0.075,  _orient_mean[4][0]  - _orient_std[4][0],  _orient_mean[4][0]  + _orient_std[4][0] , alpha=0.4)
    ax7.fill_between(np.linspace(0,_index[5]-1, _index[5])*0.075,  _orient_mean[5][0]  - _orient_std[5][0],  _orient_mean[5][0]  + _orient_std[5][0] , alpha=0.4)
    ax7.fill_between(np.linspace(0,_index[6]-1, _index[6])*0.075,  _orient_mean[6][0]  - _orient_std[6][0],  _orient_mean[6][0]  + _orient_std[6][0] , alpha=0.4)
    ax7.fill_between(np.linspace(0,_index[7]-1, _index[7])*0.075,  _orient_mean[7][0]  - _orient_std[7][0],  _orient_mean[7][0]  + _orient_std[7][0] , alpha=0.4)    
    ax7.fill_between(np.linspace(0,_index[8]-1, _index[8])*0.075,  _orient_mean[8][0]  - _orient_std[8][0],  _orient_mean[8][0]  + _orient_std[8][0] , alpha=0.4)
    #ax7.fill_between(np.linspace(0,_index[9]-1, _index[9])*0.075,  _orient_mean[9][0]  - _orient_std[9][0],  _orient_mean[9][0]  + _orient_std[9][0] , alpha=0.4)
    #ax7.fill_between(np.linspace(0,_index[10]-1,_index[10])*0.075, _orient_mean[10][0] - _orient_std[10][0], _orient_mean[10][0] + _orient_std[10][0], alpha=0.4)    
    #ax7.fill_between(np.linspace(0,_index[11]-1,_index[11])*0.075, _orient_mean[11][0] - _orient_std[11][0], _orient_mean[11][0] + _orient_std[11][0], alpha=0.4)
    ax7.legend(prop={'size': 10})
    
    ax8 = fig.add_subplot(3,3,8)
    ax8.set_title('trajectory (pitch)', fontsize=14)
    ax8.set_xlabel('time (s)')
    ax8.set_ylabel('pitch (rad)')
    ax8.plot(np.linspace(0,_index[0]-1, _index[0])*0.075, _orient_mean[0][1], lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax8.plot(np.linspace(0,_index[1]-1, _index[1])*0.075, _orient_mean[1][1], lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax8.plot(np.linspace(0,_index[2]-1, _index[2])*0.075, _orient_mean[2][1], lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax8.plot(np.linspace(0,_index[3]-1, _index[3])*0.075, _orient_mean[3][1], lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax8.plot(np.linspace(0,_index[4]-1, _index[4])*0.075, _orient_mean[4][1], lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax8.plot(np.linspace(0,_index[5]-1, _index[5])*0.075, _orient_mean[5][1], lw=2, label = 'time horizon = 3, manip_coeff = 1.0') 
    ax8.plot(np.linspace(0,_index[6]-1, _index[6])*0.075, _orient_mean[6][1], lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax8.plot(np.linspace(0,_index[7]-1, _index[7])*0.075, _orient_mean[7][1], lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax8.plot(np.linspace(0,_index[8]-1, _index[8])*0.075, _orient_mean[8][1], lw=2, label = 'time horizon = 5, manip_coeff = 1.0') 
    #ax8.plot(np.linspace(0,_index[9]-1, _index[9])*0.075, _orient_mean[9][1], lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax8.plot(np.linspace(0,_index[10]-1,_index[10])*0.075,_orient_mean[10][1],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax8.plot(np.linspace(0,_index[11]-1,_index[11])*0.075,_orient_mean[11][1],lw=2, label = 'time horizon = 7, manip_coeff = 1.0') 
    
    ax8.fill_between(np.linspace(0,_index[0]-1, _index[0])*0.075,  _orient_mean[0][1]  - _orient_std[0][1],  _orient_mean[0][1]  + _orient_std[0][1] , alpha=0.4)    
    ax8.fill_between(np.linspace(0,_index[1]-1, _index[1])*0.075,  _orient_mean[1][1]  - _orient_std[1][1],  _orient_mean[1][1]  + _orient_std[1][1] , alpha=0.4)
    ax8.fill_between(np.linspace(0,_index[2]-1, _index[2])*0.075,  _orient_mean[2][1]  - _orient_std[2][1],  _orient_mean[2][1]  + _orient_std[2][1] , alpha=0.4)
    ax8.fill_between(np.linspace(0,_index[3]-1, _index[3])*0.075,  _orient_mean[3][1]  - _orient_std[3][1],  _orient_mean[3][1]  + _orient_std[3][1] , alpha=0.4)
    ax8.fill_between(np.linspace(0,_index[4]-1, _index[4])*0.075,  _orient_mean[4][1]  - _orient_std[4][1],  _orient_mean[4][1]  + _orient_std[4][1] , alpha=0.4)
    ax8.fill_between(np.linspace(0,_index[5]-1, _index[5])*0.075,  _orient_mean[5][1]  - _orient_std[5][1],  _orient_mean[5][1]  + _orient_std[5][1] , alpha=0.4)
    ax8.fill_between(np.linspace(0,_index[6]-1, _index[6])*0.075,  _orient_mean[6][1]  - _orient_std[6][1],  _orient_mean[6][1]  + _orient_std[6][1] , alpha=0.4)
    ax8.fill_between(np.linspace(0,_index[7]-1, _index[7])*0.075,  _orient_mean[7][1]  - _orient_std[7][1],  _orient_mean[7][1]  + _orient_std[7][1] , alpha=0.4)    
    ax8.fill_between(np.linspace(0,_index[8]-1, _index[8])*0.075,  _orient_mean[8][1]  - _orient_std[8][1],  _orient_mean[8][1]  + _orient_std[8][1] , alpha=0.4)
    #ax8.fill_between(np.linspace(0,_index[9]-1, _index[9])*0.075,  _orient_mean[9][1]  - _orient_std[9][1],  _orient_mean[9][1]  + _orient_std[9][1] , alpha=0.4)
    #ax8.fill_between(np.linspace(0,_index[10]-1,_index[10])*0.075, _orient_mean[10][1] - _orient_std[10][1], _orient_mean[10][1] + _orient_std[10][1], alpha=0.4)    
    #ax8.fill_between(np.linspace(0,_index[11]-1,_index[11])*0.075, _orient_mean[11][1] - _orient_std[11][1], _orient_mean[11][1] + _orient_std[11][1], alpha=0.4)
    ax8.legend(prop={'size': 10})
    
    ax9 = fig.add_subplot(3,3,9)
    ax9.set_title('trajectory (yaw)', fontsize=14)
    ax9.set_xlabel('time (s)')
    ax9.set_ylabel('yaw (rad)')
    ax9.plot(np.linspace(0,_index[0]-1, _index[0])*0.075, _orient_mean[0][2], lw=2, label = 'time horizon = 1, manip_coeff = 0.0') 
    ax9.plot(np.linspace(0,_index[1]-1, _index[1])*0.075, _orient_mean[1][2], lw=2, label = 'time horizon = 1, manip_coeff = 0.5') 
    ax9.plot(np.linspace(0,_index[2]-1, _index[2])*0.075, _orient_mean[2][2], lw=2, label = 'time horizon = 1, manip_coeff = 1.0') 
    ax9.plot(np.linspace(0,_index[3]-1, _index[3])*0.075, _orient_mean[3][2], lw=2, label = 'time horizon = 3, manip_coeff = 0.0') 
    ax9.plot(np.linspace(0,_index[4]-1, _index[4])*0.075, _orient_mean[4][2], lw=2, label = 'time horizon = 3, manip_coeff = 0.5') 
    ax9.plot(np.linspace(0,_index[5]-1, _index[5])*0.075, _orient_mean[5][2], lw=2, label = 'time horizon = 3, manip_coeff = 1.0') 
    ax9.plot(np.linspace(0,_index[6]-1, _index[6])*0.075, _orient_mean[6][2], lw=2, label = 'time horizon = 5, manip_coeff = 0.0') 
    ax9.plot(np.linspace(0,_index[7]-1, _index[7])*0.075, _orient_mean[7][2], lw=2, label = 'time horizon = 5, manip_coeff = 0.5') 
    ax9.plot(np.linspace(0,_index[8]-1, _index[8])*0.075, _orient_mean[8][2], lw=2, label = 'time horizon = 5, manip_coeff = 1.0') 
    #ax9.plot(np.linspace(0,_index[9]-1, _index[9])*0.075, _orient_mean[9][2], lw=2, label = 'time horizon = 7, manip_coeff = 0.0') 
    #ax9.plot(np.linspace(0,_index[10]-1,_index[10])*0.075,_orient_mean[10][2],lw=2, label = 'time horizon = 7, manip_coeff = 0.5') 
    #ax9.plot(np.linspace(0,_index[11]-1,_index[11])*0.075,_orient_mean[11][2],lw=2, label = 'time horizon = 7, manip_coeff = 1.0') 
    
    ax9.fill_between(np.linspace(0,_index[0]-1, _index[0])*0.075,  _orient_mean[0][2]  - _orient_std[0][2],  _orient_mean[0][2]  + _orient_std[0][2] , alpha=0.4)    
    ax9.fill_between(np.linspace(0,_index[1]-1, _index[1])*0.075,  _orient_mean[1][2]  - _orient_std[1][2],  _orient_mean[1][2]  + _orient_std[1][2] , alpha=0.4)
    ax9.fill_between(np.linspace(0,_index[2]-1, _index[2])*0.075,  _orient_mean[2][2]  - _orient_std[2][2],  _orient_mean[2][2]  + _orient_std[2][2] , alpha=0.4)
    ax9.fill_between(np.linspace(0,_index[3]-1, _index[3])*0.075,  _orient_mean[3][2]  - _orient_std[3][2],  _orient_mean[3][2]  + _orient_std[3][2] , alpha=0.4)
    ax9.fill_between(np.linspace(0,_index[4]-1, _index[4])*0.075,  _orient_mean[4][2]  - _orient_std[4][2],  _orient_mean[4][2]  + _orient_std[4][2] , alpha=0.4)
    ax9.fill_between(np.linspace(0,_index[5]-1, _index[5])*0.075,  _orient_mean[5][2]  - _orient_std[5][2],  _orient_mean[5][2]  + _orient_std[5][2] , alpha=0.4)
    ax9.fill_between(np.linspace(0,_index[6]-1, _index[6])*0.075,  _orient_mean[6][2]  - _orient_std[6][2],  _orient_mean[6][2]  + _orient_std[6][2] , alpha=0.4)
    ax9.fill_between(np.linspace(0,_index[7]-1, _index[7])*0.075,  _orient_mean[7][2]  - _orient_std[7][2],  _orient_mean[7][2]  + _orient_std[7][2] , alpha=0.4)    
    ax9.fill_between(np.linspace(0,_index[8]-1, _index[8])*0.075,  _orient_mean[8][2]  - _orient_std[8][2],  _orient_mean[8][2]  + _orient_std[8][2] , alpha=0.4)
    #ax9.fill_between(np.linspace(0,_index[9]-1, _index[9])*0.075,  _orient_mean[9][2]  - _orient_std[9][2],  _orient_mean[9][2]  + _orient_std[9][2] , alpha=0.4)
    #ax9.fill_between(np.linspace(0,_index[10]-1,_index[10])*0.075, _orient_mean[10][2] - _orient_std[10][2], _orient_mean[10][2] + _orient_std[10][2], alpha=0.4)    
    #ax9.fill_between(np.linspace(0,_index[11]-1,_index[11])*0.075, _orient_mean[11][2] - _orient_std[11][2], _orient_mean[11][2] + _orient_std[11][2], alpha=0.4)
    ax9.legend(prop={'size': 10})
    
    plt.tight_layout(h_pad=0.5, rect=(0,0,1.0,0.96))
    fig.savefig('./fig/naive_try1.png')
    fig.savefig('./fig/naive_try1.pdf',dpi=600)
    
    
    plt.show()
if __name__ == '__main__':
    main()