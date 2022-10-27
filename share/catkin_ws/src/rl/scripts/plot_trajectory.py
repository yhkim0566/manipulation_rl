#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 


import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D




def main():
    datasets = np.load('./result/4 obstacle cost 005/obstacle_avoidance_naive1.npy', encoding='bytes')
    dataset1 = datasets[0]
    
    datasets = np.load('./result/4 obstacle cost 005/obstacle_avoidance_deriv1.npy', encoding='bytes')
    dataset2 = datasets[0]
    
    datasets = np.load('./result/4 obstacle cost 005/obstacle_avoidance_naive5.npy', encoding='bytes')
    dataset3 = datasets[0]
    
    datasets = np.load('./result/4 obstacle cost 005/obstacle_avoidance_deriv5.npy', encoding='bytes')
    dataset4 = datasets[0]
    
    datasets = np.load('./result/obstacle cost 00/obstacle_avoidance_naive1.npy', encoding='bytes')
    dataset5 = datasets[0]
    
    datasets = np.load('./result/obstacle cost 00/obstacle_avoidance_deriv1.npy', encoding='bytes')
    dataset6 = datasets[0]
    
    datasets = np.load('./result/obstacle cost 00/obstacle_avoidance_naive5.npy', encoding='bytes')
    dataset7 = datasets[0]
    
    datasets = np.load('./result/obstacle cost 00/obstacle_avoidance_deriv5.npy', encoding='bytes')
    dataset8 = datasets[0]
    
    r = 0.005
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)  
    x1 = -0.3799078443019837+ np.outer(np.cos(u), np.sin(v))*r
    y1 = 0.532240294234275  + np.outer(np.sin(u), np.sin(v))*r
    z1 = 0.4244268861275426 + np.outer(np.ones(np.size(u)), np.cos(v))*r
    
    x2 = -0.3636915453041721+ np.outer(np.cos(u), np.sin(v))*r
    y2 = 0.548578721402501  + np.outer(np.sin(u), np.sin(v))*r
    z2 = 0.4377162452901069 + np.outer(np.ones(np.size(u)), np.cos(v))*r

    x3 = -0.3806772874854791+ np.outer(np.cos(u), np.sin(v))*r
    y3 = 0.527353975259506  + np.outer(np.sin(u), np.sin(v))*r
    z3 = 0.4187225834059944 + np.outer(np.ones(np.size(u)), np.cos(v))*r
    
    x4 = -0.3718779195870111+ np.outer(np.cos(u), np.sin(v))*r
    y4 = 0.540325984871439  + np.outer(np.sin(u), np.sin(v))*r
    z4 = 0.4301771889978318 + np.outer(np.ones(np.size(u)), np.cos(v))*r
    obs1 = np.array([-0.3799078443019837, 0.532240294234275, 0.4244268861275426])
    obs2 = np.array([-0.3636915453041721, 0.548578721402501, 0.4377162452901069])
    obs3 = np.array([-0.3806772874854791, 0.527353975259506, 0.4187225834059944])
    obs4 = np.array([-0.3718779195870111, 0.540325984871439, 0.4301771889978318])
    print(np.min(np.sqrt((np.asarray(dataset1['real_cur_pos'])[:,0]-obs1[0])**2+ (np.asarray(dataset1['real_cur_pos'])[:,1]-obs1[1])**2+(np.asarray(dataset1['real_cur_pos'])[:,2]-obs1[2])**2)))
    print(np.min(np.sqrt((np.asarray(dataset2['real_cur_pos'])[:,0]-obs2[0])**2+ (np.asarray(dataset2['real_cur_pos'])[:,1]-obs2[1])**2+(np.asarray(dataset2['real_cur_pos'])[:,2]-obs2[2])**2)))
    print(np.min(np.sqrt((np.asarray(dataset3['real_cur_pos'])[:,0]-obs3[0])**2+ (np.asarray(dataset3['real_cur_pos'])[:,1]-obs3[1])**2+(np.asarray(dataset3['real_cur_pos'])[:,2]-obs3[2])**2)))
    print(np.min(np.sqrt((np.asarray(dataset4['real_cur_pos'])[:,0]-obs4[0])**2+ (np.asarray(dataset4['real_cur_pos'])[:,1]-obs4[1])**2+(np.asarray(dataset4['real_cur_pos'])[:,2]-obs4[2])**2)))
    
    fig = plt.figure(figsize=(30,20))
    fig.suptitle('Obstacle avoidance', fontsize=24)


    ax1 = fig.add_subplot(2,3,2, projection='3d')
    ax1.set_title('part of trajectory (view1)', fontsize=18)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.view_init(30, 30)
    ax1.set_xlim([-0.34, -0.40])
    ax1.set_ylim([0.50, 0.56])
    ax1.set_zlim([0.39, 0.45])
    ax1.plot_surface(x1, y1, z1, color='r')
    ax1.plot_surface(x2, y2, z2, color='r')
    ax1.plot_surface(x3, y3, z3, color='b')
    ax1.plot_surface(x4, y4, z4, color='b')
    #ax1.scatter(-0.3799078443019837, 0.532240294234275, 0.4244268861275426 , s = 100)
    #ax1.scatter(-0.3636915453041721, 0.548578721402501, 0.4377162452901069 , s = 100)
    #ax1.scatter(-0.3806772874854791, 0.527353975259506, 0.4187225834059944 , s = 100)
    #ax1.scatter(-0.3718779195870111, 0.540325984871439, 0.4301771889978318 , s = 100)
    ax1.scatter(-0.5106587048041216, 0.419292690922957, 0.8189151099771221 , s = 300, color = 'k', label='goal')
    ax1.plot(np.asarray(dataset5['real_cur_pos'])[65:135,0],np.asarray(dataset5['real_cur_pos'])[65:135,1],np.asarray(dataset5['real_cur_pos'])[65:135,2], lw=3, label='naive(horizon1)',color='r', ls='--')
    ax1.plot(np.asarray(dataset6['real_cur_pos'])[65:125,0],np.asarray(dataset6['real_cur_pos'])[65:125,1],np.asarray(dataset6['real_cur_pos'])[65:125,2], lw=3, label='deriv(horizon1)',color='r')
    ax1.plot(np.asarray(dataset7['real_cur_pos'])[65:135,0],np.asarray(dataset7['real_cur_pos'])[65:135,1],np.asarray(dataset7['real_cur_pos'])[65:135,2], lw=3, label='naive(horizon5)',color='b',ls='--')
    ax1.plot(np.asarray(dataset8['real_cur_pos'])[65:135,0],np.asarray(dataset8['real_cur_pos'])[65:135,1],np.asarray(dataset8['real_cur_pos'])[65:135,2], lw=3, label='deriv(horizon5)',color='b')
    
    ax2 = fig.add_subplot(2,3,3, projection='3d')
    ax2.set_title('part of trajectory (view2)', fontsize=18)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.view_init(30, 220)
    ax2.set_xlim([-0.34, -0.40])
    ax2.set_ylim([0.50, 0.56])
    ax2.set_zlim([0.39, 0.45])
    ax2.plot_surface(x1, y1, z1, color='r')
    ax2.plot_surface(x2, y2, z2, color='r')
    ax2.plot_surface(x3, y3, z3, color='b')
    ax2.plot_surface(x4, y4, z4, color='b')
    #ax2.scatter(-0.3799078443019837, 0.532240294234275, 0.4244268861275426 , s = 100)
    #ax2.scatter(-0.3636915453041721, 0.548578721402501, 0.4377162452901069 , s = 100)
    #ax2.scatter(-0.3806772874854791, 0.527353975259506, 0.4187225834059944 , s = 100)
    #ax2.scatter(-0.3718779195870111, 0.540325984871439, 0.4301771889978318 , s = 100)
    ax2.scatter(-0.5106587048041216, 0.419292690922957, 0.8189151099771221 , s = 300, color = 'k', label='goal')
    ax2.plot(np.asarray(dataset5['real_cur_pos'])[60:123,0],np.asarray(dataset5['real_cur_pos'])[60:123,1],np.asarray(dataset5['real_cur_pos'])[60:123,2], lw=3, label='naive(horizon1)',color='r', ls='--')
    ax2.plot(np.asarray(dataset6['real_cur_pos'])[60:123,0],np.asarray(dataset6['real_cur_pos'])[60:123,1],np.asarray(dataset6['real_cur_pos'])[60:123,2], lw=3, label='deriv(horizon1)',color='r')
    ax2.plot(np.asarray(dataset7['real_cur_pos'])[60:123,0],np.asarray(dataset7['real_cur_pos'])[60:123,1],np.asarray(dataset7['real_cur_pos'])[60:123,2], lw=3, label='naive(horizon5)',color='b',ls='--')
    ax2.plot(np.asarray(dataset8['real_cur_pos'])[60:123,0],np.asarray(dataset8['real_cur_pos'])[60:123,1],np.asarray(dataset8['real_cur_pos'])[60:123,2], lw=3, label='deriv(horizon5)',color='b')
    
    ax3 = fig.add_subplot(2,3,1, projection='3d')
    ax3.set_title('Whole trajectory (without obstacle cost)', fontsize=18)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.view_init(60, 330)
    ax3.plot_surface(x1, y1, z1, color='r')
    ax3.plot_surface(x2, y2, z2, color='r')
    ax3.plot_surface(x3, y3, z3, color='b')
    ax3.plot_surface(x4, y4, z4, color='b')
    #ax3.scatter(-0.3799078443019837, 0.532240294234275, 0.4244268861275426, s = 100)
    #ax3.scatter(-0.3636915453041721, 0.548578721402501, 0.4377162452901069, s = 100)
    #ax3.scatter(-0.3806772874854791, 0.527353975259506, 0.4187225834059944, s = 100)
    #ax3.scatter(-0.3718779195870111, 0.540325984871439, 0.4301771889978318, s = 100)
    ax3.scatter(-0.5106587048041216, 0.419292690922957, 0.8189151099771221, s = 300, color = 'k', label='goal')
    ax3.plot(np.asarray(dataset5['real_cur_pos'])[:,0],np.asarray(dataset5['real_cur_pos'])[:,1],np.asarray(dataset5['real_cur_pos'])[:,2], lw=3, label='naive(horizon1)',color='r', ls='--')
    ax3.plot(np.asarray(dataset6['real_cur_pos'])[:,0],np.asarray(dataset6['real_cur_pos'])[:,1],np.asarray(dataset6['real_cur_pos'])[:,2], lw=3, label='deriv(horizon1)',color='r')
    ax3.plot(np.asarray(dataset7['real_cur_pos'])[:,0],np.asarray(dataset7['real_cur_pos'])[:,1],np.asarray(dataset7['real_cur_pos'])[:,2], lw=3, label='naive(horizon5)',color='b',ls='--')
    ax3.plot(np.asarray(dataset8['real_cur_pos'])[:,0],np.asarray(dataset8['real_cur_pos'])[:,1],np.asarray(dataset8['real_cur_pos'])[:,2], lw=3, label='deriv(horizon5)',color='b')
    plt.legend(fontsize=15)
    
    ax4 = fig.add_subplot(2,3,5, projection='3d')
    ax4.set_title('part of trajectory (view1)', fontsize=18)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.view_init(30, 30)
    ax4.set_xlim([-0.34, -0.40])
    ax4.set_ylim([0.50, 0.56])
    ax4.set_zlim([0.39, 0.45])
    ax4.plot_surface(x1, y1, z1, color='r')
    ax4.plot_surface(x2, y2, z2, color='r')
    ax4.plot_surface(x3, y3, z3, color='b')
    ax4.plot_surface(x4, y4, z4, color='b')
    #ax4.scatter(-0.3799078443019837, 0.532240294234275, 0.4244268861275426 , s = 100)
    #ax4.scatter(-0.3636915453041721, 0.548578721402501, 0.4377162452901069 , s = 100)
    #ax4.scatter(-0.3806772874854791, 0.527353975259506, 0.4187225834059944 , s = 100)
    #ax4.scatter(-0.3718779195870111, 0.540325984871439, 0.4301771889978318 , s = 100)
    ax4.scatter(-0.5106587048041216, 0.419292690922957, 0.8189151099771221 , s = 300, color = 'k', label='goal')
    ax4.plot(np.asarray(dataset1['real_cur_pos'])[65:135,0],np.asarray(dataset1['real_cur_pos'])[65:135,1],np.asarray(dataset1['real_cur_pos'])[65:135,2], lw=3, label='naive(horizon1)',color='r', ls='--')
    ax4.plot(np.asarray(dataset2['real_cur_pos'])[65:125,0],np.asarray(dataset2['real_cur_pos'])[65:125,1],np.asarray(dataset2['real_cur_pos'])[65:125,2], lw=3, label='deriv(horizon1)',color='r')
    ax4.plot(np.asarray(dataset3['real_cur_pos'])[65:135,0],np.asarray(dataset3['real_cur_pos'])[65:135,1],np.asarray(dataset3['real_cur_pos'])[65:135,2], lw=3, label='naive(horizon5)',color='b',ls='--')
    ax4.plot(np.asarray(dataset4['real_cur_pos'])[65:130,0],np.asarray(dataset4['real_cur_pos'])[65:130,1],np.asarray(dataset4['real_cur_pos'])[65:130,2], lw=3, label='deriv(horizon5)',color='b')
    
    ax5 = fig.add_subplot(2,3,6, projection='3d')
    ax5.set_title('part of trajectory (view2)', fontsize=18)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')
    ax5.view_init(30, 220)
    ax5.set_xlim([-0.34, -0.40])
    ax5.set_ylim([0.50, 0.56])
    ax5.set_zlim([0.39, 0.45])
    ax5.plot_surface(x1, y1, z1, color='r')
    ax5.plot_surface(x2, y2, z2, color='r')
    ax5.plot_surface(x3, y3, z3, color='b')
    ax5.plot_surface(x4, y4, z4, color='b')
    #ax5.scatter(-0.3799078443019837, 0.532240294234275, 0.4244268861275426 , s = 100)
    #ax5.scatter(-0.3636915453041721, 0.548578721402501, 0.4377162452901069 , s = 100)
    #ax5.scatter(-0.3806772874854791, 0.527353975259506, 0.4187225834059944 , s = 100)
    #ax5.scatter(-0.3718779195870111, 0.540325984871439, 0.4301771889978318 , s = 100)
    ax5.scatter(-0.5106587048041216, 0.419292690922957, 0.8189151099771221 , s = 300, color = 'k', label='goal')
    ax5.plot(np.asarray(dataset1['real_cur_pos'])[60:125,0],np.asarray(dataset1['real_cur_pos'])[60:125,1],np.asarray(dataset1['real_cur_pos'])[60:125,2], lw=3, label='naive(horizon1)',color='r', ls='--')
    ax5.plot(np.asarray(dataset2['real_cur_pos'])[60:118,0],np.asarray(dataset2['real_cur_pos'])[60:118,1],np.asarray(dataset2['real_cur_pos'])[60:118,2], lw=3, label='deriv(horizon1)',color='r')
    ax5.plot(np.asarray(dataset3['real_cur_pos'])[60:125,0],np.asarray(dataset3['real_cur_pos'])[60:125,1],np.asarray(dataset3['real_cur_pos'])[60:125,2], lw=3, label='naive(horizon5)',color='b',ls='--')
    ax5.plot(np.asarray(dataset4['real_cur_pos'])[60:125,0],np.asarray(dataset4['real_cur_pos'])[60:125,1],np.asarray(dataset4['real_cur_pos'])[60:125,2], lw=3, label='deriv(horizon5)',color='b')
    
    ax6 = fig.add_subplot(2,3,4, projection='3d')
    ax6.set_title('Whole trajectory (with obstacle cost)', fontsize=18)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('z')
    ax6.view_init(60, 330)
    ax6.plot_surface(x1, y1, z1, color='r')
    ax6.plot_surface(x2, y2, z2, color='r')
    ax6.plot_surface(x3, y3, z3, color='b')
    ax6.plot_surface(x4, y4, z4, color='b')
    #ax6.scatter(-0.3799078443019837, 0.532240294234275, 0.4244268861275426, s = 100)
    #ax6.scatter(-0.3636915453041721, 0.548578721402501, 0.4377162452901069, s = 100)
    #ax6.scatter(-0.3806772874854791, 0.527353975259506, 0.4187225834059944, s = 100)
    #ax6.scatter(-0.3718779195870111, 0.540325984871439, 0.4301771889978318, s = 100)
    ax6.scatter(-0.5106587048041216, 0.419292690922957, 0.8189151099771221, s = 300, color = 'k', label='goal')
    ax6.plot(np.asarray(dataset1['real_cur_pos'])[:,0],np.asarray(dataset1['real_cur_pos'])[:,1],np.asarray(dataset1['real_cur_pos'])[:,2], lw=3, label='naive(horizon1)',color='r', ls='--')
    ax6.plot(np.asarray(dataset2['real_cur_pos'])[:,0],np.asarray(dataset2['real_cur_pos'])[:,1],np.asarray(dataset2['real_cur_pos'])[:,2], lw=3, label='deriv(horizon1)',color='r')
    ax6.plot(np.asarray(dataset3['real_cur_pos'])[:,0],np.asarray(dataset3['real_cur_pos'])[:,1],np.asarray(dataset3['real_cur_pos'])[:,2], lw=3, label='naive(horizon5)',color='b',ls='--')
    ax6.plot(np.asarray(dataset4['real_cur_pos'])[:,0],np.asarray(dataset4['real_cur_pos'])[:,1],np.asarray(dataset4['real_cur_pos'])[:,2], lw=3, label='deriv(horizon5)',color='b')

    
    
    #plt.legend(fontsize=15)
    #plt.tight_layout(h_pad=0.5, rect=(0,0,1.0,0.96))
    #plt.show()
    #fig.savefig('./fig/obstacle_avoidance.png')
    #fig.savefig('./fig/obstacle_avoidance.pdf',dpi=600)
    
if __name__ == '__main__':
    main()