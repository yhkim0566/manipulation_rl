#!/root/anaconda3/bin/python
# -*- coding: utf8 -*- 

import tensorflow as tf
import numpy as np
import time
from init_model import NeuralNet, NeuralNet_Manipulability
from init_model_onlytranslation import NeuralNet_3dof
from itertools import product
from collections import defaultdict
from generate_offline_trajectory import GenerateOfflineTrajectory

import rospy
from std_msgs.msg import Float64MultiArray, Float64

# mode
INIT = 0
TELEOP = 1
TASK_CONTROL = 2
JOINT_CONTROL = 3
RL = 4
MOVEIT = 5
IDLE = 6

class MPC_Agent():
    
    def __init__(self, model, m_model, gen_traj, thread_rate):
        
        self.gen_traj = gen_traj
        self.unit_coeff = 0.01
        self.model = model
        self.m_model = m_model
        self.thread_rate = thread_rate
        self.rate = rospy.Rate(self.thread_rate) 
        
        self.real_ik_result_pub = rospy.Publisher('/real/ik_result', Float64MultiArray, queue_size=10)
        
        self.real_pose_sub = rospy.Subscriber('/real/current_pose_rpy', Float64MultiArray, self.real_pose_callback)
        self.real_velocity_sub = rospy.Subscriber('/real/task_velocity', Float64MultiArray, self.real_velocity_callback)
        self.real_m_index_sub = rospy.Subscriber('/real/m_index', Float64, self.real_m_index_callback)


    def get_optimal_action(self,state, vel_coeff):
        
        #current state를 num action만큼 복사 states.shape = (num_actions, 6)
        states = np.tile(state,(self.action_list.shape[0],1)) 
        orientation = states[:,3:6]
        
        norm_action_list = self.action_list[:,0:3] / (np.reshape(np.sqrt(self.action_list[:,0]**2+self.action_list[:,1]**2+self.action_list[:,2]**2),(-1,1))+10E-6)
        orientation_action_list = self.action_list[:,3:6]*0.00
                   
        first_actions = np.concatenate([states[:,0:3] + norm_action_list * vel_coeff * self.unit_coeff,orientation + orientation_action_list],1)
        action = first_actions
        
        total_costs = np.zeros(self.action_list.shape[0])
        for i in range(self.horizon):
            next_states = self.model.predict(states[:,0:6],states[:,6:12],action)
            orientation = next_states[:,3:6]
            #next_states = np.concatenate([next_states[:,0:3],self.orientation,next_states[:,3:6],next_states[:,3:6]],1)
            total_costs += self.cost_fn(next_states)*np.power(0.9,i)
            action = np.concatenate([next_states[:,0:3] + norm_action_list * vel_coeff * self.unit_coeff,orientation + orientation_action_list],1)
        
            states = next_states
             
        return first_actions[np.argmin(total_costs),:].flatten()
    
    def run_policy(self, num_episode, episode_length, time_horizon, init_pose, goal_pose, manip_coeff, istest):
        self.horizon = time_horizon
        self.manip_coeff = manip_coeff
        datasets = []
        for i in range(num_episode):
            total_dist_reward = 0.0
            total_m_index_reward = 0.0
            vel_coeff = 1.0

            print('reset the episode and generate random goal')
            state = self.reset(init_pos= init_pose, goal_pos=goal_pose, istest=istest)
            self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle4 = self.generate_obstacles(init_pose, goal_pose)
            print('obstacles')
            print(self.obstacle1, self.obstacle2,self.obstacle3, self.obstacle4)
            print('start, goal')
            print(state[:,0:6], self.goal)
            rospy.set_param('/real/mode', JOINT_CONTROL)
            self.action_list = np.concatenate([np.asarray(list(product([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1],repeat=3))),np.zeros((9*9*9,3))],1)
            desired_next_pose = self.get_optimal_action(state, vel_coeff)
            dataset = defaultdict(list) 
            for j in range(episode_length):
                
                dataset['real_cur_pos'].append(state[0:6].flatten())
                dataset['real_cur_vel'].append(state[6:12].flatten())
                dataset['real_m_index'].append(self.real_m_index)
                
                #vel_coeff = (episode_length-j)/episode_length
                next_state, dist_reward, orientation_reward, m_index_reward = self.step(desired_next_pose)
                total_dist_reward += dist_reward
                total_m_index_reward += m_index_reward
                desired_next_pose = self.get_optimal_action(state, vel_coeff)
                self.prev_pose = state
                state = next_state
                if dist_reward < self.unit_coeff/10:
                    self.action_list = np.concatenate([np.zeros((5*5*5,3)), np.asarray(list(product([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1],repeat=3)))],1)
                    if orientation_reward < 0.1: ## minimum moving resolution < sqrt(x_resolution^2 + y_resolution^2 + z_resolution^2)
                        print('arrived at the goal')
                        break
                    
                if orientation_reward < 0.1:
                    self.action_list = np.concatenate([np.asarray(list(product([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1],repeat=3))),np.zeros((9*9*9,3))],1)
                    if  dist_reward < self.unit_coeff/10:## minimum moving resolution < sqrt(x_resolution^2 + y_resolution^2 + z_resolution^2)
                        print('arrived at the goal')
                        break
                #print(dist_reward,m_index_reward)
                #dataset['real_next_pos'].extend(np.expand_dims(state[0:6],1).transpose())
                #dataset['real_next_vel'].extend(np.expand_dims(state[6:12],1).transpose())
                
                dataset['dist_reward'].append(dist_reward)
                dataset['orientation_reward'].append(orientation_reward)
                dataset['pred_m_index'].append(m_index_reward.flatten())
            
            datasets.append(dataset)    
            print(j*self.unit_coeff, orientation_reward, total_m_index_reward/j) ## 1step당 이동거리 * step 수 , trajectory의 평균 m_index
            #self.update_models()
            
        return datasets
    
    #def update_models(self):
    #    self.model.train(100, train_data, eval_data, exp_name, save=True, eval_interval=10)
    #    self.m_model.train(100, train_data, eval_data, exp_name, save=True, eval_interval=10)
        
    
    def cost_fn(self, pred_next_states):
        distance_cost = np.sqrt(np.sum((pred_next_states[:,0:3]-self.goal[0:3])**2,1))
        orientation_cost = 0.0#np.sqrt(np.sum((pred_next_states[:,3:6]-self.goal[3:6])**2,1))*0.5
        if np.max(np.sqrt(np.sum((pred_next_states[:,0:3]-self.obstacle1)**2,1))) < 0.1:
            obstacle1_cost = -np.sqrt(np.sum((pred_next_states[:,0:3]-self.obstacle1)**2,1))*0.05
        else:
            obstacle1_cost = 0.0
        if np.max(np.sqrt(np.sum((pred_next_states[:,0:3]-self.obstacle2)**2,1))) < 0.1:
            obstacle2_cost = -np.sqrt(np.sum((pred_next_states[:,0:3]-self.obstacle2)**2,1))*0.05
        else:
            obstacle2_cost = 0.0
            
        if np.max(np.sqrt(np.sum((pred_next_states[:,0:3]-self.obstacle3)**2,1))) < 0.1:
            obstacle3_cost = -np.sqrt(np.sum((pred_next_states[:,0:3]-self.obstacle3)**2,1))*0.05
        else:
            obstacle3_cost = 0.0
        if np.max(np.sqrt(np.sum((pred_next_states[:,0:3]-self.obstacle4)**2,1))) < 0.1:
            obstacle4_cost = -np.sqrt(np.sum((pred_next_states[:,0:3]-self.obstacle4)**2,1))*0.05
        else:
            obstacle4_cost = 0.0
            
        #manipulability_cost = -self.m_model.predict(pred_next_states[:,0:6]).flatten()*np.mean(distance_cost)*self.manip_coeff
        scores = distance_cost + orientation_cost + obstacle1_cost + obstacle2_cost + obstacle3_cost + obstacle4_cost#+ manipulability_cost
        return scores
    
    def reward_fn(self, state):
        
        distance_cost = np.sqrt(np.sum((state[:,0:3]-self.goal[0:3])**2)) 
        orientation_cost = np.sqrt(np.sum((state[:,3:6]-self.goal[3:6])**2,1))
        obstacle_cost = -np.sqrt(np.sum((state[:,0:3]-self.obstacle1)**2,1)) -np.sqrt(np.sum((state[:,0:3]-self.obstacle2)**2,1))
        
        #manipulability_cost = self.m_model.predict(state[:,0:6]).flatten()
        print(distance_cost, orientation_cost, obstacle_cost)#, manipulability_cost)
        return distance_cost, orientation_cost, obstacle_cost#, manipulability_cost
    
    def reset(self, init_pos, goal_pos, istest):

        if not istest:
            target_traj, _, _, _ = self.gen_traj.generate_online_trajectory_and_go_to_init(index = 1)
            goal = target_traj[:,-1]
            self.goal = self.arrange_orientation_data(goal)
        else:
            self.gen_traj.generate_given_trajectory_and_go_to_init(index = 1, init_pos=init_pos)
            self.goal = goal_pos

        state = self.get_robot_state()

        return state
        
    def step(self, desired_next_pose):

        # ik solve for publishing target joint angle
        target_pose = self.gen_traj.input_conversion(desired_next_pose)
        target_angle = self.gen_traj.solve_ik_by_moveit(target_pose)
        for i in range(3):
            self.real_ik_result_pub.publish(target_angle)
        
        #wait robot moving
            self.rate.sleep() 
        
        # get state and evaluate reward
        state = self.get_robot_state()
        dist_reward,orientation_cost,m_index_reward = self.reward_fn(state)
        
        return state, dist_reward, orientation_cost, m_index_reward
         
    def get_robot_state(self):
        pose = self.real_pose
        pose = self.arrange_orientation_data(pose)
        vel = self.real_velocity
        return np.transpose(np.expand_dims(np.concatenate([pose,vel],0),1))
        
    def real_pose_callback(self, data):
        self.real_pose = data.data      
              
    def real_velocity_callback(self, data):
        self.real_velocity = data.data

    def real_m_index_callback(self, data):
        self.real_m_index = data.data   
        

    def arrange_orientation_data(self, pose):
       
        pose = np.asarray(pose)
        orientation_range = 0.5
        if pose[3] >  -0.019840669143976232 + orientation_range:
            pose[3] = pose[3] - np.pi
            
        if pose[4] >  0.8654425109456884 + orientation_range: # 0.3은 orientation range 보다 조금 더 큰 값 
            pose[4] = pose[4] - np.pi
        
        if pose[5] > 1.63721386022732566 +orientation_range:
            pose[5] = pose[5] - np.pi
            
        if pose[3] <  -0.019840669143976232 - orientation_range:
            pose[3] = pose[3] + np.pi  
                
        if pose[4] <  0.8654425109456884 - orientation_range: # 0.3은 orientation range 보다 조금 더 큰 값 
            pose[4] = pose[4] + np.pi
        
        if pose[5] < 1.63721386022732566 - orientation_range:
            pose[5] = pose[5] + np.pi

        
        return (pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]) 
    
    def generate_obstacles(self, init_pose, goal_pose):
        line = np.linspace(init_pose,goal_pose,100)
        
        #obstacle1 = line[33][0:3] 
        #obstacle2 = line[66][0:3] #np.array([-0.3969209, 0.3706089723, 0.742756550])#
        obstacle1 = np.array([-0.3636915453041721, 0.548578721402501, 0.4377162452901069])
        obstacle2 = np.array([-0.3718779195870111, 0.540325984871439, 0.4301771889978318])
        obstacle3 = np.array([-0.37990784430198377, 0.5322402942342755, 0.42442688612754265])
        obstacle4 = np.array([-0.3806772874854791, 0.5273539752595069, 0.41872258340599444])
                             

        # random obstacle
        #obstacle1 = line[33] + np.random.rand(6)*0.1 - 0.05
        #obstacle2 = line[66] + np.random.rand(6)*0.1 - 0.05
        return obstacle1, obstacle2, obstacle3, obstacle4
            
 
def split_and_arrange_dataset(datasets, ratio=0.8):
    # get dataset episode size, random sampling 8:2
    dataset_size = datasets.shape[0]
    dataset_index = np.linspace(0,dataset_size-1,dataset_size,dtype=int)
    np.random.shuffle(dataset_index)
    train_index = dataset_index[:int(dataset_size*ratio)]
    eval_index = dataset_index[int(dataset_size*ratio):]

    train_data = datasets[train_index]
    eval_data = datasets[eval_index]

    _train_data = defaultdict(list)
    for i in range(len(train_data)):
        for k in train_data[0].keys():
            _train_data[k.decode('utf-8')].extend(np.asarray(train_data[i][k]))

    _eval_data = defaultdict(list)
    for i in range(len(eval_data)):
        for k in eval_data[0].keys():
            _eval_data[k.decode('utf-8')].extend(np.asarray(eval_data[i][k]))
            
    return _train_data, _eval_data
    

       
def main():
    
    load_model = True
    load_dataset = True
    save = True
    mpc = True
    deriv = True
    rospy.init_node("mpc_loop", anonymous=True)

    # define transition model neural network
    #if deriv:
    #    model_name = '3dof_deriv'
    #else:
    #    model_name = '3dof_naive'
    #layers = [9,100,100,100,6]
    
    if deriv:
        model_name = 'deriv_ntraj50_params_ori02_xyz_08_05_in_055_03_trial2_ratio090_2222'
    else:
        model_name = 'naive_ntraj50_params_ori02_xyz_08_05_in_055_03_trial2_ratio090_2222'   

    
    layers = [18,100,100,100,12]
    
    NN = NeuralNet(layers, activation = None, deriv=deriv)
    NN.build_graph()
    #NN2 = NeuralNet(layers, activation = None, deriv=False)
    #NN2.build_graph()

    
    # define manipulability model neural network
    layers = [6,100,100,1]
    NN_Manip = NeuralNet_Manipulability(layers)
    NN_Manip.build_graph()
    
    
    # generate trajectory class and go to init pose
    gen_traj = GenerateOfflineTrajectory(thread_rate = 40, real = True, unity = False)    
    
    # init ros node and go to init pose
    
    rospy.set_param('/real/mode', INIT)
    time.sleep(5)
    
    # neural network model load or train
    if not load_model:
        
         # collect initial dataset
        if load_dataset:
            datasets = np.load('./dataset/ntraj50_params_ori02_xyz_08_05_in_055_03_trial2.npy_2.npy', encoding='bytes')
        else:
            datasets = gen_traj.start_data_collection(episode_num = 10, index = 1)
        
        train_data, eval_data = split_and_arrange_dataset(datasets,ratio=0.90)
        
        # train models using offline dataset
        epoch = 5000
        eval_interval = 100
        train_total_loss, train_state_loss, train_deriv_loss, eval_total_loss, eval_state_loss, eval_deriv_loss = NN.train(epoch, train_data, eval_data, model_name, save, eval_interval)
        #train_total_loss, train_state_loss, train_deriv_loss, eval_total_loss, eval_state_loss, eval_deriv_loss = NN2.train(epoch, train_data, eval_data, model_name2, save, eval_interval)
        #epoch = 3000
        #eval_interval = 100
        #m_train_loss, m_eval_loss = NN_Manip.train(epoch, train_data, eval_data, save, eval_interval)
        
    else:
        datasets = np.load('./dataset/ntraj50_params_ori02_xyz_08_05_in_055_03_trial2.npy_2.npy', encoding='bytes')
        train_data, eval_data = split_and_arrange_dataset(datasets) # 저장된 train eval data 불러와야함. (cheating 가능성)
        
        NN.saver.restore(NN.sess,'./saved_model/'+model_name)
        NN_Manip.saver.restore(NN_Manip.sess,'./saved_model/m_index')    


    # mpc loop
    istest=True
    if mpc:

        agent = MPC_Agent(model = NN, m_model = NN_Manip, gen_traj = gen_traj, thread_rate = 40)
        '''
        #try1
        init_pose = [0.30613005, 0.80924435, 0.54419401, 0.04087591, 1.4184055, 1.51275012]
        goal_pose = [-0.37863849083914725, 0.5847729319475161, 0.6181325218578473, 0.18783086971417767, 1.5114953916525464, 1.2863334733003216]
        
        print('1')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip00_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('2')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip05_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('3')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip10_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('4')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip00_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('5')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip05_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('6')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip10_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('7')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip00_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('8')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip05_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('9')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip10_horizon5.npy'
        np.save('./result/'+filename,datasets)
        '''
        '''
        print('10')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip00_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('11')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip05_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('12')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try1_manip10_horizon7.npy'
        np.save('./result/'+filename,datasets)        
        
        
        #try2
        init_pose = [-0.62214387,  0.45628403,  0.33778029,  0.09020013,  1.01727371,  1.58841589]
        goal_pose = [-0.2423847201614614, 0.36666238847899657, 0.7523733116634912, -0.01929954166940509, 0.7373673127157423, 1.708590343970926]
        print('1')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip00_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('2')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip05_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('3')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip10_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('4')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip00_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('5')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip05_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('6')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip10_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('7')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip00_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('8')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip05_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('9')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip10_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('10')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip00_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('11')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip05_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('12')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try2_manip10_horizon7.npy'
        np.save('./result/'+filename,datasets)        
        
        #try3
        init_pose = [-0.6599908,   0.42967212,  0.44839616,  0.08756153,  0.63682255,  1.63248467]
        goal_pose = [0.7837295521702641, 0.7928310902786284, 0.6695677180524834, -0.18023488991868655, 0.6935203516560906, 1.5474963351914952]
        print('1')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip00_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('2')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip05_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('3')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip10_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('4')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip00_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('5')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip05_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('6')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip10_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('7')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip00_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('8')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip05_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('9')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip10_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('10')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip00_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('11')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip05_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('12')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try3_manip10_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        
        #try4
        init_pose = [ 0.17817765,  0.39522357,  0.3761077,  -0.30800421,  0.85197747,  1.56139825]
        goal_pose = [0.4476672477046888, 0.5850137468948966, 0.738266858185884, -0.09463220035563447, 0.8884850697937299, 1.4480424256801538]
        
        print('1')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip00_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('2')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip05_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('3')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip10_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('4')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip00_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('5')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip05_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('6')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip10_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('7')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip00_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('8')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip05_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('9')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip10_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('10')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip00_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('11')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip05_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('12')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try4_manip10_horizon7.npy'
        np.save('./result/'+filename,datasets)
        '''
        #try5
        init_pose = [-0.31735397,  0.59395132,  0.30047592, -0.31696057,  0.91241833,  1.43027166]
        goal_pose = [-0.5106587048041216, 0.419292690922957, 0.8189151099771221, -0.31766847312907986, 1.0880740411752865, 1.613780680747806]

        print('1')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 1, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'obstacle_avoidance.npy'
        np.save('./result/'+filename,datasets)
        '''
        print('2')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip05_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('3')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 1, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip10_horizon1.npy'
        np.save('./result/'+filename,datasets)
        
        print('4')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip00_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('5')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip05_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('6')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 3, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip10_horizon3.npy'
        np.save('./result/'+filename,datasets)
        
        print('7')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip00_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('8')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip05_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('9')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 5, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip10_horizon5.npy'
        np.save('./result/'+filename,datasets)
        
        print('10')
        manip_coeff = 0.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip00_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('11')
        manip_coeff = 0.5
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip05_horizon7.npy'
        np.save('./result/'+filename,datasets)
        
        print('12')
        manip_coeff = 1.0
        datasets = agent.run_policy(num_episode = 3, episode_length = 500, time_horizon = 7, init_pose=init_pose, goal_pose = goal_pose, manip_coeff = manip_coeff,istest=istest)
        filename = 'naive_orient_try5_manip10_horizon7.npy'
        np.save('./result/'+filename,datasets)
        '''
    
if __name__ == '__main__':
    main()
