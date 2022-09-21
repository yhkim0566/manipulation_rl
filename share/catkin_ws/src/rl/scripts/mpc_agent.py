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
    
    def __init__(self, model, m_model, gen_traj, time_horizon, num_action, thread_rate=40):
        self.horizon = time_horizon
        self.num_action = num_action
        self.gen_traj = gen_traj
        self.unit_coeff = 0.05
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
        states = np.tile(state,(self.num_action,1)) 
        self.orientation = states[:,3:6]
        action_list = np.asarray(list(product([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1],repeat=3)))
        norm_action_list = action_list / (np.reshape(np.sqrt(action_list[:,0]**2+action_list[:,1]**2+action_list[:,2]**2),(-1,1))+10E-6)
                
        first_actions = np.concatenate([states[:,0:3] + norm_action_list * vel_coeff * self.unit_coeff,self.orientation],1)
        action = first_actions
        
        total_costs = np.zeros(self.num_action)
        for i in range(self.horizon):
            next_states = self.model.predict(states[:,0:6],states[:,6:12],action)
            next_state = np.concatenate([next_states[:,0:3],self.orientation,next_states[:,3:6],next_states[:,3:6]],1)
            total_costs += self.cost_fn(next_states)*np.power(0.9,i)
            action = np.concatenate([next_states[:,0:3] + norm_action_list * vel_coeff * self.unit_coeff,self.orientation],1)
            states = next_state
             
        return first_actions[np.argmin(total_costs),:].flatten()
    
    def run_policy(self, num_episode, episode_length, datasets):
        for i in range(num_episode):
            total_dist_reward = 0.0
            total_m_index_reward = 0.0
            vel_coeff = 1.0
            print('reset the episode and generate random goal')
            state = self.reset(init_pos= [], goal_pos=[], istest=True)
            print(state, self.goal)
            rospy.set_param('/real/mode', JOINT_CONTROL)
            desired_next_pose = self.get_optimal_action(state, vel_coeff)
            for j in range(episode_length):
                #dataset = defaultdict(list) 
                #dataset['real_cur_pos'].extend(np.expand_dims(state[0:6],1).transpose())
                #dataset['real_cur_vel'].extend(np.expand_dims(state[6:12],1).transpose())
                #dataset['real_m_index'].extend(self.real_m_index)
                
                #dataset['desired_next_pos'].extend(np.expand_dims(desired_next_state,1).transpose())
                #vel_coeff = (episode_length-j)/episode_length
                next_state, dist_reward, m_index_reward = self.step(desired_next_pose)
                total_dist_reward += dist_reward
                total_m_index_reward += m_index_reward
                desired_next_pose = self.get_optimal_action(state, vel_coeff)
                state = next_state
                if dist_reward < self.unit_coeff: ## minimum moving resolution < sqrt(x_resolution^2 + y_resolution^2 + z_resolution^2)
                    print('arrived at the goal')
                    break
                print(dist_reward,m_index_reward)
                #dataset['real_next_pos'].extend(np.expand_dims(state[0:6],1).transpose())
                #dataset['real_next_vel'].extend(np.expand_dims(state[6:12],1).transpose())
                
                #dataset['reward'].extend(reward)
            #dataset['total_reward'].extend(total_reward)    
            #datasets.append(dataset)    
            print(j*self.unit_coeff, total_m_index_reward/j) ## 1step당 이동거리 * step 수 , trajectory의 평균 m_index
            #self.update_models()
            
        #return datasets
    
    def update_models(self):
        self.model.train(100, train_data, eval_data, exp_name, save=True, eval_interval=10)
        self.m_model.train(100, train_data, eval_data, exp_name, save=True, eval_interval=10)
        
    
    def cost_fn(self, pred_next_states):
        distance_cost = np.sqrt(np.sum((pred_next_states[:,0:3]-self.goal[0:3])**2,1))
        manipulability_cost = -self.m_model.predict(pred_next_states[:,0:6]).flatten()*0.1
        scores = distance_cost + manipulability_cost
        return scores
    
    def reward_fn(self, state):
        
        distance_cost = np.sqrt(np.sum((state[:,0:3]-self.goal[0:3])**2))
        manipulability_cost = self.m_model.predict(state[:,0:6]).flatten()*0.1
        print(distance_cost, manipulability_cost)
        scores = distance_cost + manipulability_cost
        return scores
    
    def reset(self, init_pos, goal_pos, istest):

        if not istest:
            target_traj, _, _, _ = self.gen_traj.generate_online_trajectory_and_go_to_init(index = 1)
            self.goal = target_traj[:,-1]
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
        dist_reward,m_index_reward = self.reward_fn(state)
        
        return state, dist_reward, m_index_reward
         
    def get_robot_state(self):
        pose = self.real_pose
        vel = self.real_velocity
        return np.transpose(np.expand_dims(np.concatenate([pose,vel],0),1))
        
    def real_pose_callback(self, data):
        self.real_pose = data.data      
              
    def real_velocity_callback(self, data):
        self.real_velocity = data.data

    def real_m_index_callback(self, data):
        self.real_m_index = data.data   
        
        
 
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
    if deriv:
        model_name = '3dof_deriv'
    else:
        model_name = '3dof_naive'
    layers = [9,100,100,100,6]
    
    NN = NeuralNet_3dof(layers, activation = None, deriv=deriv)
    NN.build_graph()
    
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
            datasets = np.load('./dataset/datasets_damp_2500.npy', encoding='bytes')
        #else:
            datasets = gen_traj.start_data_collection(episode_num = 10, index = 1)
        
        train_data, eval_data = split_and_arrange_dataset(datasets,ratio=0.95)
        
        # train models using offline dataset
        epoch = 3000
        eval_interval = 100
        train_total_loss, train_state_loss, train_deriv_loss, eval_total_loss, eval_state_loss, eval_deriv_loss = NN.train(epoch, train_data, eval_data, model_name, save, eval_interval)
        
        epoch = 3000
        eval_interval = 100
        #m_train_loss, m_eval_loss = NN_Manip.train(epoch, train_data, eval_data, save, eval_interval)
        
    else:
        datasets = np.load('./dataset/datasets_damp_2500.npy', encoding='bytes')
        train_data, eval_data = split_and_arrange_dataset(datasets) # 저장된 train eval data 불러와야함. (cheating 가능성)
        
        NN.saver.restore(NN.sess,'./saved_model/'+model_name)
        NN_Manip.saver.restore(NN_Manip.sess,'./saved_model/m_index')    
        
    # mpc loop
    if mpc:
        num_action = 9*9*9 # all combinations of [-1,0,1] for 6dof
        agent = MPC_Agent(model = NN, m_model = NN_Manip, gen_traj = gen_traj, time_horizon = 5, num_action = num_action)
        datasets = agent.run_policy(num_episode = 10, episode_length = 500, datasets = train_data)
    
    #filename = 'datasets_damp_2500.npy'
    #np.save('./'+filename,datasets)
if __name__ == '__main__':
    main()
