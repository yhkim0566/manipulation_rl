#!/usr/bin/python
# -*- coding: utf8 -*- 

import tensorflow as tf
import numpy as np
import time
from init_model import NeuralNet
from itertools import product
from collections import defaultdict


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
    
    def __init__(self, time_horizon = 10, num_action = 729):
        self.horizon = time_horizon
        self.num_action = num_action
        
        self.real_ik_result_pub = rospy.Publisher('/real/ik_result', Float64MultiArray, queue_size=10)
        
        self.real_pose_sub = rospy.Subscriber('/real/current_pose_rpy', Float64MultiArray, self.real_pose_callback)
        self.real_velocity_sub = rospy.Subscriber('/real/task_velocity', Float64MultiArray, self.real_velocity_callback)
        self.real_m_index_sub = rospy.Subscriber('/real/m_index', Float64, self.real_m_index_callback)


    def get_optimal_action(self,state, goal, model):
        
        #current state를 num action만큼 복사 states.shape = (num_actions, 6)
        states = np.tile(state,(self.num_action,1)) 
        
        # action = desired next pose (state = pose and vel)
        # action.shape = (num_actions, 6), make all possible actions for 6 DOF
        first_actions = states[:,0:6] + np.asarray(list(product([-1,0,1],repeat=6))) 
        action = first_actions
        
        total_costs = np.zeros(self.num_action)
        for i in range(self.horizon):
            next_states = model.predict(states,action)
            total_costs += self.cost_fn(next_states, goal)*np.power(0.99,i)
            action = next_states[:,0:6] + np.asarray(list(product([-1,0,1],repeat=6)))
            states = next_states
             
        return first_actions[np.argmin(total_costs),:].flatten()
    
    def run_policy(self, num_episode, episode_length, model, dataset):
        for i in range(num_episode):
            total_reward = 0.0
            state = self.reset()
            print('reset the episode')
            
            goal = self.get_goal()
            print('generate random goal')
            
            desired_next_state = self.get_optimal_action(state, goal, model)
            
            for j in range(episode_length): 
                dataset['real_cur_pos'].append(np.expand_dims(state[0:6],1).transpose())
                dataset['real_cur_vel'].append(np.expand_dims(state[6:12],1).transpose())
                
                dataset['desired_next_pos'].append(np.expand_dims(desired_next_state,1).transpose())
                
                next_state,reward = self.step(desired_next_state)
                total_reward += reward
                desired_next_state = self.get_optimal_action(s, g, model)
                state = next_state
                
                dataset['real_next_pos'].append(np.expand_dims(state[0:6],1).transpose())
                dataset['real_next_vel'].append(np.expand_dims(state[6:12],1).transpose())
                
                dataset['reward'].append(reward)
            dataset['total_reward'].append(total_reward)        
    
    
    def cost_fn(self, pred_next_states, goal):
        
        scores = (pred_next_states-goal)**2
        scores += self.NN_Manipulability(pred_next_states)
        
        return scores
    
    def reset(self):

        # To do
        # go to random initial pose command instead of INIT mode
        rospy.set_param('/real/mode', IDLE)
        target_traj, traj_vel, traj_acc, target_traj_length = GenTraj.generate_online_trajectory_and_go_to_init(index = 1)
        state = self.get_robot_state()

        return state
    
        
    def step(self, desired_next_pose):
        
        desired_next_pose = self.solve_ik_by_moveit(desired_next_pose)
        self.real_ik_result_pub.publish(desired_next_pose)
        
        state = self.get_robot_state(state, goal)
        reward = self.cost_fn()
        
        return state, reward    
         
    def get_robot_state(self):
        pose = self.real_pose
        vel = self.real_velocity
        return np.concatenate([pose,vel],1)
        
    def real_pose_callback(self, data):
        self.real_pose = data.data      
              
    def real_velocity_callback(self, data):
        self.real_velocity = data.data

    def real_m_index_callback(self, data):
        self.real_m_index = data.data   

    def get_goal(self):
        goal = ''
        return goal
        
def main():
    
    dataset = defaultdict(list)
    
    pose = np.ones([729,6])
    vel = np.ones([729,6])
    desired_acc = np.ones([729,6])
    
    desired_next_pose = np.ones([729,6])
    desired_next_vel = np.ones([729,6])
    desired_next_acc = np.ones([729,6])
    
    next_pose = np.ones([729,6])
    next_vel = np.ones([729,6])
    next_acc = np.ones([729,6])
    train_data = {}
    
    train_data['pose'] = pose
    train_data['vel'] = vel
    train_data['acc'] = desired_acc
    
    train_data['next_pose'] = next_pose
    train_data['next_vel'] = next_vel
    train_data['next_acc'] = desired_next_acc
    
    train_data['desired_next_pose'] = desired_next_pose
    train_data['desired_next_vel'] = desired_next_vel
    
    eval_data = train_data
    
    epoch = 100
    exp_name = 'deriv_test'
    layers = [24,100,100,100,12]
    
    NN = NeuralNet(layers, activation = None, deriv=True)
    NN.build_graph()
    train_total_loss, train_state_loss, train_deriv_loss, eval_total_loss, eval_state_loss, eval_deriv_loss = NN.train(epoch, train_data, eval_data, exp_name, save=True,eval_interval=10)
    
    NN_Manip = NeuralNet_Manipulability(layers, activation = None, derive = False)
    NN_Manip.build_graph()
    
    
    #NN1.saver.restore(NN1.sess,'./'+exp_name)

            
if __name__ == '__main__':
    main()
