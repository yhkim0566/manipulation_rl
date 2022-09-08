#!/usr/bin/python
# -*- coding: utf8 -*- 

import tensorflow as tf
import numpy as np
import time
from init_model import NeuralNet
from itertools import product
from collections import defaultdict


import rospy
from std_msgs.msg import Float64MultiArray

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
        state = []
        action = []
        next_action = []
        next_state = []
        total_rewards = []
        for i in range(num_episode):
            total_reward = 0.0
            s = self.env.reset()
            g = self.env.get_body_com("target")[0:2]
            a = self.get_optimal_action(s, g, model, state_index)
            
            acc = self.env.get_acc()
            for j in range(episode_length): 
                state.append(np.concatenate([s[state_index],acc]))
                action.append(a)
                ns,reward,done,_ = self.env.step(a)
                if render:
                    self.env.render()
                total_reward += reward
                a = self.get_optimal_action(s, g, model, state_index)
                s = ns
                acc = self.env.get_acc()
                next_state.append(np.concatenate([s[state_index],acc]))
                next_action.append(a)
                
                if done:
                    self.env.close()
                    break
            total_rewards.append(total_reward)        
        return np.asarray(state),np.asarray(action),np.asarray(next_state), np.asarray(next_action), np.asarray(total_rewards)
    
    
    def cost_fn(self, pred_next_states, goal):
        
        scores = (pred_next_states-goal)**2
        scores += self.NN_Manipulability(pred_next_states)
        
        return scores
    
    
        
        
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
    
    NN1 = NeuralNet(layers, activation = None, deriv=True)
    NN1.build_graph()
    train_total_loss, train_state_loss, train_deriv_loss, eval_total_loss, eval_state_loss, eval_deriv_loss = NN1.train(epoch, train_data, eval_data, exp_name, save=True,eval_interval=10)
    
    
    #NN1.saver.restore(NN1.sess,'./'+exp_name)

            
if __name__ == '__main__':
    main()
