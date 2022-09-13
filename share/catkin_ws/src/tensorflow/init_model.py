#!/usr/bin/python
# -*- coding: utf8 -*- 

import tensorflow as tf
import numpy as np


class NeuralNet():
    def __init__(self, layers, activation, deriv = False, deriv_loss_weight = 0.001):
        self.layers = layers
        self.activation = activation
        self.deriv = deriv
        self.deriv_loss_weight = deriv_loss_weight
        
    def auto_diff(self,cur_state_tf, desired_next_pose, deriv_cur_state_tf, desired_next_vel, pred_next_state):
        n_th1_t = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,0],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,0],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th2_t = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,1],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,1],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th3_t = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,2],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,2],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th4_t = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,3],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,3],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th5_t = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,4],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,4],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th6_t = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,5],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,5],desired_next_pose)[0]*desired_next_vel,1),1)

        n_th1_tt = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,6],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,6],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th2_tt = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,7],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,7],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th3_tt = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,8],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,8],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th4_tt = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,9],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,9],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th5_tt = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,10],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,10],desired_next_pose)[0]*desired_next_vel,1),1)
        n_th6_tt = tf.expand_dims(tf.reduce_sum(tf.gradients(pred_next_state[:,11],cur_state_tf)[0]*deriv_cur_state_tf,1)+tf.reduce_sum(tf.gradients(pred_next_state[:,11],desired_next_pose)[0]*desired_next_vel,1),1)

        return tf.concat([n_th1_t, n_th2_t, n_th3_t, n_th4_t, n_th5_t, n_th6_t, n_th1_tt, n_th2_tt, n_th3_tt, n_th4_tt, n_th5_tt, n_th6_tt],1)

    def initialize_NN(self):        
        weights = []
        biases = []
        num_layers = len(self.layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l+1]])
            b = tf.Variable(tf.zeros([1,self.layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_init(self,size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def model(self, X, weights, biases,alpha):
        num_layers = len(weights) + 1
        H = X
        W = weights[0]
        b = biases[0]
        H = alpha*tf.add(tf.matmul(H, W), b)*tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
        for l in range(1,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = alpha*tf.add(tf.matmul(H, W), b)*tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y
    
    def build_graph(self):
        
        ## Naive case
        ## input : cur_state_tf = [cur_pose, cur_vel] , desired_next_state_tf = [desired_next_pose, desired_next_vel]
        ## output : pred_next_state_tf = [pred_next_pose, pred_next_vel]
        ## label : next_state_tf = [next_pose, next_vel]
        
        ## Derivative case
        ## input : cur_state_tf = [cur_pose, cur_vel] , desired_next_state_tf = [desired_next_pose, desired_next_vel]
        ## output : pred_next_state_tf = [pred_next_pose, pred_next_vel]
        ## additional output using auto_diff : pred_deriv_next_state_tf = [pred_next_vel, pred_next_acc]
        ## label : next_state_tf = [next_pose, next_vel]
        ## additional label **** important **** : deriv_next_state_tf = [next_vel, next_acc -> desired_next_acc]
        
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        
        self.pose = tf.placeholder(tf.float32, [None, 6], name="pose")
        self.vel = tf.placeholder(tf.float32, [None, 6], name="vel")

        self.next_pose = tf.placeholder(tf.float32, [None, 6], name="next_pose")
        self.next_vel = tf.placeholder(tf.float32, [None, 6], name="next_vel")

        self.desired_next_pose = tf.placeholder(tf.float32, [None, 6], name="desired_next_pose")
        
      
        self.cur_state_tf = tf.concat([self.pose,self.vel],1)
        self.next_state_tf = tf.concat([self.next_pose, self.next_vel],1)
        
        
        weights, biases = self.initialize_NN()
        self.pred_next_state = self.model(tf.concat([self.cur_state_tf, self.desired_next_pose],1), weights, biases, 1.0)
        
        self.state_loss  = tf.losses.mean_squared_error(self.next_state_tf, self.pred_next_state)
        
        if self.deriv:
            self.desired_next_vel = tf.placeholder(tf.float32, [None, 6], name="desired_next_vel")
            self.desired_acc = tf.placeholder(tf.float32, [None, 6], name="desired_acc")
            self.desired_next_acc = tf.placeholder(tf.float32, [None, 6], name="desired_next_acc")
            
            self.deriv_current_state_tf = tf.concat([self.vel, self.desired_acc],1)
            self.deriv_next_state_tf = tf.concat([self.next_vel,self.desired_next_acc],1)
            self.pred_deriv_next_state_tf =self.auto_diff(self.cur_state_tf, self.desired_next_pose, self.deriv_current_state_tf, self.desired_next_vel, self.pred_next_state)

            self.deriv_loss = tf.losses.mean_squared_error(self.pred_deriv_next_state_tf, self.deriv_next_state_tf)
         
            
        if self.deriv:
            self.loss = self.state_loss + self.deriv_loss * self.deriv_loss_weight
        else:
            self.loss = self.state_loss
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())    
        self.saver = tf.train.Saver()
        
        
    def train(self, epoch, train_data, eval_data, exp_name, save=True, eval_interval=100):
        # data를 이용하여 feed
        _train_total_loss = []
        _train_state_loss = []
        _train_deriv_loss = []
        
        _eval_total_loss = []
        _eval_state_loss = []
        _eval_deriv_loss = []
        
        for i in range(epoch):
            if self.deriv:
                feed_dict = {self.pose:train_data['pose'],
                             self.vel:train_data['vel'],
                             self.desired_acc:train_data['acc'],
                             self.next_pose:train_data['next_pose'],
                             self.next_vel:train_data['next_vel'],
                             self.desired_next_acc:train_data['next_acc'],
                             self.desired_next_pose:train_data['desired_next_pose'],
                             self.desired_next_vel:train_data['desired_next_vel']}
                _,train_total_loss, train_state_loss, train_deriv_loss = self.sess.run([self.optimizer, self.loss, self.state_loss, self.deriv_loss], feed_dict)
                if i%eval_interval==0:
                    eval_total_loss, eval_state_loss, eval_deriv_loss = self.evaluation(eval_data)
                    _train_total_loss.append(train_total_loss)
                    _train_state_loss.append(train_state_loss)
                    _train_deriv_loss.append(train_deriv_loss)
                    
                    _eval_total_loss.append(eval_total_loss)
                    _eval_state_loss.append(eval_state_loss)
                    _eval_deriv_loss.append(eval_deriv_loss)
                    
                    print(i,train_state_loss, train_deriv_loss, eval_state_loss, eval_deriv_loss)
            else:
                feed_dict = {self.pose:train_data['pose'],
                             self.vel:train_data['vel'],
                             self.next_pose:train_data['next_pose'],
                             self.next_vel:train_data['next_vel'],
                             self.desired_next_pose:train_data['desired_next_pose']}
                _,train_state_loss = self.sess.run([self.optimizer, self.loss], feed_dict)
                if i%eval_interval==0:
                    eval_state_loss = self.evaluation(eval_data)
                    _train_state_loss.append(train_state_loss)
                    _eval_state_loss.append(eval_state_loss)
                    print(i,train_state_loss, eval_state_loss)
        if save:
            self.saver.save(self.sess, './saved_model/'+exp_name)
                    
        return _train_total_loss, _train_state_loss, _train_deriv_loss, _eval_total_loss, _eval_state_loss, _eval_deriv_loss
    
    def evaluation(self, eval_data):
        # data를 이용하여 feed

        if self.deriv:
            feed_dict ={self.pose:eval_data['pose'],
                        self.vel:eval_data['vel'],
                        self.desired_acc:eval_data['acc'],
                        self.next_pose:eval_data['next_pose'],
                        self.next_vel:eval_data['next_vel'],
                        self.desired_next_acc:eval_data['next_acc'],
                        self.desired_next_pose:eval_data['desired_next_pose'],
                        self.desired_next_vel:eval_data['desired_next_vel']}
            total_loss, state_loss, deriv_loss = self.sess.run([self.loss, self.state_loss, self.deriv_loss], feed_dict)
            return total_loss, state_loss, deriv_loss
        else:
            feed_dict ={self.pose:eval_data['pose'],
                        self.vel:eval_data['vel'],
                        self.next_pose:eval_data['next_pose'],
                        self.next_vel:eval_data['next_vel'],
                        self.desired_next_pose:eval_data['desired_next_pose']}
            state_loss = self.sess.run(self.state_loss, feed_dict)
            return state_loss
        
        
    def predict(self, pose, vel, desired_next_pose):
        # data를 이용하여 feed
        feed_dict ={self.pose:pose,
                    self.vel:vel,
                    self.desired_next_pose:desired_next_pose}
        
        return self.sess.run(self.pred_next_state, feed_dict)
    
    
class NeuralNet_Manipulability():
    def __init__(self, layers):
        self.layers = layers
        
    def initialize_NN(self):        
        weights = []
        biases = []
        num_layers = len(self.layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l+1]])
            b = tf.Variable(tf.zeros([1,self.layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def xavier_init(self,size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def model(self, X, weights, biases,alpha):
        num_layers = len(weights) + 1
        H = X
        W = weights[0]
        b = biases[0]
        H = alpha*tf.add(tf.matmul(H, W), b)*tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
        for l in range(1,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = alpha*tf.add(tf.matmul(H, W), b)*tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y
    
    def build_graph(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        self.pose = tf.placeholder(tf.float32, [None, 6], name="pose")
        self.m_index = tf.placeholder(tf.float32, [None, 1], name="m_index")
        
        weights, biases = self.initialize_NN()
        self.pred_m_index = self.model(self.pose, weights, biases, 1.0)
        self.loss  = tf.losses.mean_squared_error(self.pred_m_index, self.m_index)
        
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())    
        self.saver = tf.train.Saver()
        
        
    def train(self, epoch, train_data, eval_data, exp_name, save=True, eval_interval=100):
        _train_loss = []
        _eval_loss = []
        
        for i in range(epoch):
            feed_dict = {self.pose:train_data['pose'],
                        self.m_index:train_data['m_index']}
            _,train_loss = self.sess.run([self.optimizer, self.loss], feed_dict)
            if i%eval_interval==0:
                eval_loss = self.evaluation(eval_data)
                _train_loss.append(train_loss)
                _eval_loss.append(eval_loss)
                print(i,train_loss, eval_loss)
        
        if save:
            self.saver.save(self.sess, './saved_model/m_index_'+exp_name)
            
        return _train_loss, _eval_loss
            
    def evaluation(self, eval_data):
        feed_dict = {self.pose:eval_data['pose'],
                    self.m_index:eval_data['m_index']}
        loss = self.sess.run(self.loss, feed_dict)
        
        return loss
    
    def predict(self, pose):
        feed_dict = {self.pose:pose}
        
        return self.sess.run(self.pred_m_index, feed_dict)