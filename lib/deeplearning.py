# -*- coding: utf-8 -*-
#
# @author Nikhil Bhagawt
# @date 4 April 2019

import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
import time
from datetime import datetime

# Main slim library
slim = tf.contrib.slim

#TF model

class pipeline_AE(object):
  
    def __init__(self, net_arch):
              
        self.input = tf.placeholder(tf.float32, [None, net_arch['input']],name='input_pipeline')
        self.output = tf.placeholder(tf.float32, [None,net_arch['output']],name='output_pipeline')                
        self.is_training = True  #toggles dropout in slim
        self.dropout = 1      

        self.preds = self.get_predictions(net_arch)
        self.loss = self.get_loss(net_arch['loss_type'])
        

    # Individual branch    
    def mlpnet_slim(self, X, net_arch):
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(net_arch['reg'])):

            # If needed, within the scope layer is made linear by setting activation_fn=None.
            # Creates a fully connected layers 
            for l in range(net_arch['n_layers']):
                hidden_output = slim.fully_connected(X, net_arch['l{}'.format(l+1)],normalizer_fn=slim.batch_norm,
                                           scope='fc{}'.format(l))
                hidden_output = slim.dropout(hidden_output, self.dropout, is_training=self.is_training)
            

            return hidden_output
   

    def get_predictions(self, net_arch):
        hidden_output = self.mlpnet_slim(self.input, net_arch)
        predictions = slim.fully_connected(hidden_output, net_arch['output'], activation_fn=None, 
                                               normalizer_fn=slim.batch_norm, scope='prediction_layer')
        return predictions

   
    # Set methods for class variables
    def set_dropout(self, dropout):
        self.dropout = dropout
      
    def set_train_mode(self,is_training):  
        self.is_training = is_training
      
    # Get methods for loss and acc metrics
    def get_loss(self,loss_type):
        if loss_type == 'mse':
            loss = tf.losses.mean_squared_error(self.output, self.preds)
        elif loss_type == 'cosine':
            loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.output, 0), tf.nn.l2_normalize(self.preds, 0), axis=0)        
        else:
            print('Unknown loss type: {}'.format(loss_type))
            loss = None
            
        return loss
        

    
# Other helper functions
def next_batch(s,e,X,y):
    X_batch = X[s:e,:]
    y_batch = y[s:e,:]    
    return X_batch,y_batch

# Train and test defs
def train_network(sess, model, data, optimizer, n_epochs, batch_size, dropout, validate_after, verbose):
    valid_frac = int(0.1*len(data['y']))
    
    # Split into train and valid data for hyperparam tuning
    X_train = data['X'][:1-valid_frac]
    y_train = data['y'][:1-valid_frac]

    X_valid = data['X'][1-valid_frac:]
    y_valid = data['y'][1-valid_frac:]

    total_batch = int(len(y_train)/batch_size)
    
    train_loss_list = []
    valid_loss_list = []
    
    # Training cycle
    for epoch in range(n_epochs):
        avg_loss = 0.
        start_time = time.time()
        # Loop over all batches
        for i in range(total_batch):
            s  = i * batch_size
            e = (i+1) *batch_size

            # Fit training using batch data
            X_batch,y_batch = next_batch(s,e,X_train,y_train)

            # Train pass
            model.set_dropout(dropout)
            _,preds,loss_value = sess.run([optimizer, model.preds,model.loss], 
                                        feed_dict={model.input:X_batch, model.output:y_batch})                

            avg_loss += loss_value

        duration = time.time() - start_time
        if verbose:
            print('epoch {}  time: {:4.2f} loss {:0.4f}'.format(epoch,duration,avg_loss/total_batch))      

        #Compute perf on entire training and validation sets (no need after every epoch)
        if epoch%validate_after == 0:
            train_loss = model.loss.eval(feed_dict={model.input:X_train, model.output:y_train})
            valid_loss = model.loss.eval(feed_dict={model.input:X_valid, model.output:y_valid})
            print('performance on entire train and valid subsets')
            print('epoch {}\t train_loss:{:4.2f}\t valid_loss:{:4.2f}\n'.format(epoch,train_loss,valid_loss))
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
  
    # Post training: Compute preds and metrics for entire train data
    X_train = data['X']
    y_train = data['y']
    
    train_preds= model.preds.eval(feed_dict={model.input:X_train})
    train_metrics = {'train_preds':train_preds,'train_loss':train_loss_list,'valid_loss':valid_loss_list}

    return model, train_metrics

def test_network(sess,model,data):
    print('Testing model')    
    model.set_dropout(1)
    model.set_train_mode(False) 
    X_test = data['X']
    y_test = data['y']
    #print(model.dropout)

    test_preds = model.preds.eval(feed_dict={model.input:X_test})
    test_loss = model.loss.eval(feed_dict={model.input:X_test, model.output:y_test})

    test_metrics = {'test_preds':test_preds,'test_loss':test_loss}
    print('Loss test set {:4.2f}'.format(test_loss))
    return model, test_metrics