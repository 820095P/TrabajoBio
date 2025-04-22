# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:44:43 2020

@author: Steven Gore
"""

#idea, use knowledge distillation to train the student model to find concepts and square them
#this could help us hold the concepts steady, in the teacher network, but let the student network learn them
#https://arxiv.org/abs/1910.01348
#https://arxiv.org/pdf/2007.01922.pdf

import warnings
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Layer
from keras.objectives import categorical_crossentropy
from keras.optimizers import RMSprop
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping
import numpy as np
import pandas as pd
import os
from datetime import date

class PrintLatentCallback(Callback):
    def __init__(self, latent_layer, filepath, exp_name = None):
        self.filepath = filepath
        self.latent_layer = latent_layer
        self.exp_name = exp_name
    
    def make_dir(self):
        today = date.today()
        d = today.strftime("%b-%d-%Y")
        if self.exp_name != None:
            self.full_path = self.filepath +'//'+ self.exp_name +'//' + d
        else:
            self.full_path = self.filepath +'//'+ 'latentCallback' +'//' + d
        os.makedirs(self.full_path)
    def make_name(self, ep):
        
        if self.exp_name != None:
            self.n = self.full_path +'//' + self.exp_name +'_Epoch_' + str(ep)
        else:
            self.n = self.full_path +'//' +'_Epoch_' + str(ep)
    def on_train_begin(self, logs = None):
        self.make_dir()
        
    def on_epoch_end(self,epoch, logs = None):
        func = K.function([self.model.input], [self.model.layers[1].layers[self.latent_layer].output])
        p = func(self.validation_data)
        self.make_name(epoch)
        np.save(self.n, p)

class TrainingSwitchCallback(EarlyStopping):
#create a callback that uses two arguments for loss_weight and switches between them based on some criterion
#Probably i will use model training params like in early stopper, but instead I will reset the number of epochs and continue training with the new loss weights
#inspired by https://github.com/keras-team/keras/issues/2595
    def __init__(self, alpha, beta, 
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto'):
        self.alpha = alpha
        self.beta = beta
        self.switch = 0
        super(TrainingSwitchCallback, self).__init__(monitor = monitor,
                                                     min_delta = min_delta,
                                                     patience = patience,
                                                     verbose = verbose,
                                                     mode = mode)
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
          if self.wait >= self.patience:
              if self.switch == 1:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.wait += 1
              else:
                  print('switching training weights, resetting epoch count')
                  print('switching training weights, resetting epoch count')
                  self.switch = 1
                  K.set_value(self.alpha,0)
                  K.set_value(self.beta,1)
                  self.wait = 0
                  self.stopped_epoch = 0
                  self.best = np.Inf if self.monitor_op == np.less else -np.Inf
                  self.best_weights = None
                  self.wait += 1
            
            

class Sampling(Layer):
    """Defines probablistic layer of VAE latent layer
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def vae_loss(beta, z_mean, z_log_sigma):
    def loss_calc(y_true, y_pred):
        xent_loss = tf.cast(tf.shape(y_true)[1], tf.float32) * categorical_crossentropy(y_true, y_pred)
        kl_loss = - beta * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=1)
        return xent_loss + kl_loss
    return loss_calc

def open_file(filename, sep = ",", header = 0, index_col = None):
    '''opens a file.
    Capable of opening pickle, csv or txt formats
    for csvs or txt the separator must be indicated
    if a specific index should be used as the column names then index_col may be used and must be the index number
    if index number is unknown but index name is then rename will take the index name and do the same thing as index_col
    '''
    #from pandas.io.parser import CParserError

    filetype = filename.split(".")[-1]
    assert filetype == "pickle" or filetype == "csv" or filetype == "txt", "filetype must be 'pickle', txt' or 'csv'"
    if filetype == "pickle":
        import pickle
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        
    if filetype == "csv":
        #assert type(data) == pd.DataFrame, "to open a 'csv' please ensure the data is a pandas DataFrame"
        data = pd.read_csv(filename, sep = sep, header = header, index_col = index_col)
        
    if filetype == "txt":
        data = pd.read_csv(filename, sep = sep, header = header, index_col = index_col)

    return data      

    
class VAE:
    """
    creates a VAE model of the depth and dimensions defined by user
    Inputs:
        encoder_dims: list of ints with the length of the encoder depth. First entry is size of input
        latent_dim: int, defines the size of the probablistic layer
        decoder_dims: list of ints with the length of the decoder depth
        classifier_dims: list of ints with the length of the classifier depth including the output class layer
        output_weights: a list of ints of length 2 where the first entry is the weight of the generative loss and the second is the weight of the classification loss
    Call build function to build the vae.
    """
    def __init__(self, encoder_dims, latent_dim, decoder_dims, classifier_dims = 0, output_weights = 0, classifier= 'Inline', beta = 1):
        
        '''classifier can take the values None, Inline and Separate
        None makes a VAE without a classifier
        Inline makes the VAE with classifier trained as a single model
        Separate makes a vae and classifier and trains them in separate steps'''
        
        assert type(encoder_dims) == list, "encoder_dims must be a list"
        assert type(latent_dim) == int, "latent_dim must be a an int"
        assert latent_dim > 0, "latent_dim must be non-zero"
        assert type(decoder_dims) == list, "decoder_dims must be a list"
        
        self.classifier = classifier
        self.output_weights = output_weights
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.decoder_dims = decoder_dims
        self.initializer = 'glorot_normal'
        if self.classifier != None:
            self.classifier_dims = classifier_dims
            assert type(classifier_dims) == list, "classifier_dims must be a list"
            assert len(output_weights) == 2, "output_weights must be of length 2"
            assert type(output_weights) == list, "output_weights must be a list"
        self.beta = beta
    def make_encoder(self):
        """
        Generates encoder with depth = len(encoder_dims)-1
        """
        self.inputs = Input(shape = (self.encoder_dims[0],), name = 'Input')
        e = [self.inputs]
        for i in self.encoder_dims[1:]:
            print(i)
            print(e)
            print(e[-1])
            e.append(Dense(i, activation = 'relu', kernel_initializer = self.initializer)(e[-1]))
        self.z_mean = (Dense(self.latent_dim, name = 'z_mean')(e[-1]))
        self.z_log_sigma = (Dense(self.latent_dim, name = 'z_log_sigma')(e[-1]))
        latent = Sampling(name = 'latent_layer')([self.z_mean, self.z_log_sigma])
        return Model(e[0], latent, name = 'encoder')
        
    def make_decoder(self):
        """
        Generates decoder with depth = len(decoder_dims) + output layer
        last layer is a sigmoid layer
        """
        d =[Input(shape = (self.latent_dim,), name = 'decoder_input')]
        for i in self.decoder_dims:
            d.append(Dense(i, activation = 'relu', kernel_initializer = self.initializer)(d[-1]))
        d.append(Dense(self.encoder_dims[0], activation = 'sigmoid')(d[-1]))
        return  Model(d[0],d[-1], name = 'decoder')
    
    def make_classifier(self):
        """
        Generates classifier with depth = len(classifier_dims)
        last layer is a softmax layer
        """
        c =[Input(shape = (self.latent_dim,), name = 'decoder_input')]
        for i in self.classifier_dims[:-1]:
            c.append(Dense(i, activation = 'relu')(c[-1]))
        c.append(Dense(self.classifier_dims[-1], activation = 'softmax')(c[-1]))
        return  Model(c[0],c[-1], name = 'classifier')
    
    
    def build(self):
        """
        Builds VAE
        for classifier = 'Separate a custom training loop is written that uses
        the encoder, decoder and classifier without compiling them into one model
        so that different training phases can be used
        """
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()
        if self.classifier == 'Inline':
            self.classifier = self.make_classifier()
            self.outputs = [self.decoder(self.encoder(self.inputs)), self.classifier(self.encoder(self.inputs))]
            self.full_vae = Model(self.inputs, self.outputs, name = 'full_vae')
            self.full_vae.summary()
            return self.full_vae.compile(optimizer = RMSprop(decay = .001), loss = [vae_loss(self.beta, self.z_mean, self.z_log_sigma), 'categorical_crossentropy'], loss_weights = self.output_weights) #need to add loss_weights
       
        elif self.classifier == None:
            self.full_vae = Model(self.inputs, self.decoder(self.encoder(self.inputs)), name = 'full_vae')
            self.full_vae.summary()
            return self.full_vae.compile(optimizer = RMSprop(decay = .001), loss = vae_loss(self.beta, self.z_mean, self.z_log_sigma))
    

        
    def traversal_network(self):
        ''' this network is the full_vae after the latent layer.
            it outputs a generated sample and a class prediction per latent input
            this output should be fed back to the full_vae to perform IG on it
            alternatively, the latent vector used in the traversal coud be used
            to do IG in the encoder
            '''
            
    def load_training_data(self):
        """
        loads x_test.pickle and y_test.pickle' as a list
        """
        return [open_file('x_test.pickle'), open_file('y_test.pickle')]

        
        
    
