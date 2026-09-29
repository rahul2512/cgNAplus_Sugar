import torch as tr 
from torch import nn, sigmoid, tanh,relu
import tensorflow as tf
import numpy as np
from torch.nn import Linear 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorflow import keras
import sys
from keras.regularizers import l2
from keras.callbacks import CSVLogger



#### Code below generates a .txt file with row as the list of hypermeters 
#### for a given NN and then that NN hypermeters were cross-validated on cluster
def hyper_param():
# batch-size {Automatic, 64, 1000, 2000}
# epoch = 40
# Adam, SGD, RMSprop
# mse
# relu, tanh, sigmoid
# random_uniform, random_normal, he_normal, xavier, glorot_uniform, glorot_normal (Xavier), 
    with open('hyperparam.txt', 'w') as f:
        print('optim', 'kinit', 'batch_size', 'epoch', 'act', 'num_nodes', 'H_layer', 'metric', 'loss', 'lr', 'p','regularizer_val', file=f)
        for optim in ['Adam']:
            for kinit in ['glorot_normal']:
                for batch_size in [32]:
                    for epoch in [200]:
                        for act in ['relu','sigmoid',]:
                            for H_layer in [4,6,8,10]:
                                for metric in ['mae']:
                                    for loss in ['mse','mae','huber']:
                                        for lr in [0.001,0.0005]:
                                            for p in [0,0.2,0.4]:
                                                for num_nodes in [400,600,800,1200,1800]:
                                                    for reg in [0]:
                                                        print(optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p,reg, file=f)
    return None


def optimizer(opt):
    if opt == 'Adam':
        optim = keras.optimizers.Adam
    elif opt == 'RMSprop':
        optim = keras.optimizers.RMSprop
    elif opt == 'SGD':
        optim = keras.optimizers.SGD
    return optim

def closs_1(y_true, y_pred):
	ls1 = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)),1)
	ls2 = tf.norm(tf.subtract(y_pred[:,6:9],y_pred[:,9:12])) - 1.526
	ls = ls1 + tf.reduce_mean(tf.square(ls2),1)
	return ls

### Initite NN model
def initiate_NN_model(inp_dim,out_dim,nbr_Hlayer,Neu_layer,activation,p_drop,lr,optim,loss,metric,kinit,final_act,regularizer_val):
    optim = optimizer(optim)
    model = keras.Sequential()
    model.add(keras.layers.Dense(Neu_layer, input_shape=(inp_dim,), activation=activation))
    for i in range(nbr_Hlayer):
        model.add(keras.layers.Dense(Neu_layer, activation=activation,kernel_initializer=kinit))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(p_drop))
    model.add(keras.layers.Dense(out_dim, activation=final_act))
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=metric)
    return model

def initiate_LM_model(inp_dim,out_dim,nbr_Hlayer,Neu_layer,activation,p_drop,lr,optim,loss,metric,kinit,final_act,regularizer_val):
    #### rest of the parameters are redundnt but kept for generalisibilty of code
    optim = optimizer(optim)
    model = keras.Sequential()
    model.add(keras.layers.Dense(out_dim, input_shape=(inp_dim,), activation='linear'))
    try:
        opt = optim(learning_rate=lr)
    except:
        opt = optim(lr=lr)
    model.compile(loss=closs_1, optimizer=opt, metrics=metric)
    print("Initialised linear regression network")
    return model

def run_model(data):
    inp_dim = data.X_Train.shape[1]
    out_dim = data.Y_Train.shape[1]
    print("Output are reduced using input param, ------------------")
    print("input, output dimensions --- ", inp_dim, out_dim)
    print(data.hyper_value)
    opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p , regularizer_val =   data.hyper_value
    #inp_dim, out_dim, nbr_Hlayer, Neu_layer, activation, p_drop, lr, optim,loss,metric,kinit
    model = data.model
    csv_logger = CSVLogger('Logs/'+data.save_path+'.csv', append=True, separator=';')
    history = model.fit(data.X_Train, data.Y_Train[data.param], validation_data = (data.X_val,data.Y_val[data.param]),epochs=epoch, batch_size=batch_size, verbose=2,shuffle=True,callbacks=[csv_logger])
    return model

def save_model(data):
    model = data.model
    model.save('./models/'+data.save_path+'.h5')
    return None

def load_model(load_path):
    model = keras.models.load_model(load_path)
    return model

def init_model(data):
    optim, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p , regularizer_val =   data.hyper_value
    inp_dim, out_dim = data.inp_dim, data.out_dim
    final_act = None
    loss = keras.losses.mean_squared_error
    if data.model_type in ['NN']:
        model = initiate_NN_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif data.model_type in ['LM']:
        model = initiate_LM_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif data.model_type in ['load']:
        if data.load_path != None:
            model = load_model(data.load_path)

    return model


