#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:46:47 2024

@author: IWAS\choaak
"""
import h5py
import numpy as np
# from scalib.metrics import SNR


# import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization , Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()

MODEL_FOLDER = ''

Reverse_AES_Sbox = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a,0x6b,                                                                                                                                                          
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]



## Training MLP for AES_HD_Ext dataset############################################
def mlp_task(input_size, classes = 256, name = '', summary = True):
    
    input_dict = {}
    input_x = Input( shape = ( input_size,), name = 'input_x')
    input_dict['input_x'] = input_x

    norm_in = BatchNormalization(axis = 1)(input_x)
    x = Dense(  64, activation = 'selu')(norm_in)
    x = BatchNormalization()(x)
    x = Dense(  32, activation = 'selu')(x)
    x = BatchNormalization()(x)
    x = Dense(  16, activation = 'selu')(x)
    x = BatchNormalization()(x)
    


    # Classification layer
    x = Dense(4, activation = 'selu', name = "fc1")(x) 
    
    # Logits layer
    x = Dense(classes, activation = 'softmax', name = 'predictions')(x)
    
    outputs = {}
    metrics = {}

    outputs['x'] = x
    metrics['x'] = 'accuracy'

    losses = {'x' : 'categorical_crossentropy'}


    model = Model(input_dict , outputs = outputs , name = 'aes_hd_model')

    # if summary:
    #     model.summary()
    # learning_rate = 0.00008
    learning_rate = 0.0001
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
    
    return model

def get_hot_encode(label_set, classes = 256):
    return np.eye(classes)[label_set]

def train_mlp(X_train,Y_train, N_prof, N_points = 10):
    epochs = 15
    batch_size = 300
   
    
    training_split = int(N_prof*0.9)
    validation_split = int(N_prof*0.1)
    
    X_prof_dict = {}  
    X_prof_valid = {} 
    
    # Standardization and Normalization--------------------------------
    X_prof = X_train[:training_split]
    X_valid = X_train[training_split:training_split + validation_split]
    
    # scaler = preprocessing.StandardScaler()
    # X_prof = scaler.fit_transform(X_prof)
    # X_valid = scaler.transform(X_valid)

    # scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    # X_prof = scaler.fit_transform(X_prof)
    # X_valid = scaler.transform(X_valid)
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))


    X_prof_dict['input_x'] = X_prof
    X_prof_valid['input_x'] = X_valid

    Y_prof_dict = {}
    Y_valid_dict = {}
    Y_prof_dict['x'] = get_hot_encode(Y_train[:training_split])
    Y_valid_dict['x'] =  get_hot_encode(Y_train[training_split : training_split + validation_split])
    
    X_profiling , X_validation = tf.data.Dataset.from_tensor_slices( (X_prof_dict, Y_prof_dict)) ,  tf.data.Dataset.from_tensor_slices( (X_prof_valid, Y_valid_dict))
    
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    X_validation = X_validation.batch(batch_size)
    
    
    file_name = 'aes_hd_model{}'.format(N_prof) 
    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                filepath= ''+ file_name +'.h5',
                                save_weights_only=True,
                                save_freq='epoch'
                               )
    model = mlp_task(N_points)
     
    history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs = epochs,
            validation_data =X_validation, callbacks =callbacks)
    
    return model



# Loading trained model-------------------------------------------

def load_model_(input_length, N_prof):
    model = mlp_task(input_length)
    file_name = 'aes_hd_model{}'.format(N_prof)
    model.load_weights(MODEL_FOLDER + file_name +'.h5')
    return model



def run_lr(N_prof, N_val, trace_prof, label_prof, trace_val, label_val):
    """ 
    Profiling and calculating the log_probability score using MLP model
    
    """
    len_prof = len(label_prof)
    sample_idx =  np.random.choice( len_prof , N_prof, replace = False)
    X_train = trace_prof[sample_idx]
    Y_train = label_prof[sample_idx]
    
    len_val = len(label_val)
    smpl_idx =  np.random.choice( len_val , N_val, replace = False)
    X_val = trace_val[smpl_idx]
    Y_val = label_val[smpl_idx]
    # -------------------------------------------
    ''' Consider the MLP training'''
    # train_model(X_train, Y_train, X_val, Y_val)
    # model = load_model(X_train.shape[1])
    
    train_mlp(X_train,Y_train, N_prof)
    model = load_model_(X_train.shape[1], N_prof)
    
    # # Predicts the scores
   
    
    predictions_val = model.predict(X_val, verbose = 0)['x']
    predictions = model.predict(X_train, verbose = 0)['x']
    
    pred_v_tmp  = np.where(predictions_val > 1.0e-10, predictions_val, 1.0e-10)
    # result_val = np.where(predictions_val > 1.0e-10, np.log(pred_v_tmp), -2)
    
    pred_tmp  = np.where(predictions > 1.0e-10, predictions, 1.0e-10)
    # result_ = np.where(predictions > 1.0e-10, np.log2(pred_tmp), -2)
    scores_lr = 0
    scores_lr_prof = 0
    
    for i in range(256):
        idx_train = np.where(Y_train == i)
        idx_val = np.where(Y_val == i)
     
        if len(idx_val) > 0: 
            x_score =  np.log(pred_v_tmp[ idx_val, i])
            scores_lr += np.mean(x_score)
            pass
        if len(idx_train) > 0: 
            x_prof_score = np.log(pred_tmp[ idx_train, i])
            scores_lr_prof += np.mean(x_prof_score)
            pass
        pass      

    scores_lr = scores_lr / 256
    scores_lr_prof = scores_lr_prof / 256           
    # -------------------------------------------    

    return scores_lr, scores_lr_prof


def compute_PI_TI(N_prof_range, N_val, trace_prof, label_prof, trace_val, label_val):
    
    PIs_lr, TIs_lr = [], []
    for N_prof in N_prof_range:
        PI_lr = 0
        TI_lr = 0
        n_exp = 10
        
        for _ in range(n_exp):
            score_lr, score_lr_prof = run_lr(N_prof, N_val, trace_prof, label_prof, trace_val, label_val)
            PI_lr += 8 + score_lr
            TI_lr += 8 + score_lr_prof
            
        PIs_lr.append(PI_lr/n_exp)
        TIs_lr.append(TI_lr/n_exp)
    
    PIs_lr = np.array(PIs_lr)
    TIs_lr = np.array(TIs_lr)
    
    return PIs_lr, TIs_lr



if __name__ == "__main__":
    
    TS = np.load('aes_hd_ext.npz')
    
    intermediate = np.array( TS['data'][:, 16:32], dtype = np.uint8)
    
    intermediate_1 = np.array(intermediate[:, 11], dtype = np.uint16)
    intermediate_2 = np.array(intermediate[:, 7], dtype = np.uint16)
    
    key = np.array( TS['data'][:, 32:], dtype = np.uint8)[:, 11]
    
    Trace = np.array(TS['traces'], dtype = np.int16)
    
    n_trace = len(Trace)
    X = np.array([ Reverse_AES_Sbox[intermediate_1[i] ^ key[0]]^intermediate_2[i]
                  for i in np.arange(0, n_trace)], dtype = np.uint16)
    
    idx = [539, 540, 542, 543, 965, 966, 967, 968, 969, 1021]
    T_prof = Trace[: int(4.5e05) ]
    Y_prof = X[: int(4.5e05)]
    T_valid = Trace[ int(4.5e05): ]
    Y_valid = X[ int(4.5e05):]
    

    fontsize = 8
    fig_width = 5
    
    # N_prof_range = np.logspace(3.5, 5.5, 20).astype('int32')
    N_prof_range = [316227]
    n_repeats = 1
    PI_mean, TI_mean = [], []
    for _ in range(n_repeats):
        PIs_lr, TIs_lr = compute_PI_TI(N_prof_range, int(5e04), T_prof[:, idx], Y_prof, T_valid[:, idx], Y_valid)
        
        PI_mean.append(PIs_lr)
        TI_mean.append(TIs_lr)


    PI_mean = np.array(PI_mean).mean(axis=0)
    TI_mean = np.array(TI_mean).mean(axis=0)
    
    print(f" PI_values {PI_mean}")
    print(f" TI_values {TI_mean}") 
    # np.save('TI_PI_MLP.npy', [N_prof_range, PI_mean, TI_mean])    




