import os
import glob
import time 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from master.preprocess import *
from master.utils.metrics import *
from master.models import unet
from sys import getsizeof

def main():
    
    #Setup parser
    '''ap = argparse.ArgumentParser()
        ap.add_argument("-r", "--random", type=int, default=22,
            help="random seed")
    args = vars(ap.parse_args())
    #rand = args["random"]'''
    rand = int(os.environ['SLURM_ARRAY_TASK_ID'])

    #Setup log names
    run_name = f'cv_wconst' 
    modeln =  run_name+f'_{rand}'
    os.makedirs(f"./log/{run_name}/",exist_ok=True)
    print(modeln)

    #Setup directory 
    imgs_path = '/work/jflores_umass_edu/data/planet/3k_new/imgs/'
    masks_path = '/work/jflores_umass_edu/data/planet/3k_new/masks/'

    #Set configs
    k_class = 'multi' #binary or multi
    batch_size = 8
    filt = 64


    with tf.device('/CPU:0'):
        #Preprocess dataset
        start_train = time.time()
        X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,k_class)

        print(f"\n\ttrain_imgs size: {getsizeof(X_train)/1000000:.2f} MB")
        print(f"\ttrain_masks size: {getsizeof(y_train)/1000000:.2f} MB")
        print(f'\nProcessing time: {(time.time() - start_train)/60:0.2f} min')


        #Calculate weights -(imbalanced data)
        w_train = get_weights(y_train,k_class) 
        w_test = get_weights(y_test,k_class)
        print('\n\tweights shape (train set): ',w_train.shape)

        
        '''#Prepare tf dataset
        train_data = tf_dataset(X_train,y_train,'train',modeln,batch_size,w_train)
        val_data = tf_dataset(X_test,y_test,'val',modeln,batch_size,w_test)

    #Setup unet and train model
    tf.keras.backend.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = unet(filter=filt) 

    mod_callbacks = [CSVLogger(f'./log/{run_name}/train_{modeln}.csv', append=True), \
                     ModelCheckpoint(f'./log/{run_name}/{modeln}.hdf5',verbose=1, \
                                     monitor='val_one_hot_mean_io_u', mode='max', save_best_only=True)]
    print(model.summary())

    start_train = time.time()
    modfit = model.fit(train_data, verbose=0, #steps_per_epoch=1,  
                       epochs=500, validation_data = val_data, 
                       callbacks=mod_callbacks)
    print(f'\nTraining time: {(time.time() - start_train)/60:0.2f} min')'''

    #For loading models
    tf.keras.backend.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = load_model(f'/work/jflores_umass_edu/hma2/log/{run_name}/{modeln}.hdf5')
        #model = load_model(f'/work/jflores_umass_edu/hma2/log/#old_3k_runs/{run_name}/{modeln}.hdf5')

    with tf.device('/CPU:0'):
        #Predict
        start_train = time.time()
        y_pred = model.predict(X_test)
        y_test,y_pred = pred_array(y_test,y_pred,'multi') #always multi!

        #!!! for multi to binary test only
        if k_class == 'binary':
            y_test, y_pred = multi_to_binary(y_test, y_pred) 

        #Check arrays
        print('\t true: ',y_test.shape)
        print('\t pred: ',y_pred.shape)
        print('\t weights: ',w_test.shape)

        #Get scores     
        cms = cm_score(y_test,y_pred,k_class)
        np.save(f'./log/{run_name}/cm_{modeln}.npy', cms, allow_pickle=True)
        get_scores_df(cms).to_csv(f'./log/{run_name}/metrics_{modeln}.csv')
        
        
        print(f'\nPostprocessing time: {(time.time() - start_train)/60:0.2f} min')


if __name__ == "__main__":
    
    main()