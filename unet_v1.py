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
from master.models import build_unet
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
    run_name = f'hma_weighted_target' #'hma_b16lr5e500'
    modeln =  run_name+f'_{rand}'
    os.makedirs(f"./log/{run_name}/",exist_ok=True)
    print(modeln)

    #Setup directories 
    imgs_path = '/work/jflores_umass_edu/planet-rivers/data/3k/imgs/'
    masks_path = '/work/jflores_umass_edu/planet-rivers/data/3k/masks/'

    #Set configs
    batch_size = 16
    k_class = 'binary' #binary or multi


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


        
        '''#Prepare tf dataset
        train_data = tf_dataset(X_train,y_train,'train',modeln,batch_size,w_train)
        val_data = tf_dataset(X_test,y_test,'val',modeln,batch_size,w_test)

    #Setup unet and train model
    tf.keras.backend.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_unet() 

    mod_callbacks = [CSVLogger(f'./log/{run_name}/{modeln}_log.csv', append=True), \
                     ModelCheckpoint(f'./log/{run_name}/{modeln}.hdf5',verbose=1, \
                                     monitor='val_one_hot_mean_io_u', mode='max', save_best_only=True)]
    #model.summary()

    start_train = time.time()
    modfit = model.fit(train_data, verbose=0, #steps_per_epoch=1,  
                       epochs=500, validation_data = val_data, 
                       callbacks=mod_callbacks)
    print(f'\nTraining time: {(time.time() - start_train)/60:0.2f} min')'''

    #For loading models
    tf.keras.backend.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        #model = load_model(f'/work/jflores_umass_edu/hma/log/{run_name}/{modeln}/{modeln}.hdf5') #uw model
        model = load_model(f'/work/jflores_umass_edu/hma2/log/{run_name}/{modeln}.hdf5')

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
        scores = get_scores(y_test,y_pred,None,w_test)   
        means = get_score_means(y_test,y_pred,w_test)
        print(f'\nPostprocessing time: {(time.time() - start_train)/60:0.2f} min')

        #Write files
        pd.DataFrame(scores).to_csv(f'./log/{run_name}/metrics_{k_class}_{modeln}.csv')
        means.to_csv(f'./log/{run_name}/means_{k_class}_{modeln}.csv')
        #np.save(f'./log/{run_name}/{modeln}.npy',get_cm(y_test,y_pred,w_test,'multi'), allow_pickle=True)
        



if __name__ == "__main__":
    
    main()