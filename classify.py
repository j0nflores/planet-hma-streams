import os
import time 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from master.preprocess import *
from master.utils import *
from master.models import unet


def main():
    #CONFIG
    
    #Setup parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--method", type=str, default='cv',
        help="classification methods: rf,cv")
    ap.add_argument("-c", "--classes", type=str, default='multi',
        help="number of classes")
    ap.add_argument("-w", "--weights", type=str, default=None,
        help="class weights")
    args = vars(ap.parse_args())
    method = args["method"]
    else:
        k_class = args["classes"]
        
    rand = int(os.environ['SLURM_ARRAY_TASK_ID'])

    #Setup log directory
    run_name = f'cv_best_rerun_orig' 
    modeln =  run_name+f'_{rand}'
    os.makedirs(f"./log/{run_name}/",exist_ok=True)
    print(modeln)


    #Setup directory 
    imgs_path = '/work/jflores_umass_edu/data/planet/3k_new/imgs/'
    masks_path = '/work/jflores_umass_edu/data/planet/3k_new/masks/'
    

    #PREPROCESS
    
    with tf.device('/CPU:0'):
        #Preprocess dataset
        start_train = time.time()
        X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,k_class)
        print(f'\nPreprocessing time: {(time.time() - start_train)/60:0.2f} min')
    
    
    if method == 'otsu':
    elif method == 'thresh':
    elif method == 'rf':
    elif method == 'cv':
        #w_train = get_weights_const(y_train,k_class)
        #w_test = get_weights_const(y_test,k_class)


        '''#Prepare tf dataset
        train_data = tf_dataset(X_train,y_train,'train',modeln,batch_size)#,w_train)
        val_data = tf_dataset(X_test,y_test,'val',modeln,batch_size)#,w_test)
        
        print(train_data.element_spec)

    #METHOD
    #Setup unet and train model
    tf.keras.backend.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = unet() 

    mod_callbacks = [CSVLogger(f'./log/{run_name}/train_{modeln}.csv', append=True), \
                     ModelCheckpoint(f'./log/{run_name}/{modeln}.hdf5',verbose=1, \
                                     monitor='val_one_hot_mean_io_u', mode='max', save_best_only=True)]
    #print(model.summary())

    start_train = time.time()
    modfit = model.fit(train_data, verbose=0, #steps_per_epoch=1,  
                       epochs=1000, validation_data = val_data, 
                       callbacks=mod_callbacks)
    print(f'\nTraining time: {(time.time() - start_train)/60:0.2f} min')'''

    #For loading models
    tf.keras.backend.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        #model = load_model(f'/work/jflores_umass_edu/hma/log/{run_name}/{modeln}/{modeln}.hdf5')
        model = load_model(f'/work/jflores_umass_edu/hma2/log/{run_name}/{modeln}.hdf5')
        #model = load_model(f'/work/jflores_umass_edu/hma2/log/#old_3k_runs/{run_name}/{modeln}.hdf5')
       

    #PREDICT & POSTPROCESS
    with tf.device('/CPU:0'):
        #Predict
        start_train = time.time()
        y_pred = model.predict(X_test)
        y_test,y_pred = pred_array(y_test,y_pred,'multi') #always multi!

        #!!! for multi to binary test only
        if k_class == 'binary':
            y_test, y_pred = multi_to_binary(y_test, y_pred) 
     
    #METRICS
        cms = cm_score(y_test,y_pred,k_class)
        np.save(f'./log/{run_name}/cm_{k_class}_{modeln}.npy', cms, allow_pickle=True)
        get_scores_df(cms).to_csv(f'./log/{run_name}/metrics_{k_class}_{modeln}.csv')
        
        print(f'\nPostprocessing time: {(time.time() - start_train)/60:0.2f} min')


if __name__ == "__main__":
    
    main()