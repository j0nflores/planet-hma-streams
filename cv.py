import os
import time 
import numpy as np
import pandas as pd
import tensorflow as tf
from master.preprocess import *
from master.postprocess import *
from master.models import *
from master.utils import *
from master.config import *


#Preprocess dataset
with tf.device('/CPU:0'):
    X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,k_class)
    train_data = tf_dataset(X_train,y_train,'train',modeln,batch_size)
    val_data = tf_dataset(X_test,y_test,'val',modeln,batch_size)

if mode == 'train':    
    #Setup unet and train model
    tf.keras.backend.clear_session()
    with tf.distribute.MirroredStrategy().scope():
        model = unet(nclass=nclass) 

    start_train = time.time()
    model.fit(train_data, verbose=0, epochs=500, 
              validation_data = val_data, callbacks=mod_callbacks)
    print(f'\nTraining time: {(time.time() - start_train)/60:0.2f} min')


if mode == 'predict':
    #For loading models
    model = cv_load(run_name,modeln)


#Postprocess
with tf.device('/CPU:0'):
    y_pred = model.predict(X_test)
    y_test,y_pred = postprocess_cv(y_test,y_pred,k_class,m2b,model=k_class)

    #Get scores     
    cms = cm_score(y_test,y_pred,k_class)
    get_scores_df(cms).to_csv(f'./log/{run_name}/metrics_{k_class}_{modeln}.csv')
    np.save(f'./log/{run_name}/cm_{k_class}_{modeln}.npy', cms, allow_pickle=True)