import os
import pickle
import pandas as pd
from master.preprocess import *
from master.postprocess import *
from master.models import *
from master.utils import *
from master.config import *

with tf.device('/CPU:0'):

    #Preprocess images
    X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,k_class)
    X_train, y_train = prep_data_rf(X_train,y_train,k_class)
    X_test, y_test = prep_data_rf(X_test,y_test,k_class)

    if mode=='train':
        # Fit model to training data - train on k_class == 'multi'
        start_train = time.time()
        rf = get_rf(X_train,y_train,tree, depth, weights)   
        with open(f'./log/{run_name}/{modeln}.pkl','wb') as f:
            pickle.dump(rf,f)
        np.save(f'./log/{run_name}/importance_{modeln}.npy', 
                rf.feature_importances_, allow_pickle=True)   
        print(f'\tTraining time: {(time.time() - start_train)/60:0.2f} min')

    if mode=='predict':
        #Load model (if already trained)
        with open(f'./log/{run_name}/{modeln}.pkl', 'rb') as f:
            rf = pickle.load(f)

    #Predict and postprocess
    start_train = time.time()
    y_pred = rf.predict(X_test)

    #for multi to binary test
    if m2b == True:
        y_test, y_pred = multi_to_binary(y_test, y_pred)

    #Get scores     
    cms = cm_score(y_test,y_pred,k_class,m2b=m2b)
    np.save(f'./log/{run_name}/cm_{k_class}_{modeln}.npy', cms, allow_pickle=True)
    get_scores_df(cms).to_csv(f'./log/{run_name}/metrics_{k_class}_{modeln}.csv')