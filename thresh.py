import os
import time
import pandas as pd
from sys import getsizeof
from master.utils.metrics import *
from master.models import *
from master.preprocess import *
from master.config import *

with tf.device('/CPU:0'): 

    #Preprocess dataset
    start_train = time.time()
    X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,kclass='binary')

    #Run thresholding
    start_train = time.time()
    y_test,y_pred = get_threshold(X_test,y_test, method=method)
    print(f'\nTime elapsed: {(time.time() - start_train)/60:0.2f} min')
    
    #Postprocess
    y_test,y_pred = pred_array(y_test,y_pred,'binary') 

    #Get scores       
    cms = cm_score(y_test,y_pred,'binary')
    np.save(f'./log/{run_name}/cm_{modeln}.npy', cms, allow_pickle=True)
    get_scores_df(cms).to_csv(f'./log/{run_name}/metrics_{modeln}.csv')

    
    