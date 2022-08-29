import time
import os
from sys import getsizeof
from master.utils.metrics import *
from master.models import *
from master.preprocess import *
import pandas as pd



def main():
    
    #Setup parser
    rand = int(os.environ['SLURM_ARRAY_TASK_ID'])
    
    #thresholding method (simple or otsu)
    method = 'simple'

    #Setup directory
    imgs_path = '/work/jflores_umass_edu/data/planet/3k_new/imgs/'
    masks_path = '/work/jflores_umass_edu/data/planet/3k_new/masks/'

    #Setup log names
    run_name = f'thresh_{method}'
    modeln =  run_name+f'_{rand}'
    os.makedirs(f"./log/{run_name}/",exist_ok=True)
    print(modeln)

    with tf.device('/CPU:0'): 

        #Preprocess dataset
        start_train = time.time()
        X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,'binary')
        print(f"\ntrain_imgs size: {getsizeof(X_train)/1000000:.2f} MB")
        print(f"train_masks size: {getsizeof(y_train)/1000000:.2f} MB")
        print(f'\nProcessing time: {(time.time() - start_train)/60:0.2f} min')

        #Calculate weights -(imbalanced data)
        w_train = get_weights(y_train,'binary')
        w_test = get_weights(y_test,'binary')

        #Run thresholding
        start_train = time.time()
        y_test,y_pred = get_threshold(X_test,y_test, method=method)
        y_test,y_pred = pred_array(y_test,y_pred,'binary') 
        
        
        #Get scores 
        scores = get_scores(y_test,y_pred,None,w_test)   
        means = get_score_means(y_test,y_pred,w_test)
        print(f'\nPostprocessing time: {(time.time() - start_train)/60:0.2f} min')

        #Write files        
        cms = cm_score(y_test,y_pred,'binary')
        np.save(f'./log/{run_name}/cm_{modeln}.npy', cms, allow_pickle=True)
        get_scores_df(cms).to_csv(f'./log/{run_name}/metrics_{modeln}.csv')
        
        
        print(f'\nPostprocessing time: {(time.time() - start_train)/60:0.2f} min')
    
if __name__ == "__main__":
    
    main()