import os
import argparse 
import pandas as pd
from master.preprocess import *
from master.models import *
from master.utils import *
import pickle

def main():
    
    #Setup parser
    '''ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--trees", type=int, default=100, help="rf n trees")
    ap.add_argument("-d", "--depth", type=int, default=5, help="rf n depth")
    ap.add_argument("-r", "--random", type=int, default=22, help="random data")
    args = vars(ap.parse_args())'''
    
    tree = 200 #int(os.environ['SLURM_ARRAY_TASK_ID']) #args["trees"]
    depth = 9 # int(os.environ['SLURM_ARRAY_TASK_ID'])  #args["depth"]
    rand = int(os.environ['SLURM_ARRAY_TASK_ID'])  #args["random"]
    
    
    #Setup log names
    run_name = f'rf_wc_t{tree}d{depth}'    
    
    #Set configs
    k_class = 'multi' #binary or multi
    weights = {0:0.05, 1:0.25, 2:0.70} #'balanced' #
    
    #Setup directory
    imgs_path = '/work/jflores_umass_edu/data/planet/3k_new/imgs/'
    masks_path = '/work/jflores_umass_edu/data/planet/3k_new/masks/'
    modeln =  run_name+f'_{rand}'
    os.makedirs(f"./log/{run_name}/",exist_ok=True)
    print(run_name)
    print('t ',tree, 'd ', depth, 'r ', rand, 'weights: ',weights)
    
    with tf.device('/CPU:0'):
        
        #Preprocess images
        start_train = time.time()
        X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,k_class)
        X_train, y_train = prep_data_rf(X_train,y_train,k_class)
        X_test, y_test = prep_data_rf(X_test,y_test,k_class)
        print(f'\tPreprocessing time: {(time.time() - start_train)/60:0.2f} min')
        
        
        # Fit model to training data - train on k_class == 'multi'
        start_train = time.time()
        rf = get_rf(X_train,y_train,tree, depth, weights)   
        with open(f'./log/{run_name}/{modeln}.pkl','wb') as f:
            pickle.dump(rf,f)
        np.save(f'./log/{run_name}/importance_{modeln}.npy', 
                rf.feature_importances_, allow_pickle=True)   
        print(f'\tTraining time: {(time.time() - start_train)/60:0.2f} min')

        '''#Load model (if already trained)
        with open(f'/work/jflores_umass_edu/hma2/log/{run_name}/{modeln}.pkl', 'rb') as f:
            rf = pickle.load(f)'''

        #Predict and postprocess
        start_train = time.time()
        y_pred = rf.predict(X_test)
        
        #!!! for multi to binary test only
        if k_class == 'binary':
            y_test, y_pred = multi_to_binary(y_test, y_pred) #only for multi to binary test
            
        #Get scores     
        cms = cm_score(y_test,y_pred,k_class)
        np.save(f'./log/{run_name}/cm_{k_class}_{modeln}.npy', cms, allow_pickle=True)
        get_scores_df(cms).to_csv(f'./log/{run_name}/metrics_{k_class}_{modeln}.csv')
        
        print(f'\nPostprocessing time: {(time.time() - start_train)/60:0.2f} min')
        

if __name__ == "__main__":
    
    main()
