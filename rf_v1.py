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
    
    tree = 100 #int(os.environ['SLURM_ARRAY_TASK_ID']) #args["trees"]
    depth = int(os.environ['SLURM_ARRAY_TASK_ID'])  #args["depth"]
    rand = 1 #int(os.environ['SLURM_ARRAY_TASK_ID'])  #args["random"]
    
    
    #Setup log names
    run_name = f'rf_d'
    modeln =  run_name+f'_{depth}'
    
    #Set configs
    k_class = 'multi' #binary or multi
    
    #Setup directory
    imgs_path = '/work/jflores_umass_edu/data/planet/3k_new/imgs/'
    masks_path = '/work/jflores_umass_edu/data/planet/3k_new/masks/'
    os.makedirs(f"./log/{run_name}/",exist_ok=True)
    print(run_name)
    
    with tf.device('/CPU:0'):
        
        #Preprocess images
        start_train = time.time()
        X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,k_class)
        X_train, y_train = prep_data_rf(X_train,y_train,k_class)
        X_test, y_test = prep_data_rf(X_test,y_test,k_class)
        print(f'\tPreprocessing time: {(time.time() - start_train)/60:0.2f} min')
        
        
        #Calculate weights -(imbalanced data)
        w_test = get_weights_rf(y_test,k_class)
        
        # Fit model to training data - train on k_class == 'multi'
        start_train = time.time()
        rf = get_rf(X_train,y_train,tree, depth)   #(X_train,y_train)
        '''with open(f'./log/{run_name}/{modeln}.pkl','wb') as f:
            pickle.dump(rf,f)
        np.save(f'./log/{run_name}/importance_{modeln}.npy', 
                rf.feature_importances_, allow_pickle=True)   ''' 
        print(f'\tTraining time: {(time.time() - start_train)/60:0.2f} min')

        '''#Load model (if already trained)
        with open(f'/work/jflores_umass_edu/hma/log/{run_name}/{modeln}.pkl', 'rb') as f:
            rf = pickle.load(f)'''

        #Predict and postprocess
        start_train = time.time()
        y_pred = rf.predict(X_test)
        #y_test, y_pred = multi_to_binary(y_test, y_pred) #only for multi to binary test
            
        #Get scores 
        scores = get_scores(y_test,y_pred,None,w_test)   
        means = get_score_means(y_test,y_pred,w_test)
        print(f'\nPostprocessing time: {(time.time() - start_train)/60:0.2f} min')

        #Write files
        pd.DataFrame(scores).to_csv(f'./log/{run_name}/metrics_{k_class}_{modeln}.csv')
        means.to_csv(f'./log/{run_name}/means_{k_class}_{modeln}.csv')
        #plot_rf_importance(f'./log/{run_name}/importance_{modeln}.png', rf)
        

if __name__ == "__main__":
    
    main()
