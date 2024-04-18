# Implement water classification for PlanetScope SR imagery
#
# @JAFlores
# (c)Fluvial@UMass-HiMAT Project


import os
import cv2
import glob
import numpy as np
from master.models import *
from master.postprocess import *
from master.utils.chips import *

def pred_multi(output_folder):
    
    chips_folder = f'{out_fold}/chips' 
    chips_path = sorted(glob.glob(os.path.join(chips_folder,"*")))
    print(f'\nNumber of chips folder to predict: {len(chips_path)}')

    #Run cv prediction 
    results = []
    for i in range(len(chips_path)):
        with tf.device('/CPU:0'):
            pred_arr = chiparray(chips_path[i])
        #print(pred.shape)
        model = load_model('./log/cv_mul/cv_multi.hdf5')
        pred_cv = model.predict(pred_arr)#,1,verbose=1)
        results.append(pred_cv)
    
        #clear memory space
        pred_arr = None
        pred_cv = None

    #Project and merge the predicted chips into Planet scene
    #run projection of masks and save to a temporary folder
    with tf.device('/CPU:0'):
        tmp_path = f'./{output_folder}/tmp_pred'
        for i in range(len(chips_path)):
            proj_pred(chips_path[i],results[i],tmp_path,multi=True)

            #Run merge
            merge_masks(tmp_path,output_folder,chips_path[i])
            shutil.rmtree(chips_path[i])
        
    #cleanup chips folder
    os.rmdir(chips_folder)


if __name__ == "__main__":

    #input folder containing raw planet SR scenes (4-band)
    #make sure that this folder only contains SR tif files
    pred_fold = './data/pred/raw_planet' 

    #output folder for predicted scenes
    out_fold = './pred_out'
    os.makedirs(out_fold,exist_ok=True)

    #Preprocess image and extract chips
    batch_chips(out_fold,pred_fold)
    
    #predict chips, transform and mosaic to full scene
    #use GPU for faster large-scale processing
    pred_multi(out_fold)
    
    
    