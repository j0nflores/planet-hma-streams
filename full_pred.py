# Implement water classification of Planet images (multiclass)
#
# @JAFlores, 11.2021 
# 
# (c)Fluvial@UMass-HiMAT Project


import os
import cv2
import glob
import numpy as np
from master.utils.chips import *
from master.models import *
from master.postprocess import *

def main():
    #Set prediction folder
    pred_fold = './pred/raw_planet_folder'
    pred_path = os.path.dirname(pred_fold) #"./pred"

    #Preprocess and predict images
    batch_chips(pred_path,pred_fold)
    pred_multi(pred_path+'/chips',pred_path)

def chiparray(img_chip_path):
    _image = []
    for img_path in sorted(glob.glob(os.path.join(img_chip_path, "*.tif"))):
        with rasterio.open(img_path,'r') as f:
                imgr = np.moveaxis(f.read(),0,-1)
                _image.append(imgr)
                
    #Convert list to array       
    _image = np.array(_image).astype('float32')/255
    #_image = normalize(_image, axis=1)
    return _image

def pred_multi(chips_folder,output_folder):

    chips_path = sorted(glob.glob(os.path.join(chips_folder,"*")))
    print(chips_path)

    #Run prediction 
    results = []
    for i in range(len(chips_path)):
        pred = chiparray(chips_path[i])
        print(pred.shape)
        model = load_model('./log/cv_mul/cv_multi.hdf5')
        predx = model.predict(pred)#,1,verbose=1)
        results.append(predx)

    #Batch reproject and merge the predicted chips into Planet scene
    tmp_path = f'./{output_folder}/tmp_pred'
    pred_path = output_folder
    
    for i in range(len(chips_path)):
        #Run projection of masks and save to a temporary folder
        proj_pred(chips_path[i],results[i],tmp_path,multi=True)

        #Run merge
        merge_masks(tmp_path,pred_path,chips_path[i])

if __name__ == "__main__":

    main()
