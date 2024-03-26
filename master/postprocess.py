#Functions to reproject and merge predicted chips


import cv2
import os
import glob
import numpy as np
from osgeo import gdal
import shutil


def postprocess_cv(true,pred,kclass,m2b=False,model='multi'):

    if model == 'binary':
        pred = pred > 0.2
        
    y_test,y_pred = pred_array(true,pred,kclass) 

    #for multi to binary test 
    if m2b == True:
        y_test, y_pred = multi_to_binary(y_test, y_pred)
 
    return y_test,y_pred
        
def pred_array(true,pred,kclass='multi'):
    if kclass == 'binary':
        true = true.reshape(-1)
        pred = pred.reshape(-1).astype('bool')
    elif kclass == 'multi':
        if true.shape[3] == 3:
            true = np.argmax(true,axis=3).reshape(-1)
        else:
            true.reshape(-1)
        pred = np.argmax(pred,axis=3).reshape(-1)
    return true, pred


# for multi to binary val test
def multi_to_binary(true,pred):
    true[true==2] = 1 
    pred[pred==2] = 1 
    y_test = true.reshape(-1)
    y_pred = pred.reshape(-1)
    return y_test, y_pred


def proj_pred(img_path, predicted_array, tmp_pred_path,multi=False):

    #Write the masked image file and convert array to boolean 0 to 1
    flist = sorted(glob.glob(img_path+"/*.tif"))       
    
    for i in range(len(flist)):
        
        #setup filenames
        fn = os.path.basename(flist[i][:-4])
        out_fn = fn+".tif"
        
        #create temp tif folder
        out_dir = os.path.join(tmp_pred_path,out_fn)
        os.makedirs(tmp_pred_path,exist_ok=True)
        
        #Get pred arrays
        if multi == True:
            arr_pred = np.argmax(predicted_array[i], axis=2)[:,:,np.newaxis]
            arr_pred[arr_pred==2] = 1
            
            #cleanup predicted chip array from raw chip and export
            mask = cv2.imread(flist[i])
            mask = mask[:,:,0].reshape(arr_pred.shape)
            arr_pred = np.ma.masked_array(arr_pred, np.logical_not(mask)).filled(0)
            cv2.imwrite(out_dir,arr_pred)
            arr_pred = None

        else:
            arr_pred = (predicted_array[i]>.3)*255
            arr_pred = np.squeeze(arr_pred,axis=2)
            Image.fromarray(arr_pred.astype('uint8')).save(out_dir) #fix for speed
            arr_pred = None
        #print("Created " +str(out_fn) + " to ../pred folder")

        #Get geoinfo of original image chip
        dataset = gdal.Open(flist[i])
        projection   = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        dataset = None

        #Copy geoinfo to water mask
        dataset2 = gdal.Open(out_dir,gdal.GA_Update)
        dataset2.SetGeoTransform(geotransform)
        dataset2.SetProjection(projection)
        dataset2.GetProjection()
        dataset2 = None
        
        
def merge_masks(mask_path,out_path,img_path):
    #setup paths
    os.makedirs(out_path,exist_ok=True)

    mask_list = glob.glob(mask_path+"/*.tif") 
    img = os.path.basename(img_path)
    out_fn = os.path.join(out_path,img+"_mask.tif")


    # build virtual raster mosaic and create geotiff
    vrt = gdal.BuildVRT(f"{mask_path}/merged.vrt", mask_list)
    options_str = '-ot Byte -of GTiff' 
    gdal.Warp(out_fn, vrt, options=options_str)
    print('\nCreated',img+'_mask in output folder')
    vrt = None

    #delete vrt and temporary masks folder
    #os.remove(f"{mask_path}/{img}_merged.vrt")
    shutil.rmtree(mask_path) 
    
