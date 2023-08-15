import os
import glob
import cv2
import time 
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
from sys import getsizeof
import rasterio

def prep_data(img_path,mask_path,val_split,random,kclass='multi',season="all"):
    start_train = time.time()
    
    train_masks = get_img_arrays(mask_path,'*png',im_set='mask')
    if kclass == 'binary':
        train_masks[train_masks==2] = 1
        train_masks = train_masks[:,:,:,np.newaxis]
    elif kclass == 'multi':
        train_masks = get_img_arrays(mask_path,'*png',im_set='mask')
        train_masks = encode_masks(train_masks)
        
    if season == "all":
        train_images = get_img_arrays(img_path)/255
        X_train,X_test,y_train,y_test = train_test_split(train_images, train_masks, test_size=val_split, random_state=random)
    else:
        idx_seas,train_images = get_img_arrays(img_path,season=season)
        idx =  np.arange(len(train_images))   
        X_train,X_test,y_train,y_test,idx_train,idx_test = train_test_split(train_images, train_masks, idx, test_size=val_split, random_state=random)
        X_train = X_train[np.array(idx_seas)[idx_train]==season]/255
        X_test = X_test[np.array(idx_seas)[idx_test]==season]/255
        y_train = y_train[np.array(idx_seas)[idx_train]==season]
        y_test = y_test[np.array(idx_seas)[idx_test]==season]
    
    print('\timage shape: ',train_images.shape)
    print('\tmask shape: ',train_masks.shape)
    print(f'\ttrain: {len(y_train)}, Val: {len(y_test)}, Seed: {random}')
    print(f"\n\ttrain_imgs size: {getsizeof(X_train)/1000000:.2f} MB")
    print(f"\ttrain_masks size: {getsizeof(y_train)/1000000:.2f} MB")
    print(f'\n\tPreprocessing time: {(time.time() - start_train)/60:0.2f} min')
    
    return X_train, X_test, y_train, y_test


def get_img_arrays(imgs_fold_path,format_lookup='*tif',im_set='img',season="all"):
    img_list = sorted(glob.glob(imgs_fold_path+format_lookup))
    dates = [os.path.basename(x)[0:8] for x in img_list]
    
    if im_set == 'mask':
        img_list = [x[0:43]+'masks_rn/'+os.path.basename(x) for x in img_list]      
        images = [cv2.imread(img_path,0) for img_path in img_list]
        images = np.array(images).astype('float32')
        return images
    
    elif im_set == 'img':
        img_list = [x[0:43]+'imgs_rn/'+os.path.basename(x)[:-4]+'.tif' for x in img_list]
        images = []
        for img_path in img_list:
            with rasterio.open(img_path,'r') as f:
                imgr = np.moveaxis(f.read(),0,-1)
            images.append(imgr)
        images = np.array(images).astype('float32')

        if season=="all":
            return images
        else:
            idx_seas = []
            for i in dates:
                month = int(i[4:6])
                if month in [9,10,11]:
                    seas = 'fall'
                elif month in [3,4,5]:
                    seas = 'spring'
                elif month in [6,7,8]:
                    seas = 'summer'
                else:
                    pass
                idx_seas.append(seas)
            return idx_seas, images


def prep_data_rf(X,y, kclass):
    x1 = X[:,:,:,0].reshape(-1)[:,np.newaxis]
    x2 = X[:,:,:,1].reshape(-1)[:,np.newaxis]
    x3 = X[:,:,:,2].reshape(-1)[:,np.newaxis]
    x4 = X[:,:,:,3].reshape(-1)[:,np.newaxis]
    X = np.concatenate((x1,x2,x3,x4),axis=1)
    if kclass == 'binary':
        y = y.reshape(-1)
    if kclass == 'multi':
        y = np.argmax(y,axis=3).reshape(-1)
    return X, y

def encode_masks(mask_arrays):
    labelencoder = LabelEncoder()
    masks_encoded = labelencoder.fit_transform(mask_arrays.reshape(-1,))
    masks_encoded_reshape = masks_encoded.reshape(mask_arrays.shape)[:,:,:,np.newaxis]
    return to_categorical(masks_encoded_reshape, num_classes=3)

def tf_dataset(x, y, cache, name, batch,w=""):
    if len(w)>1:
        data = tf.data.Dataset.from_tensor_slices((x, y, w))
    else:
        w = None
        data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.batch(batch)
    data = data.take(len(x))
    
    #For caching
    #cache_path = f'./cache/{name}'
    #os.makedirs(cache_path,exist_ok=True)
    #data = data.cache(cache_path+f'/{name}_{cache}')
    
    print('\n\t',data.element_spec)
    return data


def get_weights(y,kclass):
    if kclass == 'binary': 
        nclass = 2
        y = y.flatten()
        #class_weights = [0.25,0.75] #for constant weights
    elif kclass == 'multi':
        nclass = 3
        y = np.argmax(y,axis=3)
        #class_weights = [0.05,0.35,0.60] 
    class_weights = compute_class_weight(class_weight = 'balanced',
                                         classes = np.arange(0,nclass),
                                         y = y.flatten())
    class_weights = class_weights/np.sum(class_weights)
    sample_weights = y.astype('float32')
    for i in range(nclass):
        sample_weights[sample_weights==i] = class_weights[i]
    return sample_weights

def get_weights_const(y,kclass):
    if kclass == 'binary': 
        nclass = 2
        y = y.flatten()
        class_weights = [0.25,0.75] #for constant weights
    elif kclass == 'multi':
        nclass = 3
        y = np.argmax(y,axis=3)
        class_weights = [0.05,0.25,0.70] 
        
    sample_weights = y.astype('float32')
    for i in range(nclass):
        sample_weights[sample_weights==i] = class_weights[i]
    return sample_weights

def get_weights_rf(y,kclass):
    if kclass == 'binary': 
        nclass = 2
    elif kclass == 'multi':
        nclass = 3
    class_weights = compute_class_weight(class_weight = 'balanced',
                                         classes = np.arange(0,nclass),
                                         y = y)
    class_weights = class_weights/np.sum(class_weights)
    sample_weights = y.astype('float32')
    for i in range(nclass):
        sample_weights[sample_weights==i] = class_weights[i]
    return sample_weights