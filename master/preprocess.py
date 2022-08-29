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

def prep_data(img_path,mask_path,val_split,random,kclass='multi'):
    train_images = get_img_arrays(img_path)/255
    print('\timage shape: ',train_images.shape)

    train_masks = get_img_arrays(mask_path,'*png',0)
    if kclass == 'binary':
        train_masks[train_masks==2] = 1
        train_masks = train_masks[:,:,:,np.newaxis]
    elif kclass == 'multi':
        train_masks = get_img_arrays(mask_path,'*png',0)
        train_masks = encode_masks(train_masks)
        
    X_train,X_test,y_train,y_test = train_test_split(train_images, train_masks, test_size = val_split, random_state = random)
    
    print(f'\ttrain: {len(y_train)}, Val: {len(y_test)}, Seed: {random}')
    print('\tmask shape: ',train_masks.shape)
    
    return X_train, X_test, y_train, y_test


    
def get_img_arrays(imgs_fold_path,format_lookup='*tif',channel=1):
    img_list = sorted(glob.glob(imgs_fold_path+format_lookup))
    #images = [cv2.resize(cv2.imread(img_path,channel),dsize=size) for img_path in img_list]
    images = [cv2.imread(img_path,channel) for img_path in img_list]
    return np.array(images).astype('float32')

def encode_masks(mask_arrays):
    labelencoder = LabelEncoder()
    masks_encoded = labelencoder.fit_transform(mask_arrays.reshape(-1,))
    masks_encoded_reshape = masks_encoded.reshape(mask_arrays.shape)[:,:,:,np.newaxis]
    return to_categorical(masks_encoded_reshape, num_classes=3)

def tf_dataset(x, y, cache, name, batch,w=""):
    if len(w)>1:
        data = tf.data.Dataset.from_tensor_slices((x, y,w))
    else:
        w = None
        data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.batch(batch)
    data = data.take(len(x))
    #cache_path = f'./cache/{name}'
    #os.makedirs(cache_path,exist_ok=True)
    #data = data.cache(cache_path+f'/{name}_{cache}')
    return data

def prep_data_rf(X,y, kclass):
    x1 = X[:,:,:,0].reshape(-1)[:,np.newaxis]
    x2 = X[:,:,:,1].reshape(-1)[:,np.newaxis]
    x3 = X[:,:,:,2].reshape(-1)[:,np.newaxis]
    X = np.concatenate((x1,x2,x3),axis=1)
    if kclass == 'binary':
        y = y.reshape(-1)
    if kclass == 'multi':
        y = np.argmax(y,axis=3).reshape(-1)
    return X, y


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
    #pred = np.argmax(pred,axis=3)
    true[true==2] = 1 
    pred[pred==2] = 1 
    y_test = true.reshape(-1)
    y_pred = pred.reshape(-1)
    return y_test, y_pred


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
        class_weights = [0.05,0.35,0.60] 
        
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