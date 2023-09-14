import cv2
import numpy as np
from .utils import *
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import *
import tensorflow as tf 


def get_threshold(X,y, method='simple'):
    true_array, pred_array, th_otsu = [],[],[]    
    for i in range(len(X)):

        gray_image = get_ndwi(X[i])
        
        if method == 'simple':
            th = gray_image > 0 #threshold (Xu, 2006, Huang et al., 2018)
            
        elif method == 'otsu':
            th_val = threshold_otsu(gray_image)
            th = gray_image > th_val
            th_otsu.append(th_val)
        
        true_array.append(y[i])
        pred_array.append(th)
        
    true_array = np.stack(true_array)
    pred_array = np.stack(pred_array)[:,:,:,np.newaxis]
    
    if method == 'simple':
        return true_array, pred_array
    
    elif method == 'otsu':
        
        return true_array, pred_array, th_otsu
    
def get_ndwi(X):
    np.seterr(divide='ignore', invalid='ignore') #ignore nan pixel division
    blue,green,red,nir = cv2.split(X)
    ndwi = (green - nir) / (green + nir)
    return ndwi

def get_rf(X,y,tree=100,depth=7,weight=None):
    rf = RandomForestClassifier(n_estimators=tree, #n_jobs=-1,
                                max_depth=depth, oob_score=True, 
                                class_weight=weight,random_state=0) 
    rf = rf.fit(X, y)
    return rf
    

def unet(filter=18,kernel=3,input_size=(512,512,3),nclass=3):
    '''Build U-Net model'''
    #Contraction Path
    inputs = Input(input_size)
    c1,p1 = conv_block(filter,kernel,inputs,step=1)
    c2,p2 = conv_block(filter,kernel,p1,step=2)
    c3,p3 = conv_block(filter,kernel,p2,step=4)
    c4,p4 = conv_block_drop(filter,kernel,p3,step=8)
    
    #Bottom
    bottom = bottom_block(filter,kernel,p4,step=16)
    
    #Expansion Path
    uc1 = upconv_block(filter,kernel,bottom,c4,step=8)
    uc2 = upconv_block(filter,kernel,uc1,c3,step=4)
    uc3 = upconv_block(filter,kernel,uc2,c2,step=2)
    uc4 = upconv_block(filter,kernel,uc3,c1,step=1)
    
    #Final layer
    if nclass == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    elif nclass > 1:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
        
    uc4 = conv(nclass,kernel,step=2)(uc4)
    conv_out = Conv2D(nclass, 1, activation = activation)(uc4)
    
    #Compile model
    metrics = monitor_unet(nclass) 
    model = Model(inputs = inputs, outputs = conv_out)
    model.compile(optimizer = Adam(learning_rate = 1e-5), 
                      loss = loss, metrics = metrics)
    return model

def conv_block(filter,kernel,input,step=1):
    conv_step = conv(filter,kernel,step)(input)
    conv_step = conv(filter,kernel,step)(conv_step)
    p = MaxPooling2D(pool_size=(2, 2))(conv_step)
    return conv_step, p

def conv_block_drop(filter,kernel,input,step=1):
    conv_step = conv(filter,kernel,step)(input)
    conv_step = conv(filter,kernel,step)(conv_step)
    drop = Dropout(0.5)(conv_step)
    p = MaxPooling2D(pool_size=(2, 2))(drop)
    return drop, p

def upconv_block(filter,kernel,input,concat,step=1):
    upconv = conv(filter,2,step)(UpSampling2D(size=(2,2))(input))
    merge = concatenate([concat,upconv], axis=3)
    upconv = conv(filter,kernel,step)(merge)
    upconv = conv(filter,kernel,step)(upconv)
    return upconv
    
def bottom_block(filter,kernel,input,step):
    b = conv(filter,kernel,step)(input)
    b = conv(filter,kernel,step)(b)
    b = Dropout(0.5)(b) #Ronneberger et al.(2015), pp.239
    return b 

def conv(filter,kernel,step):
    return Conv2D(filter*step, kernel, activation='relu', 
                  kernel_initializer='he_normal', 
                  padding='same')

def monitor_unet(num_class):
    if num_class == 1:
        metrics = [
            TruePositives(name='tp'),
            FalsePositives(name='fp'),
            TrueNegatives(name='tn'),
            FalseNegatives(name='fn'), 
            BinaryAccuracy(name='acc'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR'), 
            BinaryIoU(target_class_ids=[1],threshold=0.20) 
        ]
    elif num_class > 1:
        metrics = ['acc',
            TruePositives(name='tp'),
            FalsePositives(name='fp'),
            TrueNegatives(name='tn'),
            FalseNegatives(name='fn'), 
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR'), 
            OneHotMeanIoU(num_classes=num_class) 
        ]
    return metrics


def cv_load(run,model):
    #For loading models
    tf.keras.backend.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        #model = load_model(f'/work/jflores_umass_edu/hma/log/{run_name}/{modeln}/{modeln}.hdf5')
        model = load_model(f'./log/{run}/{model}.hdf5')
        #model = load_model(f'/work/jflores_umass_edu/hma2/log/#old_3k_runs/{run_name}/{modeln}.hdf5')
    return model