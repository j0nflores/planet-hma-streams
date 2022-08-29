import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import *
import tensorflow as tf 

def get_ndwi(X):
    green,red,nir = cv2.split((X*255).astype(np.uint))
    ndwi = (green.astype(float) - nir.astype(float)) / (green.astype(float) + nir.astype(float))
    ndwi = ndwi.astype(np.float32)
    return ndwi
    
def get_threshold(X,y, method='simple'):
    
    true_array, pred_array = [],[]
    for i in range(len(X)):
        green,red,nir = cv2.split((X[i]*255).astype(np.uint))
        np.seterr(divide='ignore', invalid='ignore')
        ndwi = (green.astype(float) - nir.astype(float)) / (green.astype(float) + nir.astype(float))
        ndwi = ndwi.astype(np.float32)
        ndwi[np.isnan(ndwi)] = -1
        gray_image = ndwi * 255/np.max(ndwi)
        
        if method == 'simple':
            th = gray_image > 0 #threshold (Xu, 2006, Huang et al., 2018)
            
        elif method == 'otsu':
            gray_image = gray_image.astype(np.uint8)
            ret, th = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th = np.invert(th)
        
        true_array.append(y[i])
        pred_array.append(th)

    true_array = np.stack(true_array)
    pred_array = np.stack(pred_array)[:,:,:,np.newaxis]

    return true_array, pred_array
    

def get_rf(X,y,tree=100,depth=None):
    rf = RandomForestClassifier(n_estimators=tree, 
                                max_depth=depth, oob_score=True, 
                                class_weight={0:0.05, 1:0.35, 2:0.60},random_state=0)
    rf = rf.fit(X, y)
    return rf
    
def build_unet(filter=18,input_size=(512,512,3), nclass=3):
    inputs = Input(input_size)
    conv1 = Conv2D(filter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(filter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(filter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(filter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(filter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(filter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filter*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(filter*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(filter*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(filter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(filter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(filter*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(filter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(filter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(filter*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(filter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(filter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(filter, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(filter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(filter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(nclass*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    
    if nclass == 2:
        metrics = [
            TruePositives(name='tp'),
            FalsePositives(name='fp'),
            TrueNegatives(name='tn'),
            FalseNegatives(name='fn'), 
            BinaryAccuracy(name='acc'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR'), # precision-recall curve
            BinaryIoU(target_class_ids=[1],threshold=0.20) 
        ]
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        model = Model(inputs = inputs, outputs = conv10)
        model.compile(optimizer = Adam(learning_rate = 1e-5), 
                      loss = 'binary_crossentropy', 
                      metrics = metrics
                     )
    
    elif nclass > 2:
        metrics = ['acc',
            TruePositives(name='tp'),
            FalsePositives(name='fp'),
            TrueNegatives(name='tn'),
            FalseNegatives(name='fn'), 
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR'), # precision-recall curve
            OneHotMeanIoU(num_classes=nclass) 
        ]
        conv10 = Conv2D(nclass, 1, activation='softmax')(conv9)
        model = Model(inputs = inputs, outputs = conv10)
        model.compile(optimizer = Adam(learning_rate = 1e-5), 
                      loss = 'categorical_crossentropy', 
                      sample_weight_mode='temporal',
                      #metrics = ['acc']#metrics
                      weighted_metrics = metrics
                     ) 

    return model

def unet(filter=18,kernel=3,input_size=(512,512,3),nclass=3):
    
    #Contraction Path
    inputs = Input(input_size)
    c1,p1 = conv_block(filter,kernel,inputs,step=1)
    c2,p2 = conv_block(filter,kernel,p1,step=2)
    c3,p3 = conv_block(filter,kernel,p2,step=4)
    c4,p4 = conv_block(filter,kernel,p3,step=8)
    
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