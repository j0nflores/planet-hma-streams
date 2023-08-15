import os
import tensorflow as tf


#Dataset directory
imgs_path = './data/planet/imgs/'
masks_path = './data/planet/masks/'

#Main config
rand = int(os.environ['SLURM_ARRAY_TASK_ID']) #tile selection random seed
method = 'rf' #select model
k_class = 'binary' #binary or multi
m2b = False #convert label from multiclass to binary
mode = 'train' #train or predict

#Log configs
run_name = f'{method}_{k_class[0:3]}2' 
modeln =  run_name+f'_{rand}'
os.makedirs(f"./log/{run_name}/",exist_ok=True)
print(run_name)
print(modeln)

if method == 'rf':
    tree = 200 
    depth = 9 
    weights = None 
    print('t',tree, 'd', depth, 'r', rand, 'w',weights)


if method == 'cv':
    
    #Limit gpu memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:#memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        
    batch_size = 16
    if k_class == 'binary':
        nclass = 1
        unet_monitor = 'val_binary_io_u'
    if k_class == 'multi':
        nclass = 3
        unet_monitor = 'val_one_hot_mean_io_u'

    mod_callbacks = [tf.keras.callbacks.CSVLogger(f'./log/{run_name}/train_{modeln}.csv', append=True), \
                     tf.keras.callbacks.ModelCheckpoint(f'./log/{run_name}/{modeln}.hdf5',verbose=1, \
                                     monitor=unet_monitor, mode='max', save_best_only=True)]