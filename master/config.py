import os
import tensorflow as tf


#Training directory
imgs_path = '/work/jflores_umass_edu/data/planet/3k_new/imgs/'
masks_path = '/work/jflores_umass_edu/data/planet/3k_new/masks/'
#Test sets
#imgs_path = '/work/jflores_umass_edu/hma2/data/test/imgs/'
#masks_path = '/work/jflores_umass_edu/hma2/data/test/masks/'


#Main config
rand = int(os.environ['SLURM_ARRAY_TASK_ID']) 
method = 'rf'
k_class = 'binary' 
m2b = False
mode = 'train'

#Log
run_name = f'{method}_{k_class[0:3]}2' 
modeln =  run_name+f'_{rand}'
os.makedirs(f"./log/{run_name}/",exist_ok=True)
print(run_name)
print(modeln)

if method == 'rf':
    tree = 200 #int(os.environ['SLURM_ARRAY_TASK_ID']) #args["trees"]
    depth = 7 #None # int(os.environ['SLURM_ARRAY_TASK_ID'])  #args["depth"]
    #run_name = f'rf_{k_class[0:3]}_t{tree}d{depth}'    
    weights = None #{0:0.05, 1:0.25, 2:0.70} #'balanced' #
    print('t',tree, 'd', depth, 'r', rand, 'w',weights)


if method == 'cv':
    
    #Limit memory
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