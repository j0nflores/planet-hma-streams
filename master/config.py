import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

#Training directory
#imgs_path = '/work/jflores_umass_edu/data/planet/3k_new/imgs/'
#masks_path = '/work/jflores_umass_edu/data/planet/3k_new/masks/'
#Test sets
imgs_path = '/work/jflores_umass_edu/hma2/data/test/imgs/'
masks_path = '/work/jflores_umass_edu/hma2/data/test/masks/'


#Main config
rand = 9 #int(os.environ['SLURM_ARRAY_TASK_ID'])  #
method = 'cv'
k_class = 'binary' 
m2b = False
mode = 'train'

#Log
run_name = f'{method}_{k_class[0:3]}' 
modeln =  run_name+f'_{rand}'
os.makedirs(f"./log/{run_name}/",exist_ok=True)
print(modeln)

if method == 'rf':
    tree = 200 #int(os.environ['SLURM_ARRAY_TASK_ID']) #args["trees"]
    depth = 9 # int(os.environ['SLURM_ARRAY_TASK_ID'])  #args["depth"]
    #run_name = f'rf_t{tree}d{depth}'    
    weights = None #{0:0.05, 1:0.25, 2:0.70} #'balanced' #
    print('t ',tree, 'd ', depth, 'r ', rand, 'weights: ',weights)
    
if method == 'cv':
    batch_size = 16
    if k_class == 'binary':
        nclass = 1
        unet_monitor = 'val_binary_io_u'
    if k_class == 'multi':
        nclass = 3
        unet_monitor = 'val_one_hot_mean_io_u'

    mod_callbacks = [CSVLogger(f'./log/{run_name}/train_{modeln}.csv', append=True), \
                     ModelCheckpoint(f'./log/{run_name}/{modeln}.hdf5',verbose=1, \
                                     monitor=unet_monitor, mode='max', save_best_only=True)]