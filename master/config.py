import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


#Main config
rand = int(os.environ['SLURM_ARRAY_TASK_ID']) 
run_name = f'cv_bin' 
k_class = 'binary' 
m2b = False

#Defaults
batch_size = 16
modeln =  run_name+f'_{rand}'
os.makedirs(f"./log/{run_name}/",exist_ok=True)
print(modeln)

#Training directory
imgs_path = '/work/jflores_umass_edu/data/planet/3k_new/imgs/'
masks_path = '/work/jflores_umass_edu/data/planet/3k_new/masks/'
#Test sets
#imgs_path = '/work/jflores_umass_edu/hma2/data/test/imgs/'
#masks_path = '/work/jflores_umass_edu/hma2/data/test/masks/'

#CV log
if k_class == 'binary':
    nclass = 1
    unet_monitor = 'val_binary_io_u'
if k_class == 'multi':
    nclass = 3
    unet_monitor = 'val_one_hot_mean_io_u'
    
mod_callbacks = [CSVLogger(f'./log/{run_name}/train_{modeln}.csv', append=True), \
                 ModelCheckpoint(f'./log/{run_name}/{modeln}.hdf5',verbose=1, \
                                 monitor=unet_monitor, mode='max', save_best_only=True)]


