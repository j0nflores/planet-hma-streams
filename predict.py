import glob
from tensorflow.keras.models import load_model
from master.preprocess import get_img_arrays
from master.postprocess import *


def main():
    
    #Get slurm job array index
    aindex = int(os.environ['SLURM_ARRAY_TASK_ID'])-1

    #input image path
    fold = '/work/jflores_umass_edu/data/planet/chips'
    batch = sorted(glob.glob(fold+'/*'))
    bindex = {}
    for a,i in enumerate(range(0,len(batch),523)):
        if i == 0:
            bindex[a] = i,i+523
        elif i == 5753:
            bindex[a] = i+1,i+523-1
        else:
            bindex[a] = i+1,i+523
    img_list = batch[bindex[aindex][0]:bindex[aindex][1]]

    #Setup output paths
    outpath = '/nas/cee-water/cjgleason/jonathan/data/hma_out'
    tmp_path = f'/nas/cee-water/cjgleason/jonathan/data/hma_out/tmp_pred_{aindex}'

    model = load_model('/work/jflores_umass_edu/hma/log/hma_multi/hma_multi_10.hdf5')    
    
    err = []
    for path in img_list:

        #load image array and predict
        imgs = get_img_arrays(path+'/')
        
        try:
            #predict
            y_pred = model.predict(imgs)
            
            #Reproject and merge predictions
            proj_pred(path,y_pred,tmp_path,multi=True)
            merge_masks(tmp_path,outpath,path)
        
        except:
            err.append(path)
            print('error on', path)

    np.save(f'./log/err_{aindex}.npy',err,allow_pickle=True)
    print('Prediction done.')


if __name__ == "__main__":
    
    main()