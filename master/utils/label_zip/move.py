import os
import shutil
import glob

path = "/work/jflores_umass_edu/data/planet/3k_new/"
outpath = "/work/jflores_umass_edu/data/planet/3k_new/all"

os.makedirs(outpath,exist_ok=True)

#folds = glob.glob(path+'/*')
folds = ['#smalls','#nepal','#final_check','#add','#add2']

for i in folds:
    files = glob.glob(path+i+'/*')
    #print(files)
    for j in files:
        try:
            shutil.move(j,outpath)
        except:
            print('error',os.path.basename(j))
print('Done')


path = outpath
img_path = path+"/imgs"
mask_path = path+"/masks"
os.makedirs(img_path,exist_ok=True)
os.makedirs(mask_path,exist_ok=True)


img_files = glob.glob(path+'/*.tif')
for i in img_files:
    try:
        shutil.move(i,img_path)
        print('Moved ',os.path.basename(i))
    except:
        print('error',os.path.basename(j))

mask_files = glob.glob(path+'/*.png')
for j in mask_files:
    try:
        shutil.move(j,mask_path)
        print('Moved ',os.path.basename(j))
    except:
        print('error',os.path.basename(j))

print('Done')