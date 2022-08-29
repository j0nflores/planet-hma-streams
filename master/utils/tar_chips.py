#Pack image chips for labeling

import os
import glob
import random
import tarfile
import shutil

        
def main():
    fold = '/work/jflores_umass_edu/data/planet/chips'
    label_path = '/work/jflores_umass_edu/data/planet/label'
    img = img_fold(fold)
    selection = select_chip(img)
    copy_chip(label_path,selection)
    write_tar(os.path.dirname(label_path),label_path)
    
    
def img_fold(chip_fold):
    batch = glob.glob(chip_fold+'/*')
    #batch = [os.path.basename(x) for x in batch]
    return batch

def select_chip(img_fold):
    select = []
    for fold in img_fold:
        chips = glob.glob(fold+"/*.tif")
        if len(chips) > 0:
            select_chip = random_chip(chips)
            select.append(select_chip)
        else:
            pass
    select = [x for y in select for x in y]
    return select

def random_chip(chip_list):
    
        random.seed(1)
        return random.sample(chip_list, 1)
    
def copy_chip(outdir,path_list,verbose=False):
    for path in path_list:
        shutil.copy(path, outdir)
        if verbose == True:
            print(f'\tMoving {os.path.basename(path)} to \
            {os.path.basename(outdir)}')
    
def write_tar(outdir,folder,file_format):
    tar = tarfile.open(f"{outdir}/file.tar.gz", "w:gz")
    chips = glob.glob(folder+f"/*.{file_format}")
    for file in chips:
        tar.add(file)
    tar.close()
        

if __name__ == "__main__":
    
    main()