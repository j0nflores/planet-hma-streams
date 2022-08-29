# Preprocess Planet imagery for labeling/annotations and input to UNet
# Converts raw Planet imagery (tif file) from 16bit to 8bit, apply histogram equalization, 
# and chip/slice the image into a target dimension
#
# @JAFlores, 11.2021 
# 
# (c)Fluvial@UMass-HiMAT Project

import os
import cv2
import time
import glob
import rasterio
import numpy as np
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import box
from rasterio import plot 



def main():
    job_array = int(os.environ['SLURM_ARRAY_TASK_ID'])-1
    print(job_array)
    
    fold = '/nas/cee-water/cjgleason/jonathan/data/planet/20202021'
    batch = glob.glob(fold+'/*')
    batch = [os.path.basename(x) for x in batch]
    batch_fold = batch[job_array]
    
    
    path = f"{fold}/{batch_fold}/PSScene4Band"
    imgs = glob.glob(path+"/*.tif")
    imgs = [x for x in imgs if x[-6:] == 'SR.tif']
    
    main_st = time.time()
    rivlines = "/work/jflores_umass_edu/planetAPI/data/fromUTM/merit20.shp"
    out_fold = "/nas/cee-water/cjgleason/jonathan/data/planet" 
    batch_catch(out_fold,imgs)
    #chip_rivs(out_fold,rivlines)
    print(f'Processing completed. Time elapsed: {(time.time() - main_st)/60:0.2f} min')
    
    
def batch_chips(outdir,img_folder_path):
    path_files = glob.glob(img_folder_path+"/*.tif")
    path_files = [x for x in path_files if x[-6:] == 'SR.tif']
    err_img = []
    for i in range(len(path_files)):
        try:
            class_chips(outdir,path_files[i])
        except:
            err_img.append(i)
    np.save(f'./log/{os.path.basename(os.path.dirname(img_folder_path))}.npy',err_img)
    
    
def batch_catch(outdir,imglist):
    path_files = imglist
    err_img = []
    for i in range(len(path_files)):
        try:
            class_chips(outdir,path_files[i])
        except:
            err_img.append(i)
    #print(f'./log/catchup.npy',err_img)
    
def class_chips(outdir,tif):
    '''
    Arg: tif(string): path to raw planet imagery (e.g. C:\...\file.tif) 
    Output: TIF file of chips stored in chips folder
    '''
    #directory setting, create folder
    start_time = time.time()
    fn = os.path.basename(tif)
    out_8bits = f'{outdir}/{str(fn)}'
    print(f'Processing {fn} .....')

    #Execute 8bit conversion and chipping
    src,scaled = conv_8bits(tif)
    msk = create_mask(src)
    write_tif(out_8bits,scaled,src)
    print(f"\tConverted {str(fn)} in  {round((time.time() - start_time),1)} sec") 
    chip_img(out_8bits,512,use_nan=True)


def conv_8bits(img_path):
    src = rasterio.open(img_path) 
    data = src.read()/10000
    #data = plot.reshape_as_image(data)#,0,2)#/10000
    print(data.shape)
    print(data.dtype)
    scaled = (data  * (255 / data.max())).astype(np.uint16)
    scaled = plot.reshape_as_image(scaled).astype(np.ubyte) 
    return src, data #scaled#data #scaled

def create_mask(src):
    #Create mask array
    msk = src.read(1).astype(float)
    msk[msk == 0] = 'nan' 
    msk[msk > 0] = 255 
    msk = msk.astype(np.ubyte)
    return msk

def equalize_hist(img_array,mask):
    '''
    create a CLAHE object (optional)
    apply contrast limited adaptive histogram equalization (CLAHE) for each img channel
    helps in annotation/labeling but can be skipped for faster processing
    '''
    cla = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)) 
    channels = cv2.split(img_array)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cla.apply(ch))

    #merge equalized channels and apply mask for nan edges
    eq_image = cv2.merge(eq_channels) 
    eq_image = cv2.bitwise_and(eq_image,eq_image,mask = mask)
    eq_image_raster = plot.reshape_as_raster(eq_image) 
    return eq_image_raster  


def write_tif(out_dir_fn,img,src):
    #write tif image
    with rasterio.open(
        out_dir_fn,
        'w',
        driver='GTiff',
        height=src.shape[0],
        width=src.shape[1],
        count=4,
        dtype=img.dtype,
        crs=src.crs,
        transform=src.transform,
    ) as dst:
        dst.write(img)
    dst.close()

def chip_img(img_path,dim,use_nan=True,keep_8bits=False):

    #directory setup, create chips folder
    fn_8bits = os.path.basename(img_path)
    path_chips_tif = f'{os.path.dirname(img_path)}/chips/{str(fn_8bits[:-4])}/'
    os.makedirs(path_chips_tif,exist_ok=True) 

    #load image and get coordinates info
    img = gdal.Open(img_path)
    gt = img.GetGeoTransform() 
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]
    xsize = res * dim
    ysize = res * dim

    #get number of chips available from x and y
    xdiv = round((res * img.RasterXSize)/xsize)
    ydiv = round((res * img.RasterYSize)/ysize)

    #list x and y coordinates of chips
    xsteps = [xmin + xsize * i for i in range(xdiv+1)]
    ysteps = [ymax - ysize * i for i in range(ydiv+1)]

    #create chips from list
    for i in range(xdiv):
        for j in range(ydiv):
            count = j + ydiv*(i)
            chip_bound = (xsteps[i], ysteps[j+1], xsteps[i+1], ysteps[j])
            out_chips = path_chips_tif+str(fn_8bits[:-4])+"_"+str(count)+".tif"
            gdal.Warp(out_chips, img, 
                    outputBounds = chip_bound, dstNodata = 0, srcNodata = 0)
            #remove chip if >95% is nodata
            if use_nan == False:
                chip = rasterio.open(out_chips) 
                chiparr = chip.read((1))
                chip.close()
                if np.count_nonzero(chiparr) < ((dim**2)*.95):
                    os.remove(out_chips)
            else:
                pass
    #close the img data
    img = None
    if keep_8bits == False:
        os.remove(img_path)
    print (f'\tCreated {count} chips for',os.path.basename(img_path))
    
def get_crs(img_path):
    imgs = rasterio.open(img_path)
    proj = imgs.crs.to_epsg()
    return proj

def chip_rivs(fold_chips,path_rivlines):
    
    fold_chips = f'{fold_chips}/chips' 
    #get riverline geometry
    lines = gpd.GeoDataFrame.from_file(path_rivlines) 
    print("Checking intersecting chips within riverlines.....")

    for img in glob.glob(fold_chips+"/*"):
        
        src_path = f'{os.path.dirname(os.path.dirname(img))}/{os.path.basename(img)}.tif'
        geom_riv = lines['geometry'].to_crs(get_crs(src_path))

        for path_chips in glob.glob(img+"/*.tif"):
            rio_chip = rasterio.open(path_chips)
            geom_chip = box(*rio_chip.bounds)

            #check intersection
            intersect = np.array(geom_riv.intersects(geom_chip))
            if any(intersect) == True:
                rio_chip.close()
                rio_chip = None
                #print('keep')
            else: 
                rio_chip.close()
                rio_chip = None
                #print('remove')
                os.remove(path_chips)
    print("Finished filtering chips")


if __name__ == "__main__":

    main()
    


'''
#Construct the argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", 
    help="use for single image processing, path to input tif file")
ap.add_argument("-b", "--folder",
    help="use for batch processing, path to folder of tif file")
ap.add_argument("-d", "--dim", type=int, default=512,
    help="dimension of output chips, returns 512x512 chips (default = 512)")
args = vars(ap.parse_args())

if args["file"]:
    prep_8bit_chips(args["file"],args["dim"])

elif args["folder"]:
    batch_8bit_chips(args["folder"],args["dim"])

else: print('No files or folder is selected')
'''

