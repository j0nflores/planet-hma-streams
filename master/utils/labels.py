# Preprocess Planet imagery for labeling/annotations and input to UNet
# Organize chips and annotation as UNet training dataset, creates "train" folder
#
# @JAFlores, 11.2021 
# 
# (c)Fluvial@UMass-HiMAT Project

import os
import glob
import json
import math
import numpy as np
from osgeo import gdal
from PIL import Image, ImageDraw
from rasterio.plot import reshape_as_raster, reshape_as_image
import argparse


#Function to generate mask files from JSON files
def json_mask(file_json):
    '''
    Output: 
        PNG file stored to 'mask' folder created under the JSON file folder
    Arg:
        file_json(string): path to json file (e.g. C:/.../file.json) '''

    #config for dimension and path
    dim = (512,512)
    fn = os.path.basename(file_json)
    path_mask = os.path.dirname(os.path.dirname(os.path.dirname(file_json)))+"/train/masks"
    os.makedirs(path_mask,exist_ok=True) 

    #open json file
    with open(file_json, "r",encoding="utf-8") as f:
        label = json.load(f)

    #function to convert annotation points/coordinates into mask array
    #from labelme repo (https://github.com/wkentaro/labelme/blob/main/labelme/utils/shape.py)
    def shape_to_mask(
        img_shape, points, shape_type=None, line_width=10, point_size=5):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        xy = [tuple(point) for point in points]
        if shape_type == "circle":
            assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
            (cx, cy), (px, py) = xy
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
        elif shape_type == "rectangle":
            assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
            draw.rectangle(xy, outline=1, fill=1)
        elif shape_type == "line":
            assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == "linestrip":
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == "point":
            assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
            cx, cy = xy[0]
            r = point_size
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
        else:
            assert len(xy) > 2, "Polygon must have points more than 2"
            draw.polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask
    
    #loop over json polygon/points and generate mask array
    mask_stacked = []
    for i in range(len(label['shapes'])):    
        mask = shape_to_mask(dim, label['shapes'][i]['points'])
        mask_img = mask.astype(int) 
        mask_stacked.append([mask_img])
    
    #stack and merge mask arrays into binary image
    mask_stacked = np.stack(mask_stacked)
    mask_merged = sum(mask_stacked)
    masked = np.reshape(mask_merged, dim).astype(bool)
    
    #Write the masked image file and convert array to boolean 0 to 1
    out_fn = fn[:-5]+".png"
    out_dir = os.path.join(path_mask,out_fn)
    Image.fromarray(masked*255).convert('1').save(out_dir) #fix for speed
    print("Created " +str(out_fn) + " to ../masks folder")


    #Copy img files to 'imgs' folder for label and img matching
    #config for directory
    png_folder_path = os.path.dirname(out_dir)
    fn = os.path.basename(out_dir[:-4])

    #make data img folder directory for file destination
    data_folder_path = os.path.join(os.path.dirname(png_folder_path),"imgs")
    os.makedirs(data_folder_path,exist_ok=True)

    #Load image origin and read as array
    file_tif = os.path.join(os.path.dirname(os.path.dirname(png_folder_path)),'chips',fn.rsplit('_', 1)[0],str(fn)+'.tif')
    img_origin = Image.open(file_tif)
    img_origin_array = np.asarray(img_origin)

    #Copy the image origin file to imgs folder
    img_dir = os.path.join(data_folder_path,fn+".tif")
    Image.fromarray(img_origin_array).save(img_dir)
    print('Created '+str(os.path.basename(file_tif))+" to ../imgs folder")

        
#Function for batch processing
def batch_json_mask(chips_fold):

    path_fold = glob.glob(chips_fold+"/*")

    for json in path_fold:
        path_files = glob.glob(json+"/*.json") 

        #Loop over processing for all files in the list
        for i in range(len(path_files)):
            json_mask(path_files[i]) 

if __name__ == "__main__":
    #Construct the argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", 
        help="use for single file processing, path to input json file")
    ap.add_argument("-b", "--folder",
        help="use for batch processing, path to chips folder with json files inside")
    args = vars(ap.parse_args())

    if args["file"]:
        json_mask(args["file"])

    elif args["folder"]:
        batch_json_mask(args["folder"])

    else: print('No json file or folder is selected')
