## Mapping proglacial in HMA with Planet imagery


This repo implements semantic segmentation using UNet architecture from Tensorflow/Keras. The objective is to classify proglacial rivers in High Mountain Asia (HMA) using high-resolution Planet imagery. Raw images are preprocessed from 16-bit to 8-bit false color channels (nir,r,g,b) and are sliced into 512x512 chips before placing into the image segmentation/classification methods. Postprocessing of classified chips include merging of tiles to return the same geospatial information from the raw Planet imagery. 

### Prerequisites

Core modules:
* Tensorflow
* OpenCV
* GDAL
* Rasterio

Conda Environment:

Setup the conda environment using the *environment.yml* file. Note: This environment will include installation of core modules above and other modules might need to be updated.

```
conda env create -f environment.yml
```

### Implementation

Raw PlanetScope images should be preprocessed using *chips.py* (inside the utils folder). 

Use the labeled chips generated from annotation tools such as [PixelAnnotationTool](https://github.com/abreheret/PixelAnnotationTool). The scripts calls the training and validation data following the structure below. Note: The *pred* folder is optional output directory for prediction tasks.

```
data
└───planet
│   └───imgs
│   │   │   chip1.tif
│   │   │   chip2.tif
│   │   │   ....
│   └───masks
│       │   chip1.png
│       │   chip2.png
│       │   ....
└───pred
    │   pred1_mask.tif
    │   pred2_mask.tif
    │   ....
```

This repo includes PlanetAPI image lookup, order, and downloads (inside */master/planetAPI/*), PlanetScope raw image preprocessing, and water classifications implementations with:

```
    a) NDWI Thresholding (simple and Otsu)
    b) Random forest
    c) U-Net
```

### Sample Results

![alt text](./figs/sample.jpg "Sample")

