#Get merit reach ids within RGI glaciers

#JAFlores 04-2022 

import numpy as np
import os
import time
import pandas as pd
import geopandas as gpd
from scipy.spatial import Delaunay
from shapely.ops import polygonize, cascaded_union
from shapely.geometry import MultiLineString

#functions to group features by UTM zones and reproject
def get_proj(gdf):
    proj =[]
    for i in range(len(gdf)):
        proj.append(gdf.iloc[i:i+1,:].estimate_utm_crs())
    gdf['UTM'] = proj
    cols = gdf.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    gdf = gdf[cols]
    print('Extracted projections.....')
    return gdf


def group_utm(gdf_proj):
    by_utm = {}
    utm_list = gdf_proj.UTM.unique()
    for utm in utm_list:
        print(f'Grouping and reprojected UTM zone {utm.utm_zone}')
        by_utm[utm.utm_zone] = gdf_proj[gdf_proj.UTM==utm]
        by_utm[utm.utm_zone] = by_utm[utm.utm_zone].to_crs(utm)
        by_utm[utm.utm_zone] = by_utm[utm.utm_zone].drop('UTM',axis=1)
    print('Completed grouping features and reprojected by UTM zones.....')
    return by_utm


def write_utm(outpath, dict):
    for k,v in dict.items():
        outdir = f'{outpath}/{k}.shp'
        v.to_file(outdir)
    print('Completed exporting features.....')   


#function to compute the concave hull of a geoDataFrame of points
def concave_hull(points_gdf, alpha=35):
    
    if len(points_gdf) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return points_gdf.unary_union.convex_hull

    coords = np.array([(x,y) for x,y in zip(points_gdf.geometry.x, points_gdf.geometry.y)])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return gpd.GeoDataFrame({"geometry": [cascaded_union(triangles)]}, index=[0], crs=points_gdf.crs)


start = time.time()
print('Starting.....')

## FILTER RGI DATA ##################################################################
# Read RGI glacier polygons - https://nsidc.org/data/nsidc-0770/versions/6

gdf1 = gpd.read_file(r'..\..\..\#Datasets\HMA\RGI\10_rgi60_NorthAsia.shp')
gdf2 = gpd.read_file(r'..\..\..\#Datasets\HMA\RGI\13_rgi60_CentralAsia.shp')
gdf3 = gpd.read_file(r'..\..\..\#Datasets\HMA\RGI\14_rgi60_SouthAsiaWest.shp')
gdf4 = gpd.read_file(r'..\..\..\#Datasets\HMA\RGI\15_rgi60_SouthAsiaEast.shp')

# Mask file - the HMA basin boundary
msk_path = r'..\..\..\#Datasets\HMA\basins_hma.shp'
msk = gpd.read_file(msk_path)

# Merge shapefiles
rgi = gpd.GeoDataFrame(pd.concat([gdf1, gdf2, gdf3, gdf4]))
rgi.crs = gdf1.crs

# HMA glacier clip and filter RGI glacier area >= 5km2
rgi = rgi.clip(rgi, msk)
rgi_sub = rgi[rgi.Area>=5]
rgi_sub.to_file(r"..\..\..\#Datasets\HMA\RGI\RGI.shp")


## CLIP MERIT REACHES FROM BOUNDARY ###########################################################
# generate boundary from rgi centroids
rgi_points = rgi.copy() 
rgi_points.geometry = rgi_points['geometry'].centroid # convert geometry to centroid 
rgi_bound = concave_hull(rgi_points, alpha=1)
rgi_bound = rgi_bound.geometry.buffer(.5) #extent boundary
rgi_bound.to_file(r'..\..\..\#Datasets\HMA\RGI\RGI_bound.shp')


# Read MERIT-Basin river network vector - https://www.reachhydro.org/home/params/merit-basins
merit_path = r'..\..\..\#Datasets\HMA\MERIT\riv_pfaf_4_MERIT_Hydro_v07_Basins_v01.shp'
merit = gpd.read_file(merit_path)
merit_rgi = gpd.clip(merit, rgi_bound)
merit_rgi.to_file(r"..\..\..\#Datasets\HMA\MERIT\MERIT_RGI.shp")


## REPROJECT RGI AND MERIT DATA TO MEASURE BUFFER DISTANCE ##############################
for shape in ['RGI','MERIT_RGI']:
    #shapefile paths to reproject
    inpath = f'../../../#Datasets/HMA/{shape}.shp'  
    out = f'../../../#Datasets/HMA/UTM/{shape}'   
    os.makedirs(out,exist_ok=True)        

    #Get reprojected UTMs
    shp_proj = get_proj(gpd.read_file(inpath))
    utm_df = group_utm(shp_proj)
    #write_utm(f'{out}/{os.path.basename(inpath)[:-4]}',utm_df)

    
## BUFFER RGI AND LOCATE MERIT REACHES ########################################################
#get merit20.shp - 20 km RGI buffer in UTM projection

# load reprojected shapefiles
'''rgi42 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\RGI5\42N.shp")
rgi43 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\RGI5\43N.shp")
rgi44 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\RGI5\44N.shp")
rgi45 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\RGI5\45N.shp")
rgi46 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\RGI5\46N.shp")
rgi47 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\RGI5\47N.shp")
rgi48 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\RGI5\48N.shp")
merit42 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\MERIT_RGI\42N.shp")
merit43 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\MERIT_RGI\43N.shp")
merit44 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\MERIT_RGI\44N.shp")
merit45 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\MERIT_RGI\45N.shp")
merit46 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\MERIT_RGI\46N.shp")
merit47 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\MERIT_RGI\47N.shp")
merit48 = gpd.read_file(r"..\..\..\#Datasets\HMA\UTM\MERIT_RGI\48N.shp")'''

# create list of dataframes and UTM zones
rlist = [rgi42,rgi43,rgi44,rgi45,rgi46,rgi47,rgi48]
mlist = [merit42,merit43,merit44,merit45,merit46,merit47,merit48]
zones = np.arange(42,48).tolist()

# keep dictionary of dataframes
dict = {}
for i,j in enumerate(zones):
    dict[f'n{j}'] = {
    'merit': mlist[i],
    'rgi': rlist[i],
    'rbuff20': rlist[i].geometry.buffer(20000)} #UTM projected 20km buffer distance
    dict[f'n{j}']['mclip20'] = gpd.clip(mlist[i], dict[f'n{j}']['rbuff20'])

#List COMIDs 
comid20 = []
for i in list(dict.keys()):
    comid20.append(dict[i]['mclip20'].COMID)
clist20 = [x for y in comid20 for x in y]

#Export MERIT data
merit20 = merit[merit.COMID.isin(clist20)]
outdir = r"..\..\..\#Datasets\HMA\RGI\UTM\RGI5"
merit20.to_file(f'{outdir}/merit20.shp')

print(f'Time elapsed: {(time.time()-start)/60:.2f} min')