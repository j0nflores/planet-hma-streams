# Order and download planet images based on API image query results
# 
# Updates:
# 04-30-2022 - batch downloading, functionized 
# 11-13-2022 - updated for PSScene4band asset deprecation 
# 03-12-2023 - modified for parallel run per unit feature

import os
import glob
import time
import requests
import json
import sys
import pathlib
import numpy as np
import argparse
from shapely.geometry import shape, Polygon, mapping
from config import *

def authenticate_order():
    try:
        PLANET_API_KEY = API_KEY #Planet api key
        
    except Exception:
        print("Failed to get Planet Key: Try planet init or install Planet Command line tool")
        sys.exit(1)

    headers = {'Content-Type': 'application/json'}

    # check if API key is valid 
    response = requests.get('https://api.planet.com/compute/ops/orders/v2',auth=(PLANET_API_KEY, ""))
    if response.status_code==200:
        print('Setup OK: API key valid')
        #print(PLANET_API_KEY)
    else:
        print(f'Failed with response code {response.status_code}: reinitialize using planet init')
    
    return PLANET_API_KEY, headers, response

def order_now(order_payload):
    orders_url = 'https://api.planet.com/compute/ops/orders/v2'
    response = requests.post(orders_url, data=json.dumps(order_payload), auth=(PLANET_API_KEY, ""), headers=headers)
    if response.status_code==202:
        order_id =response.json()['id']
        url = f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
        feature_check = requests.get(url, auth=(PLANET_API_KEY, ""))
        if feature_check.status_code==200:
            print(f"Order URL: https://api.planet.com/compute/ops/orders/v2/{order_id}")
            return f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
    else:
        print(f'Failed with Exception code : {response.status_code}')


def download_results(order_url,folder, overwrite=False):
    r = requests.get(order_url, auth=(PLANET_API_KEY, ""))
    if r.status_code ==200:
        response = r.json()
        results = response['_links']['results']
        results_urls = [r['location'] for r in results]
        results_names = [r['name'] for r in results]
        print('{} items to download'.format(len(results_urls)))

        for url, name in zip(results_urls, results_names):
            path = pathlib.Path(os.path.join(folder,name))

            if overwrite or not path.exists():
                print('downloading {}'.format(name))
                r = requests.get(url, allow_redirects=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                open(path, 'wb').write(r.content)
            else:
                print('{} already exists, skipping {}'.format(path, name))
    else:
        print(f'Failed with response {r.status_code}')
    
def order_url(feature_id,ids,json_bound,batchid=0):
    payload = {
    "name": f'{feature_id}_{batchid}', # order name (any)
    "order_type":"partial", #allows for an order to complete even if few items fail
        "notifications":{
        "email": False
    },
    "products":[{  
            "item_ids": ids,
            "item_type":"PSScene",
            "product_bundle":"analytic_sr_udm2"
        }],
    "tools": [{
            "clip": {
                "aoi": json_bound
            }
        }]
    }
    return order_now(payload)
    
    
def riv_lookup(lookup_folder, feature):
    riv = f'{lookup_folder}/{feature}.npy'
    file = np.load(riv,allow_pickle=True).tolist()
    imlist = []
    for key in file.keys():
        if key != 'bounds':
            imlist.append(list(file[key].keys()))
    imlist = [x for y in imlist for x in y]

    riv_bounds = file['bounds']
    vertices_count = len(riv_bounds['coordinates'][0])
    if vertices_count > 500:
        riv_bounds = simplify_bounds(riv_bounds)
    return {'imids': sorted(imlist) , 'bounds': riv_bounds}


def simplify_bounds(polygon_dict, target_vertices=450, preserve_topology=True):
    """
    Simplify a polygon to approximately the target number of vertices.
    
    Planet OrderAPI requires <500 vertices
    https://docs.planet.com/platform/integrations/qgis/planet-qgis-plugin/#search-for-imagery
    """

    poly = shape(polygon_dict)
    if not isinstance(poly, Polygon):
        raise ValueError("Input geometry must be a single Polygon.")

    def vertex_count(p):
        return max(0, len(p.exterior.coords) - 1)

    orig_count = vertex_count(poly)
    if target_vertices >= orig_count:
        return polygon_dict, 0.0

    low, high = 0.0, 1e-6
    while vertex_count(poly.simplify(high, preserve_topology=preserve_topology)) > target_vertices and high < 1.0:
        high *= 2

    for _ in range(50):
        mid = (low + high) / 2
        simplified = poly.simplify(mid, preserve_topology=preserve_topology)
        count = vertex_count(simplified)

        if count > target_vertices:
            low = mid
        else:
            high = mid

    simplified = poly.simplify(high, preserve_topology=preserve_topology)
    return mapping(simplified)

def imgs_downloaded(im_path,comid):
    fold = f'{im_path}/{comid}'
    batch = glob.glob(fold+'/*')
    batch = [os.path.basename(x) for x in batch]
    all = []
    for dl_fold in batch:
        path = f"{fold}/{dl_fold}/PSScene"
        imgs = glob.glob(path+"/*.tif")
        imgs = [x for x in imgs if x[-11:] == 'SR_clip.tif']
        all.append(imgs)
    return [os.path.basename(x)[:-26] for y in all for x in y]
    

    
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--order",action='store_true', 
        help="order planet images")
    ap.add_argument("-d", "--download", action='store_true',
        help="download planet images, please check order status if 'success' ")
    args = vars(ap.parse_args())
    
    # Authenticate API
    PLANET_API_KEY, headers, response = authenticate_order()
    
    if args["order"]:    

        #start order, serial for api
        start = time.time()

        #check output order url directory if existing
        os.makedirs(order_path,exist_ok=True)
        urls = glob.glob(order_path+"/*.json")
        order_done = [int(os.path.basename(x)[:-5]) for x in urls ]

        #list feature id
        feature_ids = [int(os.path.basename(x)[:-4]) for x in glob.glob(f'{lookup_path}/*.npy')]
                                                                  
        #list features not ordered yet 
        left = list(set(feature_ids)-set(order_done))
        print('Not done: ',len(left))

        #run for reach ids 
        for feature in left:

            #double checks if done ordering
            if feature not in order_done:
                
                try:

                    #gather feature api lookup info
                    riv_info = riv_lookup(lookup_path,feature)
                    bounds = riv_info['bounds']
                    ids = riv_info['imids']#[0:1]
                    
                    #check list of imgs already downloaded for reach id
                    imgs_path = f'{download_path}'
                    imgs_done = imgs_downloaded(imgs_path, feature)

                    #imgs to download
                    ids = sorted(list(set(ids)-set(imgs_done)))
                    print(f'\nOrdering images: feature {feature}, imgs: {len(ids)}')

                    #batch per Planet order suggestion (450img/batch)
                    if len(ids) > 450:

                        order_urls = []
                        nbatch = int(np.ceil(len(ids)/450))

                        for batch in range(nbatch):

                            ids_order = ids[batch*450:450*(batch+1)]

                            #initiate order
                            print(f'\nBatch {batch}, imgs: {len(ids_order)}')
                            order_urls.append(order_url(feature,ids_order,bounds,batch))

                    #no batching needed
                    elif (len(ids) < 450) & (len(ids) >= 1):
                        print(f'\tOrder imgs: {len(ids)}')
                        order_urls = [order_url(feature,ids,bounds)]

                    elif len(ids) == 0:
                        print('\tno image to order')

                    #export order url list
                    with open(f'{order_path}/{feature}.json','w') as f:
                        json.dump(order_urls,f)
                        order_urls = None
                    print(f'Orders complete. Time elapsed: {(time.time()-start)/60:.2f} min\n')
                
                except: 
                   print(f'\t bad lookup/order: {feature}\n')
                    
            else: 
                print('Aldready done.')


    elif args["download"]:

        # Start download
        start = time.time()
        
        # Slurm array run for parallel downloading, 4-5 only due to Planet rate limiting
        slurm = int(os.environ['SLURM_ARRAY_TASK_ID']) 
    
        #check order folder and get feature ids to run
        feature_ids = [int(os.path.basename(x)[:-5]) for x in sorted(glob.glob(order_path+"/*.json"))]
        feature = feature_ids[slurm]
        
        #setup output folder
        print(f'\nDownloading images: riv {feature}...')
        download_path = f'{download_path}/{feature}'
        os.makedirs(download_path,exist_ok=True)

        #get order urls from input path
        urlp = f'{order_path}/{feature}.json'
        with open(urlp,'r') as f:
            urls =  json.load(f)
            url_dc = {}
            for url in urls:
                url_dc[url[45:]] = url

        #check already done urls in output path
        dl_done = [os.path.basename(x) for x in glob.glob(download_path+"/*")]

        #gather urls not yet downloaded
        left = list(set(list(url_dc.keys()))-set(dl_done))

        if len(left)>0:
            for i,link in enumerate(left):
                download_results(url_dc[link],download_path)
                print(f'\tDownloaded batch {i}. Time elapsed: {(time.time()-start)/60:.2f} min\n')

        else:
            print('\tAlready done.')
