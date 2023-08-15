# Order and download planet images based on API image query results
# 
# MEHarlan, JAFlores --- updated 04-30-2022

import os
import time
import requests
import json
import sys
import pathlib
from planet import api 
from requests.auth import HTTPBasicAuth
import numpy as np
import argparse

def authenticate_order():
    try:
        PLANET_API_KEY = api.auth.find_api_key() 
    except Exception as e:
        print("Failed to get Planet Key: Try planet init or install Planet Command line tool")
        sys.exit()

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
                print('downloading {} to {}'.format(name, path))
                r = requests.get(url, allow_redirects=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                open(path, 'wb').write(r.content)
            else:
                print('{} already exists, skipping {}'.format(path, name))
    else:
        print(f'Failed with response {r.status_code}')
    
def get_all_ids(imglist):
    seasons = ['fall','spring','summer']
    all_unq_img = []
    for seas in seasons:
        unq_img = list(set(list(imglist[seas].values())))
        all_unq_img.append(unq_img)
    all_unq_img = [x for y in all_unq_img for x in y]
    return all_unq_img

def batch_ids(id_list,batch_size=450):
    batch = []
    for i in range(round(len(id_list)/batch_size)):
        if i == 0:
            batch.append(id_list[i:batch_size])
        else:
            batch.append(id_list[batch_size*i:batch_size*i+batch_size])
    return batch

def order_batch(batch_ids):
    print(f'Ordering images .......')
    order_urls = []
    for i, idlist in enumerate(batch_ids):
        payload = {
        "name": f'batch_{i}', # change order name to whatever you would like (name is not unique)
        "order_type":"partial", #allows for an order to complete even if few items fail
            "notifications":{
            "email": False
        },
        "products":[  
            {  
                "item_ids": idlist,
                "item_type":"PSScene4Band",
                "product_bundle":"analytic_sr"
            }
            ],
        }
        order_urls.append(order_now(payload))
    
    out = "./planetAPI/outputs/order_urls"
    os.makedirs(out,exist_ok=True)
    np.save(f'{out}/urls.npy',order_urls)
    return order_urls
    

def download_batch(outdir, url_list):
    print('Downloading images.......')
    for i, url in enumerate(url_list):
        if url != None:
            download_results(url,folder=outdir) 
        print(f'\tCompleted batch {i}.......')
    

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--order",action='store_true', 
        help="order planet images")
    ap.add_argument("-d", "--download", action='store_true',
        help="download planet images, please check order status if 'success' ")
    args = vars(ap.parse_args())

    if args["order"]:
        
        # Authenticate API
        PLANET_API_KEY, headers, response = authenticate_order()
        
        # Start order
        start = time.time()
        ids_path = "./planetAPI/outputs/reach_img.npy"
        imglist =  np.load(ids_path, allow_pickle = True).tolist()
        all_ids = get_all_ids(imglist)
        all_ids = all_ids[2:] #already ordered 0,1 as test
        batch_list = batch_ids(all_ids)
        batch_urls = order_batch(batch_list)
        print(f'Order complete. Time elapsed: {(time.time()-start)/60:.2f} min')

    if args["download"]:
        
        # Authenticate API
        PLANET_API_KEY, headers, response = authenticate_order()
        
        # Start download
        start = time.time()
        urls_path = './planetAPI/outputs/order_urls/urls.npy'
        outpath = r'./data/planet/20202021/'
        batch_urls =  np.load(urls_path, allow_pickle = True).tolist()
        download_batch(outpath,batch_urls)
        print(f'Download complete. Time elapsed: {(time.time()-start)/60:.2f} min')

    else:
        print('Select action: "--order" or "--download"')

