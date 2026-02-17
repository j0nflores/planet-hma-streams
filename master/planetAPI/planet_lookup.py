# Planet API image query before orders and downloads
# exports Planet image properties by merit reach ids
# serial run

# Updates:
# 04-27-2022 - updated for filters
# 10-16-2022 - update for new PSScene API filters 
# 03-11-2023 - modify for unit reach lookups, api winter filters, clipping bounds

import os
import sys
import json
import time
import glob
import requests
import numpy as np
import geopandas as gpd
from requests.auth import HTTPBasicAuth
from shapely.geometry import Polygon
from config import *
    
def authenticate():
    try:
        PLANET_API_KEY = API_KEY #Planet api key
    except Exception as e:
        print("Failed to get Planet Key: Try planet init or install Planet Command line tool")
        sys.exit()

    payload = json.dumps({
        "email": planet_email, # Planet account email address
        "password": planet_password #Planet account password
    })

    headers = {'Content-Type': 'application/json'}

    response = requests.post(
        "https://api.planet.com/auth/v1/experimental/public/users/authenticate",
        headers=headers,
        data=payload,
    )
    #print(response.status_code)
    if response.status_code == 200:
        bearer_token = f"Bearer {response.json()['token']}"
    else:
        sys.exit(f"Failed with status code {response.status_code}")
    return PLANET_API_KEY,payload,headers,response
    
def search_payload(geom):

    geojson_geometry = {
      "type": "Polygon",
      "coordinates": [geom]
    }

    # get images that overlap with our AOI 
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geojson_geometry
    }

        
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {"gte": "2015-01-01T00:00:00.000Z",
                 "lte": "2025-12-31T00:00:00.000Z"}
    }

    '''clear_conf_filter = {
      "type": "RangeFilter",
      "field_name": "clear_confidence_percent",
      "config": {
        "gte": 95
      }
    }

    clear_filter = {
      "type": "RangeFilter",
      "field_name": "clear_percent",
      "config": {
        "gte": 100
      }
    }

    vis_filter = {
      "type": "RangeFilter",
      "field_name": "visible_percent",
      "config": {
        "gte": 100
      }
    }'''
    
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": 0.1
      }
    }
    
    hhaze_filter = {
      "type": "RangeFilter",
      "field_name": "heavy_haze_percent",
      "config": {
        "lte": 0
      }
    }
    
    lhaze_filter = {
      "type": "RangeFilter",
      "field_name": "light_haze_percent",
      "config": {
        "lte": 0
      }
    }
    
    qual_filter =  {
        "type":"StringInFilter",
        "field_name":"quality_category",
        "config":["standard"]
    }
    
    pub_filter =  {
        "type":"StringInFilter",
        "field_name":"publishing_stage",
        "config":["finalized"]
    }
    
    asset_filter = {
         "type": "AssetFilter",
         "config": [
            "ortho_analytic_4b_sr"
         ]
      }
    
    perm_filter = {
         "type":"PermissionFilter",
         "config": [
            "assets:download"
         ]
      }

    # combine our geo, date, cloud filters
    combined_filter = {
      "type": "AndFilter",
      "config": [geometry_filter,
                 date_range_filter, #clear_conf_filter,clear_filter, vis_filter,
                 cloud_cover_filter,
                 hhaze_filter,
                 lhaze_filter,
                 qual_filter,
                 asset_filter,
                 perm_filter]
    }

    item_type = "PSScene" #"PSScene4Band"

    # API request object
    search_request = {
      "item_types": [item_type], 
      "filter": combined_filter
    }
    return search_request

#@backoff.on_exception(backoff.expo,requests.exceptions.RequestException,max_tries=8, jitter=None)
def yield_features(url,auth,payload):
    page = requests.post(url, auth=auth, data=json.dumps(payload),headers=headers)
    #time.sleep(1)
    if response.status_code == 200:
        if page.json()['features']:
            for feature in page.json()['features']:
                yield feature
                #time.sleep(1)
                
            while True:
                url = page.json()['_links']['_next']
                page = requests.get(url, auth=auth)

                for feature in page.json()['features']:
                    yield feature
                    #time.sleep(1)

                if page.json()['_links'].get('_next') is None:
                    break
    
def ft_iterate(geom):
    search_json = search_payload(geom)
    
    all_features = list(
        yield_features('https://api.planet.com/data/v1/quick-search',
                       HTTPBasicAuth(PLANET_API_KEY, ''), search_json))
    
    #print(f"\tTotal Planet images available: {len(all_features)}")
    
    for feature in all_features:
        try:
            #filters winter images
            if feature['id'][4:6] not in ['12','01','02']: 
                img_bbox = feature['geometry']['coordinates'][0]
                overlap = Polygon(geom).intersection(Polygon(img_bbox)).area/Polygon(geom).area
                
                #get images with at least 10% feature overlap
                if overlap >= 0.1:
                    id_master.append(feature['id'])
                    feat.append(feature) 
                else:
                    pass
            else:
                pass
        except Exception as e:
            print(e)
    
def get_json(shp,fid=None):
    # returns dictionary of merit reach ids and geojson buffered vertices
    '''
    input: shapefile, feature index
    output: geojson-like list of bbox and comids'''

    shp = shp[fid:fid+1]
    shp = shp.to_crs(shp.estimate_utm_crs()) 
    shp = shp.buffer(768) #utm buffer with at leasts 1/2 chip length 768 meters (512x3/2)
    shp = shp.simplify(50, preserve_topology=False) #reduce geometry vertices
    shp = shp.to_crs("EPSG:4326")

    #get geojson-like clipping bounds
    geom = [list(x) for x in shp.geometry.values[0].__geo_interface__['coordinates'][0]]
    bounds = {"type": "Polygon",
                "coordinates": [geom]}
    
    return {'fid':fid,'bounds':bounds}


def export_reachimgs(out_folder,reach_id,good):
    # Export reach-image info from planet api query
    os.makedirs(f'{out_folder}',exist_ok=True)
    goodfile = f"{out_folder}/{reach_id}.npy"
    np.save(goodfile, good)

    
if __name__ == "__main__":
    
    print('Starting...')    
    start_time = time.time()    
    PLANET_API_KEY, payload, headers, response = authenticate()

    #load shapefile - should be with unique features, drop duplicates if any
    shp = gpd.read_file(shp_path) 
    shp = shp.drop_duplicates(subset='geometry')
    n_features = len(shp.geometry.unique())

    #check lookup folder files if existing
    done = [int(os.path.basename(x)[:-4]) for x in glob.glob(lookup_path+"/*.npy")]

    for idx in np.arange(0, n_features):

        #get reach geojson for Planet API
        riv_geom = get_json(shp, fid = idx)
        
        #get imgs infos from planet api to order
        if riv_geom['fid'] not in done:
            #try:
            id_master, feat, good_geom = [],[],{}
            ft_iterate(riv_geom['bounds']['coordinates'][0])
            good_geom[riv_geom['fid']] = {v['id']:v for v in feat} 
            good_geom['bounds'] = riv_geom['bounds']
            print(f'\tFeature ID {riv_geom["fid"]}: good images: {len(feat)}')
            #except:
                #print(f'\tFeature ID {riv_geom["fid"]}: error')
            
            export_reachimgs(lookup_path,riv_geom['fid'],good_geom)        
            
        else:
            print('\tAlready done.')
            
    print(f'Time elapsed: {(time.time()-start_time)/60:.2f} min')