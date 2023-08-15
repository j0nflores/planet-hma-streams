# Planet API image query
# 
# MEHarlan, JAFlores --- updated 04-27-2022

import os
import sys
import json
import time
import backoff
import requests
import numpy as np
import geopandas as gpd
from planet.api.auth import find_api_key
from requests.auth import HTTPBasicAuth
from shapely.geometry import Polygon
 
    
def authenticate():
    try:
        PLANET_API_KEY = find_api_key() #remove find_api_key and place your api key like 'api-key'
    except Exception as e:
        print("Failed to get Planet Key: Try planet init or install Planet Command line tool")
        sys.exit()

    with open('planet_password.txt') as f:
        password = f.readlines()[0]
    payload = json.dumps({
        "email": secret,
        "password": password
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
    
def search_payload(geom,season):

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

    # lookup images acquired within 2020-2021 HMA seasons (Smith & Bookhagen, 2018)
    # fall (September-October-November), winter (December-January-February)
    # spring (March-April-May), summer (June-July-August)    
    

    if season == 'fall':
        get_seas = {"gte": "2020-09-01T00:00:00.000Z",\
                    "lte": "2020-11-30T00:00:00.000Z"}
    elif season == 'winter':
        get_seas = {"gte": "2020-12-01T00:00:00.000Z",\
                    "lte": "2021-02-28T00:00:00.000Z"}
    elif season == 'spring':
        get_seas = {"gte": "2021-03-01T00:00:00.000Z",\
                    "lte": "2021-05-31T00:00:00.000Z"}
    elif season == 'summer':
        get_seas = {"gte": "2021-06-01T00:00:00.000Z",\
                    "lte": "2021-08-31T00:00:00.000Z"}
    else:
        get_seas = {"gte": "2021-01-01T00:00:00.000Z",\
                    "lte": "2021-01-31T00:00:00.000Z"}
        
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": get_seas
    }

    clear_conf_filter = {
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
    }
    
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": 0
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

    asset_filter = {
        "type": "PermissionFilter",
        "config": ["assets.analytic_sr:download"]
    }

    # combine our geo, date, cloud filters
    combined_filter = {
      "type": "AndFilter",
      "config": [geometry_filter,date_range_filter,clear_conf_filter,clear_filter,\
                 vis_filter,cloud_cover_filter,hhaze_filter,lhaze_filter,asset_filter]
    }

    item_type = "PSScene4Band"

    # API request object
    search_request = {
      "item_types": [item_type], 
      "filter": combined_filter
    }
    return search_request

@backoff.on_exception(backoff.expo,requests.exceptions.RequestException,max_tries=8, jitter=None)
def yield_features(url,auth,payload):
    page = requests.post(url, auth=auth, data=json.dumps(payload),headers=headers)
    if response.status_code == 200:
        if page.json()['features']:
            for feature in page.json()['features']:
                yield feature
            while True:
                url = page.json()['_links']['_next']
                page = requests.get(url, auth=auth)

                for feature in page.json()['features']:
                    yield feature

                if page.json()['_links'].get('_next') is None:
                    break

def ft_iterate(geom,season):
    search_json = search_payload(geom,season)
    all_features = list(
        yield_features('https://api.planet.com/data/v1/quick-search',
                       HTTPBasicAuth(PLANET_API_KEY, ''), search_json))    
    for feature in all_features:
        try:
            img_bbox = feature['geometry']['coordinates'][0]
            overlap = Polygon(geom).intersection(Polygon(img_bbox)).area/Polygon(geom).area
            if overlap >= 0.7:
                id_master.append(feature['id'])
                feat.append(feature)   
            else:
                pass   
        except Exception as e:
            print(e)
    
def get_bbox(shp_path):
    '''
    input: shapefile path
    output: geojson-like list of bbox and comids'''
    
    gpd_from_shp = gpd.read_file(shp_path)
    gpd_from_shp = gpd_from_shp.iloc[:5,:]
    
    if gpd_from_shp.crs != "EPSG:4326":
        bounds =  gpd_from_shp.to_crs("EPSG:4326").geometry.bounds
    else:
        bounds =  gpd_from_shp.geometry.bounds

    bbox = []
    comid = []
    for i in range(len(bounds)):
        mm = bounds.loc[i,:]
        bbox_loc = [[mm[0],mm[3]],[mm[2],mm[3]],[mm[2],mm[1]],[mm[0],mm[1]],[mm[0],mm[3]]]
        bbox.append(bbox_loc)
        comid.append(gpd_from_shp.COMID[i])
    return bbox,comid

def print_stats(ids, seas, sum):
    print(f'\tTotal good images: {len(ids)}')
    print(f'\tTotal unique images: {len(list(set(ids)))} \n')
    print(f'Total unique images for {seas} season: {sum} \n')  

def export_reachimgs(merit,season_name,good,bad):
    # Export reach-image info
    os.makedirs(f'./planetAPI/outputs/{merit}/{season_name}',exist_ok=True)
    goodfile = f"./planetAPI/outputs/{merit}/{season_name}/good_geom.npy"
    badfile = f"./planetAPI/outputs/{merit}/{season_name}/bad_geom.npy"
    np.save(goodfile, good)
    np.save(badfile, bad)

def export_seasonids(merit,ids):
    id_seasonfile = f"./planetAPI/outputs/{merit}/ids.npy"
    np.save(id_seasonfile, ids)  
    
    
if __name__ == "__main__":
    
    start_time = time.time()    
    PLANET_API_KEY, payload, headers, response = authenticate()
    
    # Config paths and seasons
    path = r'./data/fromUTM/merit20.shp'
    seasons = ['fall','winter','spring','summer'] 
    merit_fn = os.path.basename(path)[:-4]        
    himat_bboxes = get_bbox(path)
    comid = himat_bboxes[1]
    
    # API query per season 
    id_season = {}    
    for season in seasons:
        id_master = []
        bad_geom = []
        good_geom = {}
        print(f'Processing {merit_fn} for {season} season.....')
        print(f'\tTotal geometries: {len(himat_bboxes[0])}')
        for i, bboxes in enumerate(himat_bboxes[0]):
            try:
                feat = []
                ft_iterate(bboxes,season)
                good_geom[comid[i]] = {v['id']:v for v in feat} 
                print(f'\t\tReach COMID {comid[i]}: good images: {len(feat)}')
            except:
                bad_geom.append(comid[i])
                print(f'\t\tReach COMID {comid[i]}: error')

        #export_reachimgs(merit_fn,season,good_geom,bad_geom)        
        id_season[season] = list(set(id_master))
        sum_season = len(id_season[season])
        print_stats(id_master, season, sum_season) 

    #export_seasonids(merit_fn,id_season)
    print(f'Time elapsed: {(time.time()-start_time)/60:.2f} min')