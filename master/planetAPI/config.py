#PlanetAPI auth
API_KEY = 'planet-api-key' # place your Planet API key
planet_email = 'email' #place your Planet email
planet_password = 'password' #place your Planet password

#run name
run = 'test'

#setup i/o paths - lookup and order url directory
shp_path = './data/shapefile.shp' #input shapefile
lookup_path = f'./output/lookup/{run}' #path of image metadata after lookup/query in API
order_path = f'./output/order/{run}' #path of order urls for downloading
download_path = f'./output/imgs/{run}'  #path to save downloaded images