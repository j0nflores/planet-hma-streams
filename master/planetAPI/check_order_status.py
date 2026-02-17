import glob, json
import requests
from requests.auth import HTTPBasicAuth
from confit import *

API_KEY = 'planet-api-key' # place your Planet api key

files = glob.glob(order_path+"/*.json")

for file in files:
    with open(files[0], 'r') as f:
        data = json.load(f)
    for d in data:
        status_result = requests.get(d, auth=HTTPBasicAuth(API_KEY, ''))
        print(f"{file}: {status_result.json()['state']}")
        #data