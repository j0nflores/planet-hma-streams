import glob, json
import requests
from requests.auth import HTTPBasicAuth
from config import *

files = glob.glob(order_path + "/*.json")

for file in files:
    with open(file, "r") as f:
        data = json.load(f)

    for url in data:
        r = requests.get(url, auth=HTTPBasicAuth(API_KEY, ""))

        # print status code + a safe peek
        ctype = r.headers.get("Content-Type", "")
        print(f"{file} | {r.status_code} | {ctype} | {url[:80]}...")

        # only try JSON if it actually is JSON
        if "application/json" in ctype.lower():
            j = r.json()
            print("  keys:", list(j.keys())[:10])
            if "state" in j:
                print("  state:", j["state"])
