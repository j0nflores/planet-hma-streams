#Inspects all the seasonal PlanetScope ids for each MERIT-Basin reach ids

import os
import glob
import json

#The .npy data includes all downloadable PlanetScope imagery 
#with its corresponding image information upon API lookup using river reach geometries
#the data is PlanetLabs proprietary
def load_data(season):                  
    dat = np.load(f'./data/planet/lookup/{season}/good_geom.npy',allow_pickle=True).tolist()
    return dat

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

#Actual image list downloaded and successfully preprocessed 
im_list = [os.path.basename(x)[:-17] for x in glob.glob('./data/planet/chips/*')]

#create dataframe for image list for season filtering
dates = pd.Series(im_list).map(lambda x: x[0:8])
dates = pd.to_datetime(dates, format='%Y%m%d')  
df = pd.concat([pd.Series(im_list),dates],axis=1)
df.columns = ['img_id','date']
df.index = df.img_id

#populate PlanetScope ids by season appending the MERIT reach ids cpvered
season = 'summer'
reach_dict = load_data(season)

keep = {}
if season == 'fall':
    df_s = df[df.date<='2020-11-30']
if season == 'spring':
    df_s = df[(df.date>'2020-11-30')&(df.date<='2021-05-30')]
if season == 'summer':
    df_s = df[df.date>'2021-05-30']

for look in df_s.img_id:
    temp = []
    for i in reach_dict.keys():
        if look in list(reach_dict[i].keys()):
            temp.append(i)
    keep[look] = temp

with open(f"./data/ids/planet_{season}.json", "w") as f:
    json.dump(keep,f, cls=NpEncoder)