# Randomly pick seasonal image ids and plot dates distributions 
# after Planet API image query
# 
# JAFlores--- updated 04-27-2022

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td

def main():
    # config
    slist = ['fall','spring','summer']
    outdir = './outputs/reach_img.npy'
    plotdir = './outputs/img_dist.jpg'
    season_imgs = {}
    
    for s_index, season in enumerate(slist):
        data = load_data(season)
        reaches = get_reach(data)
        imgs = img_keep(reaches,data,s_index) 
        seas_imgs, unqs = season_stat(imgs, season)
        img_hist(unqs, s_index, season)
        print(season,'.....')
        print(f"\treach w/ imgs: {len(list(seas_imgs[season].keys()))}")
        print("\tunique imgs:", len(unqs))

    # Export plot and reach-image file
    export_plot(plotdir,120)
    np.save(outdir,seas_imgs)
    print('Exported plots and reach-image file to /outputs folder')
    
def load_data(season):                  
    dat = np.load(f'/work/jflores_umass_edu/planetAPI/outputs/merit20/{season}/good_geom.npy',allow_pickle=True).tolist()
    return dat

def get_reach(img_dict):
    return list(img_dict.keys())

def season_img(reach_imgs,season_index):
    tdelta = ['20201015','20210415','20210715'] #timedelta reference 
    if len(reach_imgs) > 1:
        min_d = []
        for imgs in reach_imgs:
            x = abs(dt.strptime(tdelta[season_index],'%Y%m%d').date() \
                    - dt.strptime(imgs[0:8],'%Y%m%d').date())
            min_d.append(x)

            if min(min_d) > td(days=15):
                img = reach_imgs[min_d.index(min(min_d))]

            else:
                window = random.sample([x for x in min_d if x <= td(days=15)],1)[0]
                img = reach_imgs[min_d.index(window)] 
        return img
    else:
        return reach_imgs[0]
                
def img_keep(reach_list, img_dict,season_index):
    keep_dict = {}
    for i in reach_list:
        random.seed(0)  
        try:
            reach_images = list(img_dict[i].keys())
            keep_dict[i] = season_img(reach_images,season_index)
        except:
            pass 
    return keep_dict  

def season_stat(img_dict, season):
    season_imgs[season] = img_dict
    unq_imgs = list(set(list(img_dict.values())))
    return season_imgs, unq_imgs   

def img_hist(unq_imgs, season_index, season):
    bins = [91,92,92] # number of days per season
    df = pd.DataFrame([dt.strptime((x[0:8]),'%Y%m%d').date() for x in unq_imgs ])
    fig = plt.figure(3, figsize=(20,6))
    plt.subplot(1,3,season_index+1)
    plt.hist(df, bins=bins[season_index], color='lightblue', edgecolor='black')
    plt.title(f'{season.capitalize()} (n={len(unq_imgs)})')
        
def export_plot(outpath,dpi):
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    #plt.show()

    
if __name__ == "__main__":
    
    main()

