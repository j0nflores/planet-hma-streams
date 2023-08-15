import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt

def df_scores():
    dict,val,met,cl,mt,seas = {},[],[],[],[],[]
    k_class = 'binary'
    mlist = ['precision', 'recall', 'specificity', 'iou', 'f1']
    coln = ['Precision', 'Recall', 'Specificity','IOU', 'F1-score']
    if k_class == 'binary':
        klist = ['Background','Water','Mean']
    elif k_class == 'multi':
        klist = ['Land/Background','Turbid/Mixed Water','Clear Water','Mean']
    seasons = ['fall','spring','summer']
    runs = ['thresh_simple','thresh_otsu','rf','cv']
    for method in runs:
        if (method == 'rf') or (method == 'cv'):
            fold = f'./log/{method}_season/'
        else:
            fold = f'./log/{method}_season/'
        for s in seasons:
            for mi, m in enumerate(mlist):
                files = sorted(glob.glob(fold+f'/*{s}.csv'))
                for file in files:
                    df = pd.read_csv(file)  
                    for i, row in enumerate(klist):#
                        val.append(df[m][i])
                        met.append(coln[mi])
                        cl.append(row)
                        mt.append(method)
                        seas.append(s)
    dict['Scores'] = val
    dict['Metric'] = met
    dict['Class'] = cl
    dict['Method'] = mt
    dict['Season'] = seas
    return pd.DataFrame(dict)

def fun(x):
    if x == 'fall':
        y = '3  F'+x[1:]
    elif x =='summer':
        y = '2  S'+ x[1:]
    elif x =='spring':
        y = '1  S'+ x[1:]
    return y

#get dataframe
dfx = df_scores()
dfx.Season = dfx.Season.map(fun)
dfx = dfx[(dfx.Class == 'Water')&(dfx.Metric != 'Specificity')].sort_values(by=['Season','Method']

#get summary
sms = dfx[(dfx.Method=='thresh_simple')&(dfx.Class=='Water')].groupby(['Season','Metric']).mean().T
ots = dfx[(dfx.Method=='thresh_otsu')&(dfx.Class=='Water')].groupby(['Season','Metric']).mean().T
rfs = dfx[(dfx.Method=='rf')&(dfx.Class=='Water')].groupby(['Season','Metric']).mean().T
cvs = dfx[(dfx.Method=='cv')&(dfx.Class=='Water')].groupby(['Season','Metric']).mean().T
summary = pd.concat([sms,ots,rfs,cvs])
print(summary)

#plot seasonal performance
sns.set_context("paper",font_scale=1.5)
g = sns.catplot(x='Metric', y="Scores", hue="Season", col="Method", kind="box",col_wrap=2, \
            estimator='mean', errorbar=('ci', 95),palette='tab10' ,aspect =1, linewidth=.3, fliersize=3,
            data=dfx, ascending=[True,False]))
                                                                            