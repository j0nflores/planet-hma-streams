import seaborn as sns
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

#plot setup
colors = [(0, 0, 0), (1, 0, 1), (0, 1, 1)]  
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
multi = LinearSegmentedColormap.from_list(cmap_name, colors)

def df_scores(fold,method,k_class):
    dict,val,met,cl,mt = {},[],[],[],[]
    files = sorted(glob.glob(fold+f'/metrics*.csv'))
    mlist = ['precision', 'recall', 'specificity', 'iou', 'f1'] #'mat'
    coln = ['Precision', 'Recall', 'Specificity','IOU', 'F1-score']
    if k_class == 'binary':
        klist = ['Background','Water','Mean']
    elif k_class == 'multi':
        klist = ['Land/Background','Turbid/Mixed Water','Clear Water','Mean']
    for mi, m in enumerate(mlist):
        for file in files:
            df = pd.read_csv(file)  
            for i, row in enumerate(klist):#
                val.append(df[m][i])
                met.append(coln[mi])
                cl.append(row)
                mt.append(method)
    dict['Scores'] = val
    dict['Metric'] = met
    dict['Class'] = cl
    dict['Method'] = mt
    return pd.DataFrame(dict)

def plot_scores(pdf,hue,cmap='Paired'):

    sns.set_context("paper",font_scale=2)
    g = sns.catplot(x="Metric", y="Scores", hue=hue, kind="bar", 
                    palette=cmap,capsize=0.15,ci=95,errwidth=1, data=pdf,aspect=1.5)
    g.set_ylim=(0, 1)
    
    ax = g.facet_axis(0, 0)
    for p in ax.patches:
        ax.text(p.get_x() - .025 , 
                p.get_height() + 0.05, 
               '{0:.2f}'.format(p.get_height()),   #Used to format it K representation
                color='black', 
                rotation='horizontal', 
                fontsize=8.5)
    plt.legend([],[], frameon=False)
   
    
#BINARY METRICS
simple_mp = './log/thresh_simple'
otsu_mp = './log/thresh_otsu'
rf_mp = './log/rf_mul'
cv_wp = './log/cv_mul'

#Get metrics
met = 'metrics'
simple = df_scores(simple_mp,'NDWI (Thresh=0)','binary')
otsu = df_scores(otsu_mp,'NDWI (Thresh=Otsu)','binary')
rf = df_scores(rf_mp,'Random Forest','binary')
cv_w = df_scores(cv_wp,'Computer Vision','binary')

#Concatenate dataframe
df_m = pd.concat([otsu,simple,rf,cv_w])#
df_m['Scores'] = df_m['Scores'].apply(lambda x: 0.99 if x>0.99 else x)

#Plot binary metrics
plot_scores(df_m[df_m.Class=='Water'],'Method')


#MULTICLASS METRICS
rf_mp = './log/rf_mul/mul'
cv_mp = './log/cv_mul'

#Get metrics
rf_c = df_scores(rf_mp,'Random Forest','multi')
cvt_c = df_scores(cv_mp,'Computer Vision','multi')

rf_binp = './log/rf_bin'
rf_bin = df_scores(rf_binp,'Random Forest','binary')
rf_bin = rf_bin[rf_bin.Class=='Water']
rf_bin.Class = rf_bin.Class.replace('Water', 'Turbid + Clear Water (Binary Model)')

rf_mbinp = './log/rf_mul'
rf_mbin = df_scores(rf_mbinp,'Random Forest','binary')
rf_mbin = rf_mbin[rf_mbin.Class=='Water']
rf_mbin.Class = rf_mbin.Class.replace('Water', 'Turbid + Clear Water (Multiclass Model)')

cv_binp = './log/cv_bin'
cv_bin = df_scores(cv_binp,'Computer Vision','binary')
cv_bin = cv_bin[cv_bin.Class=='Water']
cv_bin.Class = cv_bin.Class.replace('Water', 'Turbid + Clear Water (Binary Model)')

cv_mbinp = '/log/cv_mul/bin'
cv_mbin = df_scores(cv_mbinp,'Computer Vision','binary')
cv_mbin = cv_mbin[cv_mbin.Class=='Water']
cv_mbin.Class = cv_mbin.Class.replace('Water', 'Turbid + Clear Water (Multiclass Model)')

#Concatenate dataframe
df_mt = pd.concat([rf_c,cvt_c])
df_mt = df_mt[df_mt.Class!='Land/Background']
df_new = pd.concat([df_mt,rf_bin,rf_mbin,cv_bin,cv_mbin])
df_new = df_new[df_new.Metric != 'Specificity']

#Plot multiclass metrics
plot_scores(df_new[(df_new.Method=='Random Forest') & (df_new.Class!='Mean')],'Class','Blues')
plot_scores(df_new[(df_new.Method=='Computer Vision') & (df_new.Class!='Mean')],'Class','Blues')