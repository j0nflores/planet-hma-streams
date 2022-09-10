import sklearn
import pandas as pd
import scipy.stats as st
import numpy as np



def get_scores_df(cms):
    dict = {}
    for i in range(len(cms)):
        dict[i] = metrics(cms[i])
    df = pd.DataFrame(dict)
    df_means = pd.DataFrame(df.T.mean())
    df_means.columns = ['mean']
    df_out = pd.concat([df,df_means],axis=1).T
    return df_out

def cm_score(true,pred,k_class,m2b=False):
    dict = {}
    cv_cm = sklearn.metrics.multilabel_confusion_matrix(true,pred)
    if k_class == 'binary':
        nclass = 2
    elif k_class == 'multi':
        if m2b == False:
            nclass = 3
        else:
            nclass = 2
    for i in range(nclass):
        dict[i] = cv_cm[i].ravel() # array: tn, fp, fn, tp
    return dict


def metrics(cm_array):
    '''Classification metrics (Tharwat, 2018)'''
    dict = {}
    tn, fp, fn, tp = cm_array.ravel()
    precision = tp / (tp+fp)
    recall = tp / (tp + fn)
    specificity = tn / (fp + tn)
    dict['precision'] = precision
    dict['recall'] = recall
    dict['specificity'] = specificity
    dict['iou'] = tp/(tp + fn + fp)
    dict['f1'] = 2*(precision*recall)/(precision + recall)
    dict['geom_mean'] = np.sqrt(recall * specificity)
    dict['balanced_acc'] = (recall + specificity)/2
    return dict

def get_ci(dat_list):
    df = {}
    for j in range(len(dat_list[0])):
        dat = []
        for i in range(len(dat_list)):
            dat.append(dat_list[i][j])
        df[j] = dat
    df = pd.DataFrame(df)

    means, ups, lows = [],[],[]
    for i in range(len(dat_list[0])):
        gfg_data = df[i]
        means.append(np.mean(gfg_data))
        # create 90% confidence interval
        low, up = st.t.interval(alpha=0.90, df=len(gfg_data)-1,
                    loc=np.mean(gfg_data),
                    scale=st.sem(gfg_data))
        lows.append(low)
        ups.append(up)
    ndf = pd.concat([pd.Series(means),pd.Series(lows),pd.Series(ups)],axis=1)
    ndf.columns = ['means','lows','ups']
    return ndf

def get_cv_train(path):
    acc,loss,iou,val_acc,val_loss,val_iou  = [],[],[],[],[],[] 
    for i in range(10):
        #path = f'./data_new/b16lr5e500_{i}_log.csv'
        df= pd.read_csv(path)
        acc.append(df['acc'])
        val_acc.append(df['val_acc'])
        loss.append(df['loss'])
        val_loss.append(df['val_loss'])
        iou.append(df['one_hot_mean_io_u'])
        val_iou.append(df['val_one_hot_mean_io_u'])
    epochs = df.epoch+1
    #######################

def write_stat_csv(out,pdf,run_name):
    all = []
    for i in pdf.columns:
        col, pdf = get_stat_df(i,run_name)
        df_temp = pd.DataFrame(get_stat(col,pdf))
        all.append(df_temp)  
    pd.concat(all).to_csv(out)   
    
def get_stat_df(col_name, run_name):
    dict = {}
    for i in range(10):
        if not i == 1:
            rand = i+1
            modeln =  run_name+f'_{rand}'
            path = f'./log/{run_name}/{modeln}.csv'
            df = pd.read_csv(path).iloc[:,1:]
            dict[rand] = df[col_name]
        else:
            pass
        
    return col_name, pd.DataFrame(dict)

def get_stat(col_name, df):
    stat = {}
    list = [df.iloc[i,:] for i in range(len(df))]
    for i, val in enumerate(list):
        low, up = st.t.interval(alpha=.9, df=len(val)-1,
                    loc=np.mean(val),
                    scale=st.sem(val))
        stat[f'class{i}'] = {f'{col_name}_mean':np.mean(val),'ci':np.mean(val)-low}
    #metrics = pd.concat([pdf,pd.DataFrame(metric)])
    #os.makedirs(os.path.dirname(path),exist_ok=True)
    #metrics.to_csv(path)
    return pd.DataFrame(stat)

def ci(val):
    low,up = st.t.interval(alpha=.95, 
                           df=len(val)-1,
                           loc=np.mean(val),
                           scale=st.sem(val))
    return np.mean(val)-low

#aggregate results
def df_tab(df):
    dict = {}
    cl = df.Class.unique()
    for i,c in enumerate(cl):
        dict[c] = df[df.Class==c].groupby(['Method','Metric']).mean().T
        #dict[c].index = ['c']
        dict[f'{c}_ci'] = df[df.Class==c].groupby(['Method','Metric']).agg(lambda x: ci(x).round(2)).T
        dict[c].index = [f'{i} {c}']
        dict[f'{c}_ci'].index = [f'{i+3} {c}_ci']
        
    df = pd.concat(dict.values())
    df = df[df.columns[::-1]]
    df = df.sort_index().round(2)
    return df

