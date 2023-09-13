import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pickle
from tensorflow.keras.models import load_model
from master.preprocess import *
from master.models import *
from sklearn import metrics 
import json

def plot_sa(ob,pred,ylim,fn):
    plt.figure(figsize=(6,5))
    plt.scatter(ob,pred,alpha=1, color='k')
    plt.grid(False)
    plt.ylabel('Predicted, $\mathregular{km^2}$',fontsize=14)
    plt.xlabel('Observed, $\mathregular{km^2}$',fontsize=14)
    plt.ylim(-.2,ylim)
    plt.xlim(-.2,ylim)
    plt.xticks(np.arange(0, ylim, 0.25))
    plt.yticks(np.arange(0, ylim, 0.25))
    plt.plot([-.2, ylim], [-.2, ylim], color='red',linewidth=0.75)
    plt.savefig(f'./figs/sa_{fn}.jpg',dpi=120)
    
#Setup directory 
imgs_path = './data/planet/imgs/'
masks_path = './data/planet/masks/'

#Set configs
k_class = 'binary'
rand = 6

#Preprocess dataset
start_train = time.time()
X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,k_class)

#Thresholding 
simple_test,simple_pred = get_threshold(X_test,y_test, method='simple')
otsu_test, otsu_pred, th = get_threshold(X_test,y_test, method='otsu')

#Random Forest 
with open(f'./model/rf/rf_6.pkl', 'rb') as f:
    rf = pickle.load(f)
rf_X_test, rf_y_test = prep_data_rf(X_test,y_test,k_class)
rf_pred = rf.predict(rf_X_test)
rf_pred = rf_pred.reshape((otsu_pred.shape))


#Computer Vision
cv = load_model('./model/cv/cv_6.hdf5')
cv_pred = cv.predict(X_test)
cv_pred = np.argmax(cv_pred,axis=3)

#Get surface area
f = 9/1000**2 #3x3m2 to km2
obs = [np.count_nonzero(y_test[i])*f for i in range(len(y_test))]
pred_simple = [np.count_nonzero(simple_pred[i])*f for i in range(len(simple_pred))]
pred_otsu = [np.count_nonzero(otsu_pred[i])*f for i in range(len(otsu_pred))]
pred_rf = [np.count_nonzero(rf_pred[i])*f for i in range(len(rf_pred))]
pred_cv = [np.count_nonzero(cv_pred[i])*f for i in range(len(cv_pred))]

#calculate metrics
sa_met = {}
sa_met['r2_otsu'] = metrics.r2_score(obs,pred_otsu)
sa_met['mae_otsu'] = metrics.mean_absolute_error(obs,pred_otsu)
sa_met['rmse_otsu'] = metrics.mean_squared_error(obs,pred_otsu,squared=False)

sa_met['r2_simple'] = metrics.r2_score(obs,pred_simple)
sa_met['mae_simple'] = metrics.mean_absolute_error(obs,pred_simple)
sa_met['rmse_simple'] = metrics.mean_squared_error(obs,pred_simple,squared=False)

sa_met['r2_rf'] = metrics.r2_score(obs,pred_rf)
sa_met['mae_rf'] = metrics.mean_absolute_error(obs,pred_rf)
sa_met['rmse_rf'] = metrics.mean_squared_error(obs,pred_rf,squared=False)

sa_met['r2_cv'] = metrics.r2_score(obs,pred_cv)
sa_met['mae_cv'] = metrics.mean_absolute_error(obs,pred_cv)
sa_met['rmse_cv'] = metrics.mean_squared_error(obs,pred_cv,squared=False)

with open("./log/sa_metrics.json", "w") as i :
    json.dump(sa_met,i)
    
#plot results (Figure 8)
plot_sa(obs,pred_cv,2.5,'cv')
plot_sa(obs,pred_simple,2.5,'simple')
plot_sa(obs,pred_otsu,2.5,'otsu')
plot_sa(obs,pred_rf,2.5,'rf')