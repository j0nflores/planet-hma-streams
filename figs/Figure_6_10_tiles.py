import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pickle
from tensorflow.keras.models import load_model
from master.preprocess import *
from master.models import *

#plot setup
colors = [(0, 0, 0), (1, 0, 1), (0, 1, 1)]  
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
multi = LinearSegmentedColormap.from_list(cmap_name, colors)
binary = ListedColormap([(0, 0, 0), 'cyan'])
cols = 7
cmap = multi

def plot_sub():
    cols, i = 7, 0
    plt.figure(figsize=(20,3))
    plt.subplot(1,cols,1)
    plt.imshow(X_test[i][:,:,[2,1,0]])
    plt.title(i)
    plt.axis('off')
    plt.subplot(1,cols,2)
    plt.imshow(yy[i],cmap=multi)
    plt.axis('off')
    plt.subplot(1,cols,3)
    plt.hist(get_ndwi(X_test[i]).ravel(), bins=256, color='gray')
    plt.axvline(th[i], color='r',linestyle='dashed',label='Th. Otsu')
    plt.axvline(0, color='y',linestyle='dashed',label='Th. 0')
    plt.xlim(-1,1)
    plt.yticks([])
    plt.box(False) 
    plt.legend(frameon=False)
    plt.subplot(1,cols,4)
    plt.imshow(simple_pred[i],cmap=binary)
    plt.axis('off')
    plt.subplot(1,cols,5)
    plt.imshow(otsu_pred[i],cmap=binary)
    plt.axis('off')
    plt.subplot(1,cols,6)
    plt.imshow(rf_pred[i],cmap=binary)
    plt.axis('off')
    plt.subplot(1,cols,7)
    plt.imshow(cv_pred[i],cmap=binary)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'./figs/bin{i}.jpg',dpi=120)

def plot_multi():
    cols, i = 4, 0
    plt.figure(figsize=(20,3))
    plt.subplot(1,cols,1)
    plt.imshow(X_test[i][:,:,[2,1,0]])
    plt.title(i)
    plt.axis('off')
    plt.subplot(1,cols,2)
    plt.imshow(yy[i],cmap=multi)
    plt.axis('off')
    plt.subplot(1,cols,3)
    plt.imshow(rf_pred[i],cmap=multi)
    plt.axis('off')
    plt.subplot(1,cols,4)
    plt.imshow(cv_pred[i],cmap=multi)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'./figs/mul{i}.jpg',dpi=120)
    
#Setup directory 
imgs_path = './data/planet/imgs/'
masks_path = './data/planet/masks/'

#Set configs
k_class = 'multi' #binary or multi
rand = 6

#Preprocess dataset
start_train = time.time()
X_train, X_test, y_train, y_test = prep_data(imgs_path,masks_path,0.3,rand,k_class)

#Thresholding 
simple_test,simple_pred = get_threshold(X_test,y_test, method='simple')
otsu_test, otsu_pred, th = get_threshold(X_test,y_test, method='otsu')
simple_pred.shape

#Random Forest 
with open(f'./model/rf/rf_6.pkl', 'rb') as f:
    rf = pickle.load(f)
rf_X_test, rf_y_test = prep_data_rf(X_test,y_test,k_class)
rf_pred = rf.predict(rf_X_test)
rf_pred = rf_pred.reshape((otsu_pred.shape))

#Computer Vision
cv = load_model('./model/cv_6.hdf5')
cv_pred = cv.predict(X_test)
cv_pred = np.argmax(cv_pred,axis=3)

yy = np.argmax(y_test,axis=3)
print(yy.shape)
print(simple_pred.shape)
print(otsu_pred.shape)
print(rf_pred.shape)
print(cv_pred.shape)
    
plot_sub()
plot_multi()