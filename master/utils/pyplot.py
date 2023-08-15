import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import os


def plot_rf_importance(path, rf):
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    importances_g = pd.Series(rf.feature_importances_, index= ['Green','Red', 'NIR'])
    importances_g.plot.bar(yerr=std,color='blue')
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.tight_layout()
    plt.savefig(path,dpi=150)
    
def plot_cnn_fit(fit, run):
    acc = pd.DataFrame(fit.history['acc'])
    val_acc = pd.DataFrame(fit.history['val_acc'])
    loss = pd.DataFrame(fit.history['loss'])
    val_loss = pd.DataFrame(fit.history['val_loss'])
    iou = pd.DataFrame(fit.history['binary_io_u'])
    val_iou = pd.DataFrame(fit.history['val_binary_io_u'])
    epochs = pd.DataFrame(range(1, len(loss) + 1))

    plt.figure(figsize=(10, 8))
    plt.plot(list(epochs[0]), list(loss[0]), 'o', label='Training')
    plt.plot(list(epochs[0]), list(val_loss[0]), label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f'./log/{run}/{run}_loss.png')

    plt.figure(figsize=(10, 8))
    plt.plot(list(epochs[0]), list(acc[0]), 'o', label='Training')
    plt.plot(list(epochs[0]), list(val_acc[0]), label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f'./log/{run}/{run}_acc.png')

    plt.figure(figsize=(10, 8))
    plt.plot(list(epochs[0]), list(iou[0]), 'o', label='Training')
    plt.plot(list(epochs[0]), list(val_iou[0]), label='Validation')
    plt.title('Training and Validation IOU')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean IOU', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f'./log/{run}/{run}_iou.png')
    
    
def plot_cv_train_ci():
    fig, ax1 = plt.subplots(figsize=(10,8),dpi=300)
    plt1 = ax1.plot(epochs, get_ci(loss).means, label='Training Loss Mean',color='k')
    plt2 = ax1.plot(epochs, get_ci(loss).ups, label='Training Loss 90% CI',color='k',linestyle='dashed',alpha=0.7)
    ax1.plot(epochs, get_ci(loss).lows, label='Validation Loss',color='k',linestyle='dashed',alpha=0.7)

    plt3 = ax1.plot(epochs, get_ci(val_loss).means, label='Validation Loss Mean',color='r')
    plt4 = ax1.plot(epochs, get_ci(val_loss).ups, label='Validation Loss 90% CI',color='r',linestyle='dashed',alpha=0.7)
    ax1.plot(epochs, get_ci(val_loss).lows, label='Validation Loss',color='r',linestyle='dashed',alpha=0.7)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    #ax1.grid(color='gray', linestyle='-',axis='y')
    #ax1.legend(fontsize=12, loc="upper left")#, bbox_to_anchor=(0.85,0.2))
    # Twin Axes

    ax2 = ax1.twinx()
    plt5 = ax2.plot(epochs, get_ci(iou).means, label='Training mIOU Mean',color='magenta')
    plt6 = ax2.plot(epochs, get_ci(iou).ups, label='Training  mIOU 90% CI',color='magenta',linestyle='dashed',alpha=0.7)
    ax2.plot(epochs, get_ci(iou).lows, label='Training  mIOU',color='magenta',linestyle='dashed',alpha=0.7)

    plt7 = ax2.plot(epochs, get_ci(val_iou).means, label='Validation mIOU Mean',color='green')
    plt8 = ax2.plot(epochs, get_ci(val_iou).ups, label='Validation mIOU 90% CI',color='green',linestyle='dashed',alpha=0.7)
    ax2.plot(epochs, get_ci(val_iou).lows, label='Validation mIOU',color='green',linestyle='dashed',alpha=0.7)
    ax2.set_ylabel('mIOU', fontsize=12)


    # Display
    #ax2.legend(fontsize=12, loc="center right")#bbox_to_anchor=(0.80,0.75))
    lns = plt1 + plt2 + plt3 +  plt4 + plt5 + plt6 + plt7 + plt8
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc=(.60,.25), fontsize=12)
    plt.savefig(f'plot_multi.jpg')
    plt.show()
    


    