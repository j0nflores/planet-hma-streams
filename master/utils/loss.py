import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, axis=-1):
    smooth = tf.keras.backend.epsilon()
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
    intersection = tf.reduce_sum(y_true * y_pred,axis)
    union = tf.reduce_sum(y_true + y_pred, axis)
    return 1-((2.*intersection + smooth)/union+smooth)

def dice_loss(y_true, y_pred, kclass=3):
    dice = 0
    weights = [0.05,0.25,0.70]
    for i in range(kclass):
        dice += dice_coef(y_true, y_pred) * weights[i]
    return dice

def jaccard_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)



'''
def dice_loss(y_true, y_pred, axis=-1): Working
    smooth = tf.keras.backend.epsilon()
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)
    intersection = tf.reduce_sum(y_true * y_pred,axis)
    dice = (2.*intersection + smooth)/(tf.reduce_sum(y_true,axis) + tf.reduce_sum(y_pred,axis)+smooth)
    return 1-dice'''

'''
def dice_loss(y_true, y_pred, kclass=3):
    dice = 0
    weights = [0.05,0.25,0.70]
    for i in range(kclass):
        dice += (1 - dice_coef(y_true[:,:,i], y_pred[:,:,i]))*weights[i]
    return  dice


def multi_class_dice_loss(y_true, y_pred,logits_size):

    def _compute_dice_loss(flat_input, flat_target):
        alpha=0.2
        smooth=1e-8
        flat_input = ((1.0 - flat_input) ** alpha) * flat_input
        interection = K.sum(flat_input * flat_target, -1)
        loss = 1.0 - ((2.0 * interection + smooth) /(K.sum(flat_input) + K.sum(flat_target) + smooth))
        return loss

    loss=0.0
    for label_idx in range(logits_size):
        flat_input_idx = y_pred[:, label_idx]
        flat_target_idx = y_true[:, label_idx]
        loss_idx = _compute_dice_loss(flat_input_idx, flat_target_idx)

        loss += loss_idx

    return loss'''