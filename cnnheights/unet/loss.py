#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen


import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

def tversky(y_true, y_pred, weights=None, alpha=0.6, beta=0.4):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss

    Notes
    -----

    - weights is defined if you're running as test (without generator)

    """
    
    if weights is None: 

        y_t = y_true[...,0]
        y_t = y_t[...,np.newaxis]
        # weights
        y_weights = y_true[...,1]
        y_weights = y_weights[...,np.newaxis]

    else: 
        y_t = y_true 
        y_weights = weights
    
    ones = 1 
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t

    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)

def accuracy(y_true, y_pred):
    """compute accuracy"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.equal(K.round(y_t), K.round(y_pred))

def dice_coef(y_true, y_pred, smooth=0.0000001):
    """compute dice coef"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis] # WHAT does this do???

    #intersection = np.logical_and(y_true, y_pred)

    #return 2. * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))    

    intersection = K.sum(K.abs(y_t * y_pred), axis=-1)
    union = K.sum(y_t, axis=-1) + K.sum(y_pred, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=-1)
    
    '''
        # KEPT RETURNING BELOW ESOTERIC ERROR (TENSORFLOWWWWWW):

        return K.mean((2. * intersection + smooth) / (union + smooth), axis=-1)
    File "/home/fjuhsd/miniconda3/envs/cnnheights38/lib/python3.8/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
        raise e.with_traceback(filtered_tb) from None
    File "/home/fjuhsd/miniconda3/envs/cnnheights38/lib/python3.8/site-packages/tensorflow/python/framework/constant_op.py", line 102, in convert_to_eager_tensor
        return ops.EagerTensor(value, ctx.device_name, dtype)
    TypeError: Cannot convert 2.0 to EagerTensor of dtype int64
    '''


def dice_loss(y_true, y_pred):
    """compute dice loss"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return 1 - dice_coef(y_t, y_pred) # y_t, y_pred

def true_positives(y_true, y_pred):
    """compute true positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round(y_t * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((y_t) * (1 - y_pred))

def sensitivity(y_true, y_pred):
    """compute sensitivity (recall)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))

def specificity(y_true, y_pred):
    """compute specificity (precision)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tn = true_negatives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))
