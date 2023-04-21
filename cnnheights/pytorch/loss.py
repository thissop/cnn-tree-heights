import torch
import torch.nn as nn

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def accuracy(): 
    pass 

def torch_tversky_loss(y_true, y_pred, weights, alpha=0.6, beta=0.4):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param y_pred: the logits
    :param y_true: the segmentation ground_truth
    :param weights: the weights of the boundaries 
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :return: the loss
    """
    
    ones = 1 
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    tp = torch.sum(weights * p0 * g0) # check that it returns same result as tf.reduce_sum
    fp = alpha * torch.sum(weights * p0 * g1)
    fn = beta * torch.sum(weights * p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    tversky_loss = 1.0 - torch.mean(score) # tf.reduce_mean()

    return tversky_loss

def calc_loss(y_true, y_pred, weights, metrics, alpha:float=0.6, beta:float=0.4):
    import sklearn

    tversky_loss = torch_tversky_loss(y_true=y_true, y_pred=y_pred, weights=weights, alpha=alpha, beta=beta)

    pred = torch.sigmoid(y_pred) 
    dice = dice_loss(pred, y_true) 

    y_pred = pred.detach().numpy().squeeze()
    y_pred[y_pred<0.5] = 0
    y_pred[y_pred>=0.5] = 1

    accuracy = sklearn.metrics.accuracy_score(y_true.detach().numpy().squeeze(), y_pred) 

    metrics['dice_loss'] = dice.data.cpu().numpy() * y_true.size(0) 
    metrics['tversky_loss'] = tversky_loss.data.cpu().numpy() * y_true.size(0) 
    metrics['accuracy'] = accuracy 

    return tversky_loss, metrics 