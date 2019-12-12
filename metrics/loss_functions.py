#from tensorflow.keras.objectives import *
from tensorflow.keras.metrics import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf


def lse(y_true, y_pred):
    loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
    return loss

def cycle_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.math.abs(y_pred - y_true))
    return loss




# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_sparse_crossentropy_ignoring_last_label_3d(y_true, y_pred):

    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def weighted_crossentropy(y_true,y_pred):

    nb_classes=10

    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1])
#    unpacked = tf.unstack(y_true, axis=-1)
#    y_true = tf.stack(unpacked[:-1], axis=-1)
    weights=1000/(K.sum(y_true,axis=0)+1)
    weights=weights/K.sum(weights)
    cross_entropy = -K.sum(nb_classes*weights*(y_true * log_softmax), axis=1)

    return K.mean(cross_entropy)


def asymmetric_loss(y_true,y_pred):

    alpha=0.3
    beta=0.7

    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    y_pred = tf.nn.softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    ones=K.ones(K.shape(y_true))
    p0=y_pred
    p1=ones-y_pred
    g0=y_true
    g1=ones-y_true

    num=K.sum(p0*g0,axis=0)
    den=num+alpha*K.sum(p0*g1,axis=0)+beta*K.sum(p1*g0,axis=0)
    T=K.sum(num/den)
    Ncl=K.cast(K.shape(y_true[-1]),'float32')
    return Ncl-T

def weighted_dice_loss(y_true,y_pred):

    print (K.int_shape(y_pred))
    smooth=1.
    y_pred=K.reshape(y_pred,(-1,K.int_shape(y_pred)[-1]))
    y_pred=tf.nn.sigmoid(y_pred)
    y_true=K.one_hot(tf.to_int32(K.flatten(y_true)),K.int_shape(y_pred)[-1])
    weights=1000/(K.sum(y_true,axis=0)+1)
    weights=weights/K.sum(weights)
    den = K.sum(y_true * y_pred,axis=0)+smooth
    num = K.sum(y_true,axis=0) + K.sum(y_pred,axis=0)+smooth
    return weights*(1-den/num)

def dice_loss_3d(y_true,y_pred):
    smooth=1.
    y_pred=K.reshape(y_pred,(-1,K.int_shape(y_pred)[-1]))
    softmax=tf.nn.softmax(y_pred)
    y_true=K.one_hot(tf.to_int32(K.flatten(y_true)),K.int_shape(y_pred)[-1])
    intersection = K.sum(y_true * softmax,axis=0)
    return 1-K.mean((2. * intersection + smooth) / (K.sum(y_true,axis=0) + K.sum(softmax,axis=0) + smooth))


# Softmax cross-entropy loss function for coco segmentation
# and models which expect but do not apply sigmoid on each entry
# tensorlow only
def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                                        axis=-1)

"""
The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors.
https://arxiv.org/abs/1708.02002
"""

def focal_loss(y_true, y_pred, gamma=2):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1. - eps)
    return -K.sum(K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred),
                  axis=-1)


def dice(y_true, y_pred, binarise=False, smooth=0.1):
    y_pred = y_pred[..., 0:y_true.shape[-1]]

    # Cast the prediction to binary 0 or 1
    if binarise:
        y_pred = np.round(y_pred)

    # Symbolically compute the intersection
    y_int = y_true * y_pred
    return np.mean((2 * np.sum(y_int, axis=(1, 2, 3)) + smooth)
                   / (np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred, axis=(1, 2, 3)) + smooth))


def dice_coef(y_true, y_pred):
    '''
    DICE Loss.
    :param y_true: a tensor of ground truth data
    :param y_pred: a tensor of predicted data
    '''
    # Symbolically compute the intersection
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3)) + 0.1
    union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3)) + 0.1
    return K.mean(2 * intersection / union, axis=0)


# Technically this is the negative of the Sorensen-Dice index. This is done for minimization purposes
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def make_dice_loss_fnc(restrict_chn=1):
    log.debug('Making DICE loss function for the first %d channels' % restrict_chn)

    def dice_fnc(y_true, y_pred):
        y_pred_new = y_pred[..., 0:restrict_chn] + 0.
        intersection = K.sum(y_true * y_pred_new, axis=(1, 2, 3))
        union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred_new, axis=(1, 2, 3)) + 0.1
        return 1 - K.mean(2 * (intersection + 0.1) / union, axis=0)

    return dice_fnc


def kl(args):
    mean, log_var = args
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    return K.reshape(kl_loss, (-1, 1))


def ypred(y_true, y_pred):
    return y_pred
