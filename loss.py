import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.losses import LossFunctionWrapper, losses_utils


class AtomInfMask(layers.Layer):
    """ Where there isn't an atom, replace the prediction with -inf so the spin
    is zero after the softmax layer """
    
    def call(self, inputs, mask=None):
        inputs = tf.squeeze(inputs)
        return tf.where(mask, inputs, tf.ones_like(inputs) * inputs.dtype.min)
    

def kl_with_logits(y_true, y_pred):
    """ It's typically more numerically stable *not* to perform the softmax,
    but instead define the loss based on the raw logit predictions. This loss
    function corrects a tensorflow omissions where there isn't a KLD loss that
    accepts raw logits. """

    # Mask nan values in y_true with zeros
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))

    return (
        tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True) -
        tf.keras.losses.categorical_crossentropy(y_true, y_true, from_logits=False))


class KLWithLogits(LossFunctionWrapper):
    """ Keras sometimes wants these loss function wrappers to define how to
    reduce the loss over variable batch sizes """
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='kl_with_logits'):

        super(KLWithLogits, self).__init__(
            kl_with_logits,
            name=name,
            reduction=reduction)

# def masked_categorical_crossentropy_from_logits(y_true, y_pred):
#     """ Mask nan values in y_true with zeros and replace y_pred predictions with -inf """
#     y_true_mask = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
#     y_pred_mask = tf.where(tf.math.is_finite(y_true), y_pred, tf.ones_like(y_pred) * y_pred.dtype.min)
#     loss = tf.keras.losses.categorical_crossentropy(y_true_mask, y_pred_mask, from_logits=True)
#     return tf.reduce_mean(loss)

# class MaskedCategoricalCrossentropy(LossFunctionWrapper):
#     def __init__(self,
#                  reduction=losses_utils.ReductionV2.AUTO,
#                  name='masked_categorical_crossentropy'):
#         super(MaskedCategoricalCrossentropy, self).__init__(
#             masked_categorical_crossentropy_from_logits,
#             name=name,
#             reduction=reduction)
        
# class AtomSoftmax(layers.Layer):
#     def call(self, inputs, mask=None):
#         inputs_exp = tf.exp(tf.squeeze(inputs)) * tf.cast(mask, inputs.dtype)
#         inputs_sum = tf.reduce_sum(inputs_exp, axis=1, keepdims=True)
#         return inputs_exp / inputs_sum
