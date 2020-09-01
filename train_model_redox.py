import os
import math
import numpy as np
from tensorflow.keras import layers

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import tensorflow_addons as tfa
import nfp

from preprocess_inputs import preprocessor
preprocessor.from_json('tfrecords/preprocessor.json')

from loss import AtomInfMask, KLWithLogits

def parse_example(example):
    parsed = tf.io.parse_single_example(example, features={
        **preprocessor.tfrecord_features,
        **{'spin': tf.io.FixedLenFeature([], dtype=tf.string),
           'bur_vol': tf.io.FixedLenFeature([], dtype=tf.string),
           'redox': tf.io.FixedLenFeature([], dtype=tf.string)}})

    # All of the array preprocessor features are serialized integer arrays
    for key, val in preprocessor.tfrecord_features.items():
        if val.dtype == tf.string:
            parsed[key] = tf.io.parse_tensor(
                parsed[key], out_type=preprocessor.output_types[key])
    
    # Pop out the prediction target from the stored dictionary as a seperate dict
    parsed['spin'] = tf.io.parse_tensor(parsed['spin'], out_type=tf.float64)
    parsed['bur_vol'] = tf.io.parse_tensor(parsed['bur_vol'], out_type=tf.float64)
    parsed['redox'] = tf.io.parse_tensor(parsed['redox'], out_type=tf.float64)
    
    
    spin = parsed.pop('spin')
    bur_vol = parsed.pop('bur_vol')
    redox = parsed.pop('redox')    
    targets = redox
    
    return parsed, targets


max_atoms = 80
max_bonds = 100
batch_size = 128

# Here, we have to add the prediction target padding onto the input padding
# Here, we have to add the prediction target padding onto the input padding
padded_shapes = (preprocessor.padded_shapes(max_atoms=None, max_bonds=None),
                 [None])

padding_values = (preprocessor.padding_values, tf.constant(np.nan, dtype=tf.float64))
#                   {'spin': tf.constant(np.nan, dtype=tf.float64),
#                    'bur_vol': tf.constant(np.nan, dtype=tf.float64)})
                   #'redox': tf.constant(np.nan, dtype=tf.float64)})

num_train = len(np.load('redf_split.npz', allow_pickle=True)['train'])

train_dataset = tf.data.TFRecordDataset('tfrecords_redf/train.tfrecord.gz', compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=num_train).repeat()\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.TFRecordDataset('tfrecords_redf/valid.tfrecord.gz', compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=5000).repeat()\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)


from train_model import model, atom_state, bond_state, connectivity, global_state
from layers import GlobalUpdate

model.load_weights('20200825_combined_losses/best_model.hdf5')
model.trainable = False

global_state = GlobalUpdate(16, 8)([atom_state, bond_state, connectivity, global_state])
redox_pred = layers.Dense(2, name='redox')(global_state)

redox_model = tf.keras.Model(model.inputs, redox_pred)

if __name__ == "__main__":

    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1E-4, 1, 1E-5)
    optimizer = tf.keras.optimizers.Adam(1E-2)
    # weight_decay  = tf.keras.optimizers.schedules.InverseTimeDecay(1E-5, 1, 1E-5)
    #
    # optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    
    redox_model.compile(
        loss=nfp.masked_mean_absolute_error,
        optimizer=optimizer)
    
    redox_model.summary()

    model_name = '20200831_just_redox_xfer_frozen'

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    filepath = model_name + "/best_model.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=0)
    csv_logger = tf.keras.callbacks.CSVLogger(model_name + '/log.csv')

    redox_model.fit(train_dataset,
                    validation_data=valid_dataset,
                    steps_per_epoch=math.ceil(num_train/batch_size),
                    validation_steps=math.ceil(5000/batch_size),
                    epochs=500,
                    callbacks=[checkpoint, csv_logger],
                    verbose=1)
