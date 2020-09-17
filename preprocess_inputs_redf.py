import numpy as np
import pandas as pd
import tensorflow as tf
import nfp

from tqdm import tqdm
from rdkit.Chem import MolFromSmiles, AddHs

# Since we're going to re-use the other model's weights, we have
# to be sure that we're also using the same atom and bond classes
from preprocess_inputs import preprocessor
preprocessor.from_json('tfrecords/preprocessor.json')

if __name__ == '__main__':
            
    redf_spin = pd.read_csv('/projects/rlmolecule/pstjohn/atom_spins/redf_spins.csv.gz')
    redf_bv = pd.read_csv('/projects/rlmolecule/pstjohn/atom_spins/redf_buried_volume.csv.gz')
    redf = redf_spin.merge(redf_bv, on=['smiles', 'atom_index', 'atom_type'], how='left')

#     # Load redox properties
#     redf_redox = pd.read_csv('/projects/rlmolecule/pstjohn/atom_spins/redf_redox.csv.gz')
#     redf_redox = redf_redox.set_index('smiles').reindex(redf.smiles.unique())    

    # Get a shuffled list of unique SMILES
    redf_smiles = redf.smiles.unique()
    rng = np.random.default_rng(1)
    rng.shuffle(redf_smiles)

    # split off 5000 each for test and valid sets
    test, valid, train = np.split(redf_smiles, [500, 1000])

    # Save these splits for later
    np.savez_compressed('redf_split.npz', train=train, valid=valid, test=test)

    redf_train = redf[redf.smiles.isin(train)]
    redf_valid = redf[redf.smiles.isin(valid)]

    def inputs_generator(df, train=True):

        for smiles, idf in tqdm(df.groupby('smiles')):
            input_dict = preprocessor.construct_feature_matrices(smiles, train=train)
            spin = idf.set_index('atom_index').sort_index().spin
            fractional_spin = spin.abs() / spin.abs().sum()
            input_dict['spin'] = fractional_spin.values
            input_dict['bur_vol'] = idf.bur_vol.values
#            input_dict['redox'] = redf_redox.loc[smiles].values
            
            assert len(fractional_spin.values) == input_dict['n_atom']

            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()

    serialized_train_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(redf_train, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords_redf/train.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)

    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(redf_valid, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords_redf/valid.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)

#    preprocessor.to_json('tfrecords_redf/preprocessor.json')
