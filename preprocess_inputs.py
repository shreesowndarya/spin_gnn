import numpy as np
import pandas as pd
import tensorflow as tf
import nfp

from tqdm import tqdm
from rdkit.Chem import MolFromSmiles, AddHs

                
def atom_featurizer(atom):
    """ Return an integer hash representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetNumRadicalElectrons(),
        atom.GetFormalCharge(),
        atom.GetChiralTag(),
        atom.GetIsAromatic(),
        nfp.get_ring_size(atom, max_size=6),
        atom.GetDegree(),
#        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_featurizer(bond, flipped=False):
    
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))
    
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(nfp.get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring]).strip()

preprocessor = nfp.SmilesPreprocessor(
    atom_features=atom_featurizer, bond_features=bond_featurizer, explicit_hs=True)
    

if __name__ == '__main__':
            
    cdf = pd.read_csv('/projects/rlmolecule/pstjohn/atom_spins/cdf_spins.csv.gz')

    # Get a shuffled list of unique SMILES
    cdf_smiles = cdf.smiles.unique()
    rng = np.random.default_rng(1)
    rng.shuffle(cdf_smiles)

    # split off 5000 each for test and valid sets
    test, valid, train = np.split(cdf_smiles, [5000, 10000])

    # Save these splits for later
    np.savez_compressed('split.npz', train=train, valid=valid, test=test)

    cdf_train = cdf[cdf.smiles.isin(train)]
    cdf_valid = cdf[cdf.smiles.isin(valid)]

    def inputs_generator(df, train=True):

        for smiles, idf in tqdm(df.groupby('smiles')):
            input_dict = preprocessor.construct_feature_matrices(smiles, train=train)
            spin = idf.set_index('atom_index').sort_index().spin
            fractional_spin = spin.abs() / spin.abs().sum()
            input_dict['spin'] = fractional_spin.values
            
            assert len(fractional_spin.values) == input_dict['n_atom']

            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()

    serialized_train_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(cdf_train, train=True),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords/train.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)

    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(cdf_valid, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords/valid.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)

    preprocessor.to_json('tfrecords/preprocessor.json')
