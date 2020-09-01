from tensorflow.keras import layers
import tensorflow as tf

from layers import EdgeUpdate, NodeUpdate, GlobalUpdate

atom_features = 128
num_messages = 6

def build_embedding_model(preprocessor,
                          dropout=0.0,
                          atom_features=128,
                          num_messages=6,
                          num_heads=8,
                          name='atom_embedding_model'):

    assert atom_features % num_heads == 0, "Wrong feature / head dimension"
    head_features = atom_features // num_heads
    
    # Define keras model
    n_atom = layers.Input(shape=[], dtype=tf.int64, name='n_atom')
    atom_class = layers.Input(shape=[None], dtype=tf.int64, name='atom')
    bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

    input_tensors = [atom_class, bond_class, connectivity, n_atom]

    # Initialize the atom states
    atom_state = layers.Embedding(preprocessor.atom_classes, atom_features,
                                  name='atom_embedding', mask_zero=True)(atom_class)

    # Initialize the bond states
    bond_state = layers.Embedding(preprocessor.bond_classes, atom_features,
                                  name='bond_embedding', mask_zero=True)(bond_class)


    global_state = GlobalUpdate(head_features, num_heads)([atom_state, bond_state, connectivity])

    def message_block(atom_state, bond_state, connectivity, global_state, i):

        bond_state = EdgeUpdate(dropout=dropout)([atom_state, bond_state, connectivity, global_state])
        atom_state = NodeUpdate(dropout=dropout)([atom_state, bond_state, connectivity, global_state])
        
        # Don't do a global update on the final layer; pre-training doesn't use a global output
        if i < num_messages - 1:
            global_state = GlobalUpdate(head_features, num_heads, dropout=dropout)(
                [atom_state, bond_state, connectivity, global_state])

        return atom_state, bond_state, global_state

    for i in range(num_messages):
        atom_state, bond_state, global_state = message_block(
            atom_state, bond_state, connectivity, global_state, i)

    atom_embedding_model = tf.keras.Model(input_tensors, [atom_state, bond_state, global_state], name=name)
    
    return atom_embedding_model
