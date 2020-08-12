# spin_gnn

Code to train a neural network to predict fractional atom spin. The order of operations here is

1. `parse_spins.py`: queries the databases and dumps the spins to csv files. I wasn't in a rush, so this script is slow. This took the better part of a weekend to run.
2. `preprocess_inputs.py`: Splits the data into training and testing datasets, and converts the SMILES inputs into tensorflow-encoded graph arrays. These inputs are stored in a `tfrecords` fileformat, an unfriendly but efficient binary format for storing training data. See the tensorflow [docs](https://www.tensorflow.org/tutorials/load_data/tfrecord) if you're curious.
3. `train_model.py`: Actually defines and runs the GNN model. There's a couple oddities to this specific example, in that the atom spins should sum to 1, and there's a variable number of atoms per example. So I have to write a couple custom keras objects in `loss.py`.
4. `gpu_submit.sh`: submits the `train_model.py` calculation to run on an Eagle GPU node.
