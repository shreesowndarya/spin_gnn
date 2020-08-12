import numpy as np
import pandas as pd
import psycopg2
import cclib
import rdkit.Chem
from tqdm import tqdm

dbparams = {
    'dbname': 'bde',
    'port': 5432,
    'host': 'yuma.hpc.nrel.gov',
    'user': 'bdeops',
    'password': '*********',  # deleting the password from the repo
    'options': f'-c search_path=bde',
}

with psycopg2.connect(**dbparams) as conn:

    redf = pd.read_sql_query("""
    SELECT * from redoxcompound where estate='radical' and status='finished'
    """, conn)    

def get_spins(row):
    
    mol = rdkit.Chem.MolFromSmiles(row.smiles)
    molH = rdkit.Chem.AddHs(mol)
    try:
        data = cclib.io.ccread(row.logfile)
        spins = data.atomspins['mulliken']
        for atom, spin in zip(molH.GetAtoms(), data.atomspins['mulliken']):
            yield row.smiles, atom.GetSymbol(), atom.GetIdx(), spin
            
    except AttributeError as ex:
        print(f'Problem with {row.smiles}', flush=True)
        

def df_generator(df):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        yield from get_spins(row)
        
with open('/scratch/pstjohn/redf_spins.csv', 'wt') as f:
    f.write('smiles, atom_type, atom_index, spin\n')
    f.writelines((', '.join((str(x) for x in data)) + '\n' for data in df_generator(redf)))

with psycopg2.connect(**dbparams) as conn:

    cdf = pd.read_sql_query("""
    SELECT * from unique_compound 
    where type='fragment' and status='finished'
    """, conn)
    
with open('/scratch/pstjohn/cdf_spins.csv', 'wt') as f:
    f.write('smiles, atom_type, atom_index, spin\n')
    f.writelines((','.join((str(x) for x in data)) + '\n' for data in df_generator(cdf)))
