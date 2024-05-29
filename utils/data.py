#currently an empty file for loading, wrangling data
# more to come
import pandas as pd
from utils.io import ase_db_to_df

def load_data(filename):
    # load data from a csv file and derive formula and species columns from structure
    df = ase_db_to_df(filename)
    # try:
    #     # structure provided as Atoms object
    #     df['structure'] = df['structure'].apply(eval).progress_map(lambda x: Atoms.fromdict(x))
    
    # except:
    #     # no structure provided
    #     species = []

    # else:
    #     df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    #     df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    #     species = sorted(list(set(df['species'].sum())))
    
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    species = sorted(list(set(df['species'].sum())))

    #df['bec'] = df['bec'].apply(eval).apply(np.array)

    return df , species