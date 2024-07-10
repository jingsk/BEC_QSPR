# model
import torch
import torch_geometric as tg
import torch_scatter
import e3nn
from e3nn import o3
from e3nn.io import CartesianTensor
from e3nn.o3 import ReducedTensorProducts
from typing import Dict, Union
from becqsdr.data import train_valid_test_split
#from utils.e3nn import Network
from becqsdr.model import E3NN

import cmcrameri.cm as cm

# crystal structure data
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms

# data pre-processing and visualization
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd

# utilities
from tqdm import tqdm
from becqsdr.data import load_db
import yaml
import wandb

import warnings
warnings.filterwarnings("ignore")

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_df = True

r_max = 3.5 # cutoff radius
db_file_name = 'data/bec_run.db'

with open("./config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# build data
def build_data(entry, am_onehot, type_encoding, type_onehot, r_max=3.5):

    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    
    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    
    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x_in=am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z_in=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        y=CartesianTensor("ij=ji").from_cartesian(torch.from_numpy(entry.diel), rtp=ReducedTensorProducts('ij=ji', i='1o')).unsqueeze(0),
        b=CartesianTensor("ij=ij").from_cartesian(torch.from_numpy(entry.bec)).unsqueeze(0)
    )

    return data


def load_build_data(db_file_name, r_max):
    # load data
    df, species = load_db(db_file_name)
    species = [Atom(k).number for k in species]
    Z_max = max([Atom(k).number for k in species])

    # one-hot encoding atom type and mass
    type_encoding = {}
    specie_am = []
    for Z in range(1, Z_max+1):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
        specie_am.append(specie.mass)

    type_onehot = torch.eye(len(type_encoding))
    am_onehot = torch.diag(torch.tensor(specie_am))
    if load_df:
        df = pd.read_pickle("./df_data.pkl")
    else:
        df['data'] = df.progress_apply(lambda x: build_data(x, am_onehot, type_encoding, type_onehot, r_max), axis=1)
        df.to_pickle("./df_data.pkl")
    return df, Z_max

df, Z_max = load_build_data(db_file_name, r_max)
print(f'Zmax={Z_max}')

# Train/valid/test split
test_size = 0.1
idx_train, idx_valid, idx_test = train_valid_test_split(df.data, valid_size=test_size, test_size=test_size)
# Format dataloaders
batch_size = 4
dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].tolist(), batch_size=batch_size, shuffle=True)
dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].tolist(), batch_size=batch_size)
dataloader_test = tg.loader.DataLoader(df.iloc[idx_test]['data'].tolist(), batch_size=batch_size)


run = wandb.init(config=config)
lr = wandb.config.lr
num_neighbors = wandb.config.num_neighbors
emb_dim = wandb.config.emb_dim
num_layers = wandb.config.num_layers
max_iter = wandb.config.max_iter

args_enn = {'in_dim': Z_max,
            'emb_dim': emb_dim,
            'num_layers': num_layers,
            'max_radius': r_max,
            'num_neighbors': num_neighbors,
        }

enn = E3NN(**args_enn).to(device)
opt = torch.optim.Adam(enn.parameters(), lr=lr)
scheduler = None #torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
model_name = f'lr{lr}_num_neighbors{num_neighbors}_emb_dim{emb_dim}_num_layers{num_layers}'
model_path = f'models/{model_name}.torch'

resume = False
    
if resume:
    print(f'loading from {model_path}')
    saved = torch.load(model_path, map_location=device)
    enn.load_state_dict(saved['state'])
    opt.load_state_dict(saved['optimizer'])
    try:
        scheduler.load_state_dict(saved['scheduler'])
    except:
        scheduler = None
    history = saved['history']
    s0 = history[-1]['step'] + 1
    print(f'Starting from step {s0:d}')

else:
    history = []
    s0 = 0

# fit E3NN
for results in enn.fit(opt, dataloader_train, dataloader_valid, history, s0, max_iter=max(0,max_iter-s0), device=device,
                    scheduler=scheduler):
    with open(model_path, 'wb') as f:
        torch.save(results, f)
    wandb.log({"val_loss": history[-1]['valid'], 
               "train_loss": history[-1]['train'],
               "epoch": history[-1]['step']})

#visualize training and model
#saved = torch.load(model_path, map_location=device)
#history = saved['history']

# steps = [d['step'] + 1 for d in history]
# loss_train = [d['train']['loss'] for d in history]
# loss_valid = [d['valid']['loss'] for d in history]

# fig, ax = plt.subplots(figsize=(3.5,3))
# ax.plot(steps, loss_train, label='Train', color='red')
# ax.plot(steps, loss_valid, label='Valid.', color='blue')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Loss')
# ax.legend(frameon=False)
# ax.set_yscale('log')
# fig.savefig('images/loss.png', dpi=300, bbox_inches='tight')

# from becqsdr.model import visualize_output

# #enn.load_state_dict(saved['state'])
# entry = df.iloc[idx_test].iloc[0]
# visualize_output(entry, enn, device)

