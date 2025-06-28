from torch_geometric.nn.conv import RGCNConv
from pprint import pprint
from convertmol import parse_sdf_file, bond_type_dict, single_bond_stereo_dict, double_bond_stereo_dict
from torch_geometric.data import Data, DataListLoader, Dataset, InMemoryDataset, Batch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import *
from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops, remove_self_loops

from typing import Tuple, List, Dict, Union
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
import torch
from torch import nn
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
class RGCN(nn.Module):
    def __init__(self, layers, atom_embedding_in, atom_embedding_out, num_relation):
        super().__init__()
        self.atom_embedding=nn.Embedding(atom_embedding_in, atom_embedding_out)#nn.Embedding(len(ATOMIC_SYMBOL), 15)
        self.num_relation=num_relation#len(BONDTYPE)
        self.layers=layers
        self.atom_embedding_in=atom_embedding_in
        self.atom_embedding_out=atom_embedding_out
        self.num_relation=num_relation
        rgcns=[]
        for i in range(len(layers)-1):
            rgcns.append(RGCNConv(layers[i], layers[i+1], self.num_relation))
        self.rgcns = nn.ModuleList(rgcns)
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'layers': self.layers,
            'atom_embedding_in': self.atom_embedding_in,
            'atom_embedding_out': self.atom_embedding_out,
            'num_relation': self.num_relation,
        }
    def forward(self, data):
        x=data.atom_type
        edge_index=data.edge_index
        edge_type=data.edge_type
        batch=data.batch
        #x, edge_index, edge_type, batch=None
        h=self.atom_embedding(x)
        if len(edge_type.shape)>1:
            edge_type = edge_type[:,0].long()
        for i,rgcn in enumerate(self.rgcns):
            h=rgcn(h, edge_index, edge_type)
            if i+1<len(self.rgcns):
                h=h.tanh()
        if batch is not None:
            h= scatter(h,batch,dim=0,reduce='mean').mean(-1)
        else:
            h= h.mean((-2,-1))
        return h#*width+offset
