from torch_geometric.nn.conv import RGCNConv
from pprint import pprint
from convertmol import parse_sdf_file, bond_type_dict, single_bond_stereo_dict, double_bond_stereo_dict
from torch_geometric.data import Data, DataListLoader, Dataset, InMemoryDataset, Batch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import *
from torch_geometric.nn.models import SchNet as BaseSchNet
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
from typing import Optional, Callable
class SchNet(BaseSchNet):
    def __init__(self, 
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
                ):
        self.__hidden_channels=hidden_channels
        self.__num_filters=num_filters
        self.__num_interactions=num_interactions
        self.__num_gaussians=num_gaussians
        self.__cutoff=cutoff
        self.__interaction_graph=interaction_graph
        self.__max_num_neighbors=max_num_neighbors
        self.__readout=readout
        self.__dipole=dipole
        self.__mean=mean
        self.__std=std
        self.__atomref=atomref
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            interaction_graph=interaction_graph,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
            dipole=dipole,
            mean=mean,
            std=std,
            atomref=atomref,
        )
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'hidden_channels':self.__hidden_channels,
            'num_filters':self.__num_filters,
            'num_interactions':self.__num_interactions,
            'num_gaussians':self.__num_gaussians,
            'cutoff':self.__cutoff,
            'interaction_graph':self.__interaction_graph,
            'max_num_neighbors':self.__max_num_neighbors,
            'readout':self.__readout,
            'dipole':self.__dipole,
            'mean':self.__mean,
            'std':self.__std,
            'atomref':self.__atomref,
        }
    def forward(self, data):
        return super().forward(data.atom_type,data.pos,data.batch).squeeze(-1)