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

from .mlp import MLP
class XGCNConv(MessagePassing):
    def __init__(
        self,
        mlp_dims_node: List[int],
        mlp_dims_edge: List[int],
        aggr: str = 'sum',
        dropout_rate: float = 0.0,
        **kwargs
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.node_dimses = mlp_dims_node
        self.edge_dimses = mlp_dims_edge
        self.aggr=aggr
        self.dropout_rate=dropout_rate
        self.node_mlp = nn.ModuleList([
            MLP(mlp_dims_node[0], mlp_dims_node[1:-1], mlp_dims_node[-1], dropout_rate=dropout_rate)
            for _ in range(mlp_dims_edge[-1])
        ])
        self.edge_mlp = MLP(mlp_dims_edge[0], mlp_dims_edge[1:-1], mlp_dims_edge[-1], dropout_rate=dropout_rate)
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'mlp_dims_node': self.node_dimses,
            'mlp_dims_edge': self.edge_dimses,
            'aggr': self.aggr,
            'dropout_rate': self.dropout_rate,
        }
    
    def forward(self, x: Tensor,
                edge_index: Adj, edge_attr: Tensor):
        #edge_attr: (typ, dist), |e|
        #edge_index: 2, |e|
        #print("edge_attr:",edge_attr.shape)
        edge_attr = self.edge_mlp(edge_attr)
        #print("edge_attr")
        size = (x.size(0), x.size(0))

        out = torch.zeros(x.size(0), self.node_dimses[-1], device=x.device)
        
        for i in range(self.edge_dimses[-1]):
            h = self.propagate(edge_index, edge_attr=edge_attr, x=x,
                               size=size, i=i)
            h2=self.node_mlp[i](h)
            out = out + h2
        return out
        
    def message(self, x_j: Tensor, edge_attr, i) -> Tensor:
        return x_j*edge_attr[:,i].view(-1,1)

        
class XGCN(nn.Module):
    def __init__(self, 
                 node_dimses, 
                 edge_dimses, 
                 atom_emb_i, 
                 atom_emb_o,
                 bond_emb_i,
                 bond_emb_o,
                 l_atom_emb_i,
                 l_atom_emb_o,
                 r_atom_emb_i,
                 r_atom_emb_o,
                 orig_atom_emb_i,
                 orig_atom_emb_o,
                 dest_atom_emb_i,
                 dest_atom_emb_o,
                 l_bond_emb_i,
                 l_bond_emb_o,
                 r_bond_emb_i,
                 r_bond_emb_o,
                 anchor_bond_emb_i,
                 anchor_bond_emb_o,
                 ring_agg_inner_dims,
                 ring_agg_o,
                 edge_combiner_inner_dims,
                 edge_combiner_o,
                 dropout_rate=0.0, 
                 aggr: str = 'sum'):
        super().__init__()
        __A=l_atom_emb_o
        __B=r_atom_emb_o
        __C=orig_atom_emb_o
        __D=dest_atom_emb_o
        __E=l_bond_emb_o
        __F=r_bond_emb_o
        __G=anchor_bond_emb_o
        #O1=15
        #O2=22
        self.atom_type_emb = nn.Embedding(atom_emb_i, atom_emb_o)#nn.Embedding(len(ATOMIC_SYMBOL),O1)
        self.bond_type_emb = nn.Embedding(bond_emb_i, bond_emb_o)#nn.Embedding(len(BONDTYPE),O2)
        #A=B=C=D=15
        self.l_atom_type_emb = nn.Embedding(l_atom_emb_i, l_atom_emb_o)#nn.Embedding(len(ATOMIC_SYMBOL),A)
        self.r_atom_type_emb = nn.Embedding(r_atom_emb_i, r_atom_emb_o)#nn.Embedding(len(ATOMIC_SYMBOL),B)
        self.anchor_orig_atom_type_emb = nn.Embedding(orig_atom_emb_i, orig_atom_emb_o)#nn.Embedding(len(ATOMIC_SYMBOL),C)
        self.anchor_dest_atom_type_emb = nn.Embedding(dest_atom_emb_i, dest_atom_emb_o)#nn.Embedding(len(ATOMIC_SYMBOL),D)
        #E=F=G=len(BONDTYPE)
        self.l_bond_type_emb = nn.Embedding(l_bond_emb_i, l_bond_emb_o)#nn.Embedding(len(BONDTYPE),E)
        self.r_bond_type_emb = nn.Embedding(r_bond_emb_i, r_bond_emb_o)#nn.Embedding(len(BONDTYPE),F)
        self.anchor_bond_type_emb = nn.Embedding(anchor_bond_emb_i, anchor_bond_emb_o)#nn.Embedding(len(BONDTYPE),G)
        
        self.xgcns = nn.ModuleList([
            XGCNConv(mlp_dims_node, mlp_dims_edge, dropout_rate=dropout_rate, aggr=aggr)
            for mlp_dims_node, mlp_dims_edge in zip(node_dimses, edge_dimses)
        ])

        self.ring_aggregator_mlp=MLP(__A+__B+__C+__D+__E+__F+__G+1,ring_agg_inner_dims,ring_agg_o)#MLP(__A+__B+__C+__D+__E+__F+__G+1,[50,64],22)
        self.aggregator_edge_combined = MLP(bond_emb_o+ring_agg_o,edge_combiner_inner_dims,edge_combiner_o)#MLP(22+22,[50,64],22)

        self.node_dimses=node_dimses 
        self.edge_dimses=edge_dimses 
        self.atom_emb_i=atom_emb_i 
        self.atom_emb_o=atom_emb_o
        self.bond_emb_i=bond_emb_i
        self.bond_emb_o=bond_emb_o
        self.l_atom_emb_i=l_atom_emb_i
        self.l_atom_emb_o=l_atom_emb_o
        self.r_atom_emb_i=r_atom_emb_i
        self.r_atom_emb_o=r_atom_emb_o
        self.orig_atom_emb_i=orig_atom_emb_i
        self.orig_atom_emb_o=orig_atom_emb_o
        self.dest_atom_emb_i=dest_atom_emb_i
        self.dest_atom_emb_o=dest_atom_emb_o
        self.l_bond_emb_i=l_bond_emb_i
        self.l_bond_emb_o=l_bond_emb_o
        self.r_bond_emb_i=r_bond_emb_i
        self.r_bond_emb_o=r_bond_emb_o
        self.anchor_bond_emb_i=anchor_bond_emb_i
        self.anchor_bond_emb_o=anchor_bond_emb_o
        self.ring_agg_inner_dims=ring_agg_inner_dims
        self.ring_agg_o=ring_agg_o
        self.edge_combiner_inner_dims=edge_combiner_inner_dims
        self.edge_combiner_o=edge_combiner_o
        self.dropout_rate=dropout_rate
        self.aggr=aggr

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def get_config(self):
        return {
            'node_dimses':self.node_dimses,
            'edge_dimses':self.edge_dimses,
            'atom_emb_i':self.atom_emb_i,
            'atom_emb_o':self.atom_emb_o,
            'bond_emb_i':self.bond_emb_i,
            'bond_emb_o':self.bond_emb_o,
            'l_atom_emb_i':self.l_atom_emb_i,
            'l_atom_emb_o':self.l_atom_emb_o,
            'r_atom_emb_i':self.r_atom_emb_i,
            'r_atom_emb_o':self.r_atom_emb_o,
            'orig_atom_emb_i':self.orig_atom_emb_i,
            'orig_atom_emb_o':self.orig_atom_emb_o,
            'dest_atom_emb_i':self.dest_atom_emb_i,
            'dest_atom_emb_o':self.dest_atom_emb_o,
            'l_bond_emb_i':self.l_bond_emb_i,
            'l_bond_emb_o':self.l_bond_emb_o,
            'r_bond_emb_i':aaself.r_bond_emb_i,
            'r_bond_emb_o':self.r_bond_emb_o,
            'anchor_bond_emb_i':self.anchor_bond_emb_i,
            'anchor_bond_emb_o':self.anchor_bond_emb_o,
            'ring_agg_inner_dims':self.ring_agg_inner_dims,
            'ring_agg_o':self.ring_agg_o,
            'edge_combiner_inner_dims':self.edge_combiner_inner_dims,
            'edge_combiner_o':self.edge_combiner_o,
            'dropout_rate':self.dropout_rate,
            'aggr':self.aggr,
        }
    
        
    def forward(self, 
                atom_type: Tensor,      # [|V|]
                edge_index: Adj,        # [2,|E|]
                edge_type: Tensor,      # [|E|]
                bond_anchor: Tensor,    # [|C|]
                bond_inbound: Tensor,   # [|C|, 2]
                angle_deltas: Tensor,   # [|C|]
                batch=None):
        
        l_edge_type = edge_type[bond_inbound[:,0]] # [|C|]
        r_edge_type = edge_type[bond_inbound[:,1]] # [|C|]
        anchor_type = edge_type[bond_anchor]       # [|C|]
        assert l_edge_type.shape==r_edge_type.shape==anchor_type.shape==bond_anchor.shape
        l_atom_type = atom_type[edge_index[0,bond_inbound[:,0]]]
        r_atom_type = atom_type[edge_index[0,bond_inbound[:,1]]]
        anchor_orig_atom_type = atom_type[edge_index[0,bond_anchor]]
        anchor_dest_atom_type = atom_type[edge_index[1,bond_anchor]]
        assert l_atom_type.shape==r_atom_type.shape==anchor_orig_atom_type.shape==anchor_dest_atom_type.shape==bond_anchor.shape

        l_edge_emb = self.l_bond_type_emb(l_edge_type) #[|C|,D_b]
        r_edge_emb = self.r_bond_type_emb(r_edge_type) #[|C|,D_b]
        anchor_edge_emb = self.anchor_bond_type_emb(anchor_type) #[|C|,D_b]

        l_atom_emb = self.l_atom_type_emb(l_atom_type)
        r_atom_emb = self.r_atom_type_emb(r_atom_type)
        anchor_orig_emb = self.anchor_orig_atom_type_emb(anchor_orig_atom_type)
        anchor_dest_emb = self.anchor_dest_atom_type_emb(anchor_dest_atom_type)
        #idk what to name it
        combined_ring_conv_map = torch.cat([
            l_edge_emb, 
            r_edge_emb, 
            anchor_edge_emb, 
            
            l_atom_emb, 
            r_atom_emb, 
            anchor_orig_emb,
            anchor_dest_emb,

            angle_deltas.view(-1,1)
        ],-1) 
        combined_ring_conv_map=self.ring_aggregator_mlp(combined_ring_conv_map)
        #print("combined_ring_conv_map.T:",combined_ring_conv_map.T.shape)
        #print("bond_anchor.unsqueeze(0):",bond_anchor.unsqueeze(0).shape)
        #print("edge_type:",edge_type.shape)
        ring_conv_out=scatter(combined_ring_conv_map.T, bond_anchor.unsqueeze(0), reduce='sum', dim_size=edge_type.shape[0]).T
        #print("ring_conv_out:",ring_conv_out.shape)
        edge_type_emb = self.bond_type_emb(edge_type)
        
        combined_edge = self.aggregator_edge_combined(torch.cat([
            edge_type_emb,
            ring_conv_out,
        ],-1))

        ########################
        h = self.atom_type_emb(atom_type)
        ########################
        for i,xgcn in enumerate(self.xgcns):
            h = xgcn(h, edge_index, combined_edge)
            if i+1<len(self.xgcns):
                h=h.tanh()
        if batch is not None:
            h= scatter(h,batch,dim=0,reduce='mean').mean(-1)
        else:
            h= h.mean((-2,-1))
        return h#*width+offset
