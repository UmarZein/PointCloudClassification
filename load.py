from tqdm.auto import tqdm
import os
import torch
from torch_geometric.data import InMemoryDataset, Data

import numpy as np

def read_off_fast(path):
    with open(path, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise ValueError(f'Invalid OFF header: {path}')
        n_verts, n_faces, _ = map(int, f.readline().strip().split())
        data = np.loadtxt(f, max_rows=n_verts)  # FAST bulk load
    return torch.from_numpy(data).float()

class ModelNet40PyG(InMemoryDataset):
    def __init__(self, root_dir, split='train', transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.the_raw_dir = os.path.join(self.root_dir,'raw')
        super().__init__(root_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Required but not used; raw data lives in category folders
        return []

    @property
    def processed_file_names(self):
        return [f'modelnet40_{self.split}.pt']

    def download(self):
        # Assume files already downloaded/unzipped in root_dir
        pass

    def process(self):
        data_list = []
        classes = sorted(os.listdir(self.the_raw_dir))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls in tqdm(classes, smoothing=0.0, dynamic_ncols=True):
            split_dir = os.path.join(self.the_raw_dir, cls, self.split)
            if not os.path.isdir(split_dir):
                continue
            label = class_to_idx[cls]
            for fname in tqdm(os.listdir(split_dir), desc=cls, smoothing=0.0, dynamic_ncols=True):
                if fname.endswith('.off'):
                    path = os.path.join(split_dir, fname)
                    pos = read_off_fast(path)  # (N, 3)
                    data = Data(pos=pos, y=torch.tensor([label], dtype=torch.long))
                    data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
from torch_geometric.loader import DataLoader

train = ModelNet40PyG('./data/uncompressed/ModelNet40/', split='train')
test = ModelNet40PyG('./data/uncompressed/ModelNet40/', split='test')