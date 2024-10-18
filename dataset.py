import torch
from torch.utils.data import Dataset
import dask.dataframe as dd
import os

class JSONL_Dataset(Dataset):
    def __init__(self, directories, normalize_y=True):
        self.file_paths = []
        for directory in directories:
            for file in os.listdir(directory):
                if file.endswith('.jsonl'):
                    self.file_paths.append(os.path.join(directory, file))

        # Load data into a Dask DataFrame
        self.data = dd.read_json(self.file_paths, lines=True)
        # Compute the DataFrame to avoid multiple loads
        self.data = self.data.compute()
        
        self.x = [torch.tensor(item) for item in self.data['accl']]
        self.y = [torch.tensor(item) for item in self.data['k']]
        self.x = torch.stack(self.x).permute(1, 0, 2)
        self.y = torch.stack(self.y).permute(1, 0)

        # Store original y for denormalization
        self.original_y = self.y.clone()

        # Normalize x
        self.x_mean = self.x.mean(dim=(0, 1), keepdim=True)
        self.x_std = self.x.std(dim=(0, 1), keepdim=True)
        self.x = (self.x - self.x_mean) / self.x_std

        # Optionally normalize y
        if normalize_y:
            self.y_mean = self.y.mean(dim=0, keepdim=True)
            self.y_std = self.y.std(dim=0, keepdim=True)
            self.y = (self.y - self.y_mean) / self.y_std
        else:
            self.y_mean = torch.zeros_like(self.y[0])
            self.y_std = torch.ones_like(self.y[0])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def denormalize_y(self, y):
        return y * self.y_std + self.y_mean 

# Create an instance of the dataset
# ds = JSONL_Dataset(['./data1', './data2', './data3'])