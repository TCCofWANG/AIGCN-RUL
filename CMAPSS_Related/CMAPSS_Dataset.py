import sys
sys.path.append("..")
from torch.utils.data import Dataset

"""
In this file, the three datasets will be processed by the Dataset; 
The output will be used as input to the Experiment_CMaps(gai).py
"""

class CMAPSSData(Dataset):

    def __init__(self, data_x, data_y, norm_index):
        self.data_x = data_x
        self.data_y = data_y
        self.norm_index = norm_index
        self.data_channel = data_x.shape[2]

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        norm_index = self.norm_index[index]
        sample_y = self.data_y[index]

        return sample_x, sample_y, norm_index

    def __len__(self):
        return len(self.data_x)