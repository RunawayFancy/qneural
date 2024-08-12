#%%
import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
from glob import glob


ANNOTATION = {
    'excited': 1,
    'ground': 0
}


class MyDataset(Dataset):
    def __init__(self, regex: str = '../Data/readout_iq/train/*.csv'):
        df = pd.DataFrame()
        self.regex = regex
        
        assert glob(regex)
        
        for file_path in glob(regex):
            new_df = pd.read_csv(file_path, header=None)
            
            new_df['label'] = ANNOTATION[os.path.basename(file_path).split('.')[0]]
            # print(new_df['label'])
            df = pd.concat([df, new_df])
        
        self.data = df.reset_index(drop=True).astype(np.float32)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        label = row.loc['label']
        data = row.drop('label').values
        return torch.from_numpy(data).float(), torch.Tensor([label]).float()

    def __len__(self):
        return len(self.data)
#%%
if __name__ == '__main__':
    dataset = MyDataset()
    dataset[0]




#%%

    