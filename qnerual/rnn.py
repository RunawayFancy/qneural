#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataloading import MyDataset
from utils import AvgMeter, get_class_name

from datetime import datetime

class RNNClassifier(nn.Module):
    def __init__(self, d_in: int = 2, d_hid: int = 8, d_out: int = 1) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=d_hid, num_layers=1, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(d_hid, d_out),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        
        """
        Do not pad, because we have a fixed size of (m, n),
        e.g. (1000, 2) = 2000 in total
        m for number of samples per signal
        n for number of signals

        e.g., for a IQ sequence, asusume a single shot measurement gives 
        1000 sample points of [I,Q], then, m=1000, n=2
        """
        seq = torch.nn.utils.rnn.pack_sequence(torch.split(x, x.shape[0], dim=0))
        _, (hidden_state, _) = self.lstm(seq)
        o = self.output(hidden_state)
        return o


def train_rnn(model: RNNClassifier, dataset: Dataset, epochs: int = 1000, device: str = 'cpu') -> None:
    print(device)
    # important to set shuffle=True
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    # put model on device
    model.to(device)
    loss_meter = AvgMeter()
    for epoch in range(epochs):
        loss_meter.reset()
        for batch, target in loader:
            
            # put batch data on device
            batch, target = batch.to(device).view(1000, 2), target.to(device)
        
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.flatten(), target.flatten())

            loss_meter.update(loss.item())

            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print(f'Epoch: {epoch+1}, Batch Loss: {loss_meter.avg}')
    

def load_model(ckpt_path: str) -> RNNClassifier:
    checkpoint = torch.load(ckpt_path)
    # checkpoint is a dict, = how you saved it
    
    # re-init model skeleton
    model = RNNClassifier(**checkpoint['init_kwargs'])
    # put trained weights into model
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def evaluate_model(dataset: Dataset, model: RNNClassifier, device: str = 'cpu'):
    loader = DataLoader(dataset)
    model.to(device)
    acc = AvgMeter()
    acc_list = []
    with torch.no_grad():
        for batch, target in loader:
            batch, target = batch.to(device).view(1000, 2), target.to(device)
            # Sigmod outputs normaliz to (0,1)
            # As labels here is 0 and 1 only. 
            # e.g., round 0.3 -> 0, 0.8 -> 1
            output = model(batch).round()
            acc_list.append(torch.sum(output.flatten() == target.flatten()).item())
            acc.update(torch.sum(output.flatten() == target.flatten()).item())
    print(acc.avg)
    return acc_list


def infer_model(dataset: Dataset, model: RNNClassifier, device: str = 'cpu'):
    """
    If dataset has no label. 
    """
    evaluate_model(dataset, model, device)

    pass



#%%
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('warn: you are training on cpu')

    # data loading
    train_data_pth = '../Data/readout_iq/train/*.csv'
    dataset = MyDataset(regex=train_data_pth)
    
    # hyperparameters
    epochs = 10
    init_kwargs = {'d_in': 2, 'd_hid': 8, 'd_out': 1}

    # train RNN
    mymodel = RNNClassifier(**init_kwargs)
    train_rnn(mymodel, dataset, epochs=epochs, device='cuda:0')\
    
    torch.save({'state_dict': mymodel.state_dict(),
                'epoch': epochs, 
                'init_kwargs': init_kwargs,
                'data_path': train_data_pth}, f'../log/{get_class_name(mymodel)}_{datetime.now():%Y%m%d_%H%M%S}.pth') 
    print(f'Log pth file save to ../log/{get_class_name(mymodel)}_{datetime.now():%Y%m%d_%H%M%S}.pth')
    pass
    
#%% Load

mymodel = load_model('../log/RNNClassifier_20240812_001635.pth')

#%% Evaluate

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('warn: you are training on cpu')

# data to train on
train_data_pth = '../Data/readout_iq/evalu/*.csv'
dataset = MyDataset(regex=train_data_pth)

acc_lst = evaluate_model(dataset, mymodel, device)

# %%
