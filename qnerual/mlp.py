#%%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataloading import MyDataset
from utils import AvgMeter, get_class_name

from datetime import datetime

class MLPClassifier(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int = 1) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hid), 
            nn.LeakyReLU(.1),
            nn.Dropout(.5),
            nn.Linear(d_hid, d_hid),
            nn.LeakyReLU(.1),
            nn.Dropout(.5),
            nn.Linear(d_hid, d_out),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def train_classifier(model: MLPClassifier, dataset: Dataset, epochs: int = 1000, device: str = 'cpu') -> None:
    print(device)
    # important to set shuffle=True
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # put model on device
    model.to(device)

    for epoch in range(epochs):
        for batch, target in loader:
            
            # put batch data on device
            batch, target = batch.to(device), target.to(device)
        
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print(f'Epoch: {epoch+1}, Batch Loss: {loss.item()}')


def load_model(ckpt_path: str) -> MLPClassifier:
    checkpoint = torch.load(ckpt_path)
    # checkpoint is a dict, = how you saved it
    
    # re-init model skeleton
    model = MLPClassifier(**checkpoint['init_kwargs'])
    # put trained weights into model
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def evaluate_model(dataset: Dataset, model: MLPClassifier, device: str = 'cpu'):
    loader = DataLoader(dataset)
    model.to(device)
    acc = AvgMeter()
    with torch.no_grad():
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            # Sigmod outputs normaliz to (0,1)
            # As labels here is 0 and 1 only. 
            # e.g., round 0.3 -> 0, 0.8 -> 1
            output = model(batch).round()
            acc.update(torch.sum(output==target).item())
    print(acc.avg)
    pass


def infer_model(dataset: Dataset, model: MLPClassifier, device: str = 'cpu'):
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

    # data to train on
    train_data_pth = '../Data/readout_iq/train/*.csv'
    dataset = MyDataset(regex=train_data_pth)
    
    # hyperparameters
    epochs = 10
    init_kwargs = {'d_in': 2000, 'd_hid': 100, 'd_out': 1}
    
    # model init
    mymodel = MLPClassifier(**init_kwargs)
    out = train_classifier(mymodel, dataset, epochs=epochs, 
                           device=device)    
    torch.save({'state_dict': mymodel.state_dict(),
                'epoch': epochs, 
                'init_kwargs': init_kwargs,
                'data_path': train_data_pth}, f'../log/{get_class_name(mymodel)}_{datetime.now():%Y%m%d_%H%M%S}.pth') 
    print(f'Log pth file save to ../log/{get_class_name(mymodel)}_{datetime.now():%Y%m%d_%H%M%S}.pth')
    pass

#%% Load

mymodel = load_model('../log/20240811-225129.pth')

#%% Evaluate

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('warn: you are training on cpu')

# data to train on
train_data_pth = '../Data/readout_iq/evalu/*.csv'
dataset = MyDataset(regex=train_data_pth)

evaluate_model(dataset, mymodel, device)