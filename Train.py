import torch
from torch import nn, optim
from Dataloader import create_data_loaders
from Model import Model # Change Model here
import numpy as np
from tqdm import tqdm
import json

def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in tqdm(val_loader, desc="Validation", leave=False)]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    device = get_default_device()
    model = to_device(model, device)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
     # Save history to JSON
    with open("train-history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    return history

if __name__ == '__main__':
    device = get_default_device()
    loaders, _ = create_data_loaders()
    train_dl = DeviceDataLoader(loaders['train'], device)
    val_dl = DeviceDataLoader(loaders['val'], device)
    a, b, c = next(iter(train_dl))
    
    model = Model()
    model = to_device(model, device)
    
    num_epochs = 100
    lr = 0.0001
    history = fit(num_epochs, lr, model, train_dl, val_dl)
    
    # best_model = Model()
    # best_model.load_state_dict(torch.load('best_model.pth'))
