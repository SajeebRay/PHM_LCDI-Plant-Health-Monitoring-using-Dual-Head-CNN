#DH-CNN Model

import torch
from torch import nn
import torch.nn.functional as F

class ImageClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.best_val_acc2 = 0.0

    def training_step(self, batch):
        images, labels1, labels2 = batch
        labels1, labels2 = labels1.long(), labels2.long()  # Ensure labels are long tensors
        out1, out2 = self(images)
        loss1 = F.cross_entropy(out1, labels1)
        loss2 = F.cross_entropy(out2, labels2)
        loss = loss1 + loss2
        return loss

    def validation_step(self, batch):
        images, labels1, labels2 = batch
        labels1, labels2 = labels1.long(), labels2.long()  # Ensure labels are long tensors
        out1, out2 = self(images)
        loss1 = F.cross_entropy(out1, labels1)
        loss2 = F.cross_entropy(out2, labels2)
        loss = loss1 + loss2
        acc1 = self.accuracy(out1, labels1)  
        acc2 = self.accuracy(out2, labels2) 
        return {'val_loss': loss.detach(), 'val_acc1': acc1, 'val_acc2': acc2}

        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs1 = [x['val_acc1'] for x in outputs]
        epoch_acc1 = torch.stack(batch_accs1).mean()
        batch_accs2 = [x['val_acc2'] for x in outputs]
        epoch_acc2 = torch.stack(batch_accs2).mean()
        
        if epoch_acc2 > self.best_val_acc2:
            self.best_val_acc2 = epoch_acc2
            torch.save(self.state_dict(), 'best_model.pth')
        
        return {'val_loss': epoch_loss.item(), 'val_acc1': epoch_acc1.item(), 'val_acc2': epoch_acc2.item()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, "
              f"val_acc1: {result['val_acc1']:.4f}, val_acc2: {result['val_acc2']:.4f}")
    
    @staticmethod
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class Model(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(1024*4*4, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(256, 14)  # Plant name prediction
        self.fc2 = nn.Linear(256, 38)  # Disease prediction
        
    def forward(self, xb):
        x = self.network(xb)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2
