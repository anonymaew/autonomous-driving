import torch
import torch.nn as nn
from torchvision import models
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()
        self.model = None
        self.model_input = None
        self.name = name

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(path, map_location=device))

    def params(self):
        return self.model.parameters()

    def train_model(self, train_loader, val_loader, criterion, optimizer):
        best_val_loss = float('inf')
        early_stop, patience = 0, 5
        loss_train, loss_val = [], []
        while True:
            self.model.train()
            train_loader = tqdm(train_loader)
            loss_train.append(0)
            for img, targets in train_loader:
                img, targets = img.to(device), targets.to(device)
                optimizer.zero_grad()
                output = self.model_input(img)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                train_loader.set_postfix({'loss': loss.item()})
                loss_train[-1] += loss.item()/len(train_loader)
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for img, targets in val_loader:
                    img, targets = img.to(device), targets.to(device)
                    output = self.model_input(img)
                    loss = criterion(output, targets)
                    val_loss += loss.item()
            loss_val.append(val_loss/len(val_loader))
            plt.clf()
            plt.plot(loss_train, label='train')
            plt.plot(loss_val, label='val')
            plt.legend()
            plt.show()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(f'best_{self.name}.pth')
                early_stop = 0
            else:
                early_stop += 1
            if early_stop >= patience:
                break


class ResNet(Model):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__('resnet')
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 2)
        )
        self.model = self.resnet.fc
        self.model_input = self.resnet


class ViTNet(Model):
    def __init__(self, num_classes=2):
        super(ViTNet, self).__init__('vit')
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Sequential(
            nn.Linear(self.vit.head.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 2)
        )
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.head.parameters():
            param.requires_grad = True
        self.model = self.vit.head
        self.model_input = self.vit
