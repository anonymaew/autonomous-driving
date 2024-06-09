"""
1. python -m venv training
2. training\Scripts\activate (In base backend directory)
3. pip install h5py

Cuda environment setup:
1. cmd: nvidia-smi
2. Check what cuda version you would need to install (Right side).
3. Install: Correct CUDA Toolkit. (Example Toolkit: https://developer.nvidia.com/cuda-downloads)
4. Install: Correct torch version for your CUDA Toolkit within virtual environment from the website: https://pytorch.org/get-started/locally/ (Make sure to 'pip uninstall torch torchvision torchaudio' first)
Example command for synthura virtual environment: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from torchvision.models import resnet18

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        attention_weights = torch.sigmoid(self.conv(x))
        return x * attention_weights

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.avg_pool(x)
        attention = attention.view(attention.size(0), -1)
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        attention = attention.view(attention.size(0), attention.size(1), 1, 1)
        return x * attention

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Network(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        self.spatial_attention = SpatialAttention(512)
        self.channel_attention = ChannelAttention(512)

        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.steer_output = nn.Linear(64, 1)

    def forward(self, x):
        x = x.to(self.device)

        x = self.feature_extractor(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        steer = self.steer_output(x)

        return steer

def train(model, train_dataloader, val_dataloader, steer_criterion, optimizer, scheduler, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_steps = len(train_dataloader)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        model.train()
        running_train_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['image'].to(device)
            steer_targets = batch['steer'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            steer_output = model(images)
            steer_loss = steer_criterion(steer_output, steer_targets)

            steer_loss.backward()
            optimizer.step()

            running_train_loss += steer_loss.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_steps:
                print(f"  Batch [{batch_idx+1}/{total_steps}], Train Loss: {steer_loss.item():.4f}")

        epoch_train_loss = running_train_loss / total_steps

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['image'].to(device)
                steer_targets = batch['steer'].to(device).unsqueeze(1)

                steer_output = model(images)
                steer_loss = steer_criterion(steer_output, steer_targets)

                running_val_loss += steer_loss.item()

        epoch_val_loss = running_val_loss / len(val_dataloader)

        scheduler.step(epoch_val_loss)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")

    print("\nTraining completed!")
    print("Final model saved as 'model.pth'")
    torch.save(model.state_dict(), 'resnet.pth')

class CarlaDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'Images')
        self.steer_file = os.path.join(data_dir, 'SteerValues', 'steer_values.txt')

        self.image_files = sorted(os.listdir(self.image_dir))

        with open(self.steer_file, 'r') as file:
            self.steer_values = [float(line.strip()) for line in file]

        if len(self.image_files) != len(self.steer_values):
            raise ValueError("Number of image files and steer values do not match.")

        print(f"Number of image files: {len(self.image_files)}")
        print(f"Number of steer values: {len(self.steer_values)}")

    def __getitem__(self, index):
        image_file = self.image_files[index]
        steer_value = self.steer_values[index]

        image_path = os.path.join(self.image_dir, image_file)

        image = Image.open(image_path).convert('RGB')
        image = image.resize((200, 88))

        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        steer = torch.tensor(steer_value).float()

        return {'image': image, 'steer': steer}

    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    data_dir = 'archive/data'
    dataset = CarlaDataset(data_dir)

    print(f"Total dataset length: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"\nTrain dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    model = Network()
    model.feature_extractor[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    steer_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    num_epochs = 50

    print("\nGPU available:", torch.cuda.is_available())
    print("\nStarting training...")

    train(model, train_dataloader, val_dataloader, steer_criterion, optimizer, scheduler, num_epochs)