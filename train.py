import torch
from torch import nn
from lib.dataset import MyDataset, resnet_transforms
from lib.model import ResNet, ViTNet
from tqdm import tqdm
from torchvision.transforms import v2 as transforms


BATCH_SIZE = 64
NUM_WORKERS = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# root_data = '../data/carla'
root_data = './AgentHuman'

train_dataset = MyDataset(f'{root_data}/SeqTrain', resnet_transforms(
    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)]))
train_dataset = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataset = MyDataset(f'{root_data}/SeqVal', resnet_transforms())
test_dataset = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.params(), lr=0.001)
for model_class in [ResNet, ViTNet]:
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
