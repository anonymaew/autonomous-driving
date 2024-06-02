import torch
from torch import nn
from lib.dataset import resnet_transforms, unnormalize
from lib.model import ResNet, ViTNet
from tqdm import tqdm
from torchvision.transforms import v2 as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# inference
model = ResNet()
model.load('./best_resnet.pth', device)
model.to(device)
model.eval()

# TODO: load image
img = torch.randn(80, 200, 3).numpy()
img = resnet_transforms()(img).to(device).unsqueeze(0)

with torch.no_grad():
    pred = model.model_input(img)
    pred = unnormalize(pred.cpu()[0])
    print(f'Predicted: speed={pred[0]:.2f} km/h, steer={pred[1]:.2f}')
    # TODO: connect prediction to carla
