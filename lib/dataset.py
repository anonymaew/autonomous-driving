import torch
import os
import h5py
from torchvision.transforms import v2 as transforms


def normalize(target):
    speed, steer = target
    speed = speed / 100
    steer = steer / 3.14 * 2
    return [speed, steer]


def unnormalize(target):
    speed, steer = target
    speed = speed * 100
    steer = steer * 3.14 / 2
    return [speed, steer]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        files = list(sorted(os.listdir(root)))
        # sample only a tenth
        self.files = files[::10]
        self.transforms = transforms

    # return an item from a file
    def __getitem__(self, idx):
        ifile, indx = divmod(idx, 200)
        filename = self.files[ifile]
        file = h5py.File(os.path.join(self.root, filename), 'r')

        img = file['rgb'][indx]
        # convert to [C, H, W]
        # img = torch.Tensor(img)
        img = self.transforms(img)
        targets = file['targets'][indx]
        # speed normalized to [-20,80]
        speed = targets[10]
        # steering normalized to [-pi/2,pi/2]
        steer = targets[0]
        targets = torch.Tensor(normalize([speed, steer]))
        return img, targets

    def __len__(self):
        # there are 200 data in each file
        return len(self.files) * 200


def resnet_transforms(fs=[]):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ] + fs + [
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
#                            0              1              2         3
# count  657600.000000  657600.000000  657600.000000  657600.0
# mean        0.001634       0.504136       0.174912       0.0
# std         0.155913       0.330658       0.379184       0.0
# min        -1.154329       0.000000       0.000000       0.0
# 25%        -0.025604       0.500000       0.000000       0.0
# 50%        -0.000035       0.500000       0.000000       0.0
# 75%         0.029197       0.531509       0.000000       0.0
# max         1.198919       1.000000       1.000000       0.0
#                    4              5              6              7
# count  657600.000000  657600.000000  657600.000000  657600.000000
# mean        0.000874       0.001198       0.504136       0.174912
# std         0.029557       0.269248       0.330658       0.379184
# min         0.000000      -1.084527       0.000000       0.000000
# 25%         0.000000      -0.044904       0.500000       0.000000
# 50%         0.000000      -0.000020       0.500000       0.000000
# 75%         0.000000       0.052067       0.531509       0.000000
# max         1.000000       1.162445       1.000000       1.000000
#                   8              9              10            11
# count  657600.000000  657600.000000  657600.000000  6.576000e+05
# mean    19979.611328   14595.629883      17.935951  3.656573e+05
# std     12157.963867   11404.205078      14.463256  1.541955e+06
# min      -402.615082    -366.399353     -18.739027  0.000000e+00
# 25%      9232.743408    4794.885498      10.102288  0.000000e+00
# 50%     18332.208984   13344.725586      18.982355  0.000000e+00
# 75%     33336.832031   22752.416016      22.477591  0.000000e+00
# max     39834.148438   33434.875000      82.729424  1.219020e+07
#                   12            13             14             15
# count  657600.000000  6.576000e+05  657600.000000  657600.000000
# mean     3903.930664  1.568297e+05       0.002759       0.000722
# std     28991.193359  7.332248e+05       0.036254       0.014483
# min         0.000000  0.000000e+00       0.000000       0.000000
# 25%         0.000000  0.000000e+00       0.000000       0.000000
# 50%         0.000000  0.000000e+00       0.000000       0.000000
# 75%         0.000000  0.000000e+00       0.000000       0.000000
# max    219261.093750  5.653156e+06       1.000000       1.000000
#                  16            17            18           19
# count  6.576000e+05  6.576000e+05  6.576000e+05     657600.0
# mean   2.700081e-02 -4.537514e-03  8.686926e-04  670389824.0
# std    1.061534e+01  1.059627e+01  6.254452e-01  230773792.0
# min   -9.322595e+02 -1.355793e+03 -2.753123e+01          0.0
# 25%   -1.765830e+00 -1.092384e+00 -7.357985e-02  684826224.0
# 50%   -2.184980e-10  1.553839e-13 -3.836304e-11  693298688.0
# 75%    1.731111e+00  1.293783e+00  5.107727e-02  702258384.0
# max    1.167349e+03  1.220937e+03  6.842194e+01  960998272.0
#                  20             21            22             23
# count  6.576000e+05  657600.000000  6.576000e+05  657600.000000
# mean   9.877211e+04       0.047544  1.126767e-02       0.000693
# std    2.346736e+05       0.747218  6.624379e-01       0.006823
# min    0.000000e+00      -1.000000 -1.000000e+00      -0.151561
# 25%    1.802300e+04      -0.926630 -1.289988e-01      -0.001767
# 50%    4.073600e+04       0.000865  1.020555e-07       0.000016
# 75%    6.793800e+04       0.994011  1.647543e-01       0.002510
# max    1.756740e+06       1.000000  1.000000e+00       0.105420
#                   24        25        26        27
# count  657600.000000  657600.0  657600.0  657600.0
# mean        3.356726       0.0       0.0       0.0
# std         1.173195       0.0       0.0       0.0
# min         0.000000       0.0       0.0       0.0
# 25%         2.000000       0.0       0.0       0.0
# 50%         3.000000       0.0       0.0       0.0
# 75%         4.000000       0.0       0.0       0.0
# max         5.000000       0.0       0.0       0.0
