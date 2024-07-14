import os
from glob import glob
import torch
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,

)
from monai.data import Dataset, DataLoader
from monai.utils import first

data_dir = 'datasets'

file_extension = '*.nii.gz'
train_imgs = sorted(glob(os.path.join(data_dir, 'TrainData', file_extension)))
train_labels = sorted(glob(os.path.join(data_dir, 'TrainLabels', file_extension)))
val_imgs = sorted(glob(os.path.join(data_dir, 'ValData', file_extension)))
val_labels = sorted(glob(os.path.join(data_dir, 'ValLabels', file_extension)))

train_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(train_imgs, train_labels)]
val_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(val_imgs, val_labels)]

keys = ['image', 'label']
orig_transforms = Compose(
    [
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        ToTensord(key=keys)
    ]
)

train_transforms = Compose(
    [
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2)),
        ScaleIntensityRanged(keys='image', a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key='image'),
        Resized(keys=keys, spatial_size=[128, 128, 128]),
        ToTensord(keys=keys)
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2)),
        ScaleIntensityRanged(keys='image', a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=keys)
    ]
)

orig_ds = Dataset(data=train_files, transform=orig_transforms)
orig_loader = DataLoader(orig_ds, batch_size=1)

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

test_pat = first(train_loader)
orig_pat = first(orig_loader)

print(torch.min(test_pat['image']))
print(torch.max(test_pat['image']))

plt.figure('Test', figsize=(12, 6))

plt.subplot(131)
plt.title('Orig Patient')
plt.imshow(orig_pat['image'][0, 0, :, :, 30], cmap='gray')

plt.subplot(132)
plt.title('Slice of a Patient')
plt.imshow(test_pat['image'][0, 0, :, :, 30], cmap='gray')

plt.subplot(133)
plt.title('Label of a Patient')
plt.imshow(test_pat['label'][0, 0, :, :, 30])

plt.show()