import glob
import os
import numpy as np
import torch
import monai
from monai.transforms import (
    AddChanneld,
    CastToTyped,
    LoadNiftid,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.data import DataLoader, Dataset


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadNiftid(keys),
        AddChanneld(keys),  #adds channels dimension C x H x W x D i.e [512, 512, 291] to [1, 512, 512, 291]
        Orientationd(keys, axcodes="LPS"),  #from which direction to view
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),  #Initially the pixdim/voxel spacing was 0.74x0.74x1
                                                                                              #And origin resolution was 512x512x291
                                                                                              #Now resolution will be ceil(512 * 0.74/1.25 x 512*0.74/1.25 X 291 *1/5)
                                                                                              #=304x304x59
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),  #originally, max Intensity=1527.1516, min Intensty=-2048
                                                                                                     #we want min Intensity = 0, max. Intensity = 1
                                                                                                     #-1000 will be mapped to 0, 500 will be mapped to 1
                                                                                                     #and I>500 will be mapped to 1, I<-1000 will be mapped to 0                                                                                                    #-1000<I<500 will be mapped to 0<I_t<1
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(-0.05, 0.05),
                    scale_range=(-0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),  #create a list of 3 samples, by cropping
                                                                                                              #the positions where there is no foreground
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),   #randomly add Gaussian noise to image, std should be small, preserve shape
                RandFlipd(keys, spatial_axis=0, prob=0.5),          #randomly flip image along axes, preserve shape
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)


def get_train_loader(data_folder="COVID-19-20_v2", batch_size=2, train_fraction=0.9):
    images = sorted(glob.glob(os.path.join(data_folder, "Train", "*_ct.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_folder, "Train", "*_seg.nii.gz")))
    keys = ("image", "label")
    train_frac = train_fraction
    val_frac = 1 - train_frac
    n_train = int(train_frac * len(images)) + 1
    n_val = min(len(images) - n_train, int(val_frac * len(images)))

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[1]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]

    train_transforms = get_xforms("train", keys)
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_transforms = get_xforms("val", keys)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
