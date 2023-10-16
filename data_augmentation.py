#!/usr/bin/env python
# coding: utf-8

import numpy as np
import imgaug.augmenters as iaa


def augment():
    return iaa.SomeOf((0, 12), [
            iaa.Add((-250, 250)),
            iaa.Multiply((0.75, 1.25)),
            iaa.CropAndPad(percent=(-0.1, 0.1)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}, mode="constant", cval=-1024, order=[0,1]), 
            iaa.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}, mode="constant", cval=-1024, order=[0,1]),
            iaa.Affine(rotate=(-90, 90), mode="constant", cval=-1024, order = [0,1]),
            iaa.Affine(shear=(-15, 15), order = [0,1]),
            iaa.OneOf([
                iaa.Dropout((0.001,0.003)),
                iaa.SaltAndPepper((0.001,0.002)),
                ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.0, 2.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 5)),
                ]),
            iaa.AdditiveGaussianNoise(scale=(0, 0.25 * 255))
            ], random_order=True)

# obtain image_data from read_nii
# image_data: [width, height, slices]
# return images_deterministic: [width, height, slices]
def augmentor(image_data):
    aug = augment()

    # apply same operations to all slices
    aug_det = aug.to_deterministic()

    # convert to [slices, width, height, 1]
    # image_data = [np.expand_dims(image_data[:, :, i], axis=-1) for i in range(image_data.shape[-1])]
    images_deterministic = aug_det.augment_images([image_data])
    return np.asarray(images_deterministic[0])



