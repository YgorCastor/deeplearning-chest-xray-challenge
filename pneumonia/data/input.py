from typing import Tuple
import os

import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from pneumonia.data.preparation import PIL_INTERPOLATION_METHODS

SEED = 42

DATA_AUGMENTED_GENERATOR_PARAMS = dict(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    fill_mode="constant",
    cval=0.0,
)

RAW_GENERATOR_PARAMS = dict(
    rescale=1. / 255
)


def create_train_generator(dataset_dir: str, target_size: Tuple[int, int], interpolation: str,
                           batch_size: int, dir_name: str="train", data_augmented: bool = True, data_augmentation_params: dict = {},
                           width_resize_range: float=0.0,
                           seed: int = SEED) -> DirectoryIterator:
    generator_params = {**DATA_AUGMENTED_GENERATOR_PARAMS, **data_augmentation_params,
                        "preprocessing_function":
                            _resize_and_crop_width(target_size, width_resize_range, PIL_INTERPOLATION_METHODS[
                                interpolation])} if data_augmented else RAW_GENERATOR_PARAMS
    return ImageDataGenerator(**generator_params) \
        .flow_from_directory(os.path.join(dataset_dir, dir_name),
                             target_size=target_size,
                             batch_size=batch_size,
                             class_mode="binary",
                             interpolation=interpolation,
                             seed=seed)


def create_val_generator(dataset_dir: str, target_size: Tuple[int, int], interpolation: str,
                         batch_size: int, dir_name: str="val", seed: int = SEED) -> DirectoryIterator:
    return ImageDataGenerator(**RAW_GENERATOR_PARAMS) \
        .flow_from_directory(os.path.join(dataset_dir, dir_name),
                             target_size=target_size,
                             batch_size=batch_size,
                             class_mode="binary",
                             interpolation=interpolation,
                             shuffle=False,
                             seed=seed)


def create_test_generator(dataset_dir: str, target_size: Tuple[int, int], interpolation: str,
                          batch_size: int, dir_name: str="test", seed: int = SEED) -> DirectoryIterator:
    return ImageDataGenerator(**RAW_GENERATOR_PARAMS) \
        .flow_from_directory(os.path.join(dataset_dir, dir_name),
                             target_size=target_size,
                             batch_size=batch_size,
                             class_mode="binary",
                             interpolation=interpolation,
                             shuffle=False,
                             seed=seed)


def _resize_and_crop_width(target_size: Tuple[int, int], width_range: float, interpolation: int):
    def _parametrized_resize_and_crop_width(im: Image):
        width_ratio = 1. + np.random.uniform(-width_range, width_range)
        new_width = int(target_size[0] * width_ratio)

        im = im.resize((new_width, target_size[1]), interpolation)
        if new_width < target_size[0]:
            delta_w = target_size[0] - new_width
            padding = (delta_w // 2, 0, delta_w - (delta_w // 2), 0)
            im = ImageOps.expand(im, padding)
        else:
            im = ImageOps.fit(im, target_size)
        return im

    return _parametrized_resize_and_crop_width
