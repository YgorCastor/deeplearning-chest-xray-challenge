from typing import List, Tuple

import luigi
import os

from PIL import Image, ImageOps
from tqdm import tqdm
import requests
import math
import zipfile
from glob import glob
import random
import pathlib
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from pneumonia.data.augmentor import augment_folder

SEED = 42

DATASET_URL = "https://data.mendeley.com/datasets/rscbjbr9sj/2/files/41d542e7-7f91-47f6-9ff2-dd8e5a5a7861/ChestXRay2017.zip?dl=1"
ZIP_FILE = "output/ChestXRay2017.zip"
RAW_DATASET_DIR = "output/raw_dataset"
DATASET_DIR = "output/dataset"

PIL_INTERPOLATION_METHODS = dict(
    nearest=Image.NEAREST,
    bilinear=Image.BILINEAR,
    bicubic=Image.BICUBIC,
    hamming=Image.HAMMING,
    box=Image.BOX,
    lanczos=Image.LANCZOS,
)


class DownloadDataset(luigi.Task):
    dataset_url = luigi.Parameter(default=DATASET_URL)

    def output(self):
        return luigi.LocalTarget(ZIP_FILE)

    def run(self):
        # Streaming, so we can iterate over the response.
        r = requests.get(self.dataset_url, stream=True)
        outputPath = self.output().path

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        wrote = 0
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        with open(outputPath, 'wb') as f:
            for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                             unit_scale=True):
                wrote = wrote + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            raise ConnectionError("ERROR, something went wrong")


class UnzipDataset(luigi.Task):
    extracted_dataset_dir = luigi.Parameter(default=RAW_DATASET_DIR)

    def requires(self):
        return DownloadDataset()

    def output(self):
        return luigi.LocalTarget(self.extracted_dataset_dir)

    def run(self):
        with zipfile.ZipFile(self.input().path, "r") as zip_ref:
            zip_ref.extractall(self.output().path)
            
class SplitTrainVal(luigi.Task):
    val_size = luigi.FloatParameter(default=0.2)
    dataset_dir = luigi.Parameter(default=DATASET_DIR)

    def requires(self):
        return UnzipDataset()

    def output(self):
        return luigi.LocalTarget("%s_%.2fval" % (self.dataset_dir, self.val_size))

    def run(self):
        extracted_dataset = self.input().path

        train_normal_dir = "%s/train/NORMAL" % self.output().path
        train_pneumonia_dir = "%s/train/PNEUMONIA" % self.output().path
        val_normal_dir = "%s/val/NORMAL" % self.output().path
        val_pneumonia_dir = "%s/val/PNEUMONIA" % self.output().path
        test_normal_dir = "%s/test/NORMAL" % self.output().path
        test_pneumonia_dir = "%s/test/PNEUMONIA" % self.output().path

        pathlib.Path(train_normal_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(train_pneumonia_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(val_normal_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(val_pneumonia_dir).mkdir(parents=True, exist_ok=True)
        _symlink_dir("%s/chest_xray/test/NORMAL" % extracted_dataset, test_normal_dir)
        _symlink_dir("%s/chest_xray/test/PNEUMONIA" % extracted_dataset, test_pneumonia_dir)

        random.seed(SEED)
        normal_images: List[str] = glob("%s/chest_xray/train/NORMAL/*.jpeg" % extracted_dataset)
        pneumonia_images: List[str] = glob("%s/chest_xray/train/PNEUMONIA/*.jpeg" % extracted_dataset)
        random.shuffle(normal_images)
        random.shuffle(pneumonia_images)

        normal_split_index = int(self.val_size * len(normal_images))
        val_split_index = int(self.val_size * len(pneumonia_images))

        train_normal_images = normal_images[normal_split_index:]
        val_normal_images = normal_images[:normal_split_index]
        train_pneumonia_images = pneumonia_images[val_split_index:]
        val_pneumonia_images = pneumonia_images[:val_split_index]

        for image in train_normal_images:
            os.symlink(os.path.abspath(image), os.path.join(train_normal_dir, os.path.basename(image)))
        for image in val_normal_images:
            os.symlink(os.path.abspath(image), os.path.join(val_normal_dir, os.path.basename(image)))
        for image in train_pneumonia_images:
            os.symlink(os.path.abspath(image), os.path.join(train_pneumonia_dir, os.path.basename(image)))
        for image in val_pneumonia_images:
            os.symlink(os.path.abspath(image), os.path.join(val_pneumonia_dir, os.path.basename(image)))


class BalanceDataset(luigi.Task):
    val_size = luigi.FloatParameter(default=0.2)
    sampling_strategy = luigi.ChoiceParameter(choices=["oversample", "undersample"], default="oversample")

    def requires(self):
        return SplitTrainVal(val_size=self.val_size)

    def output(self):
        return luigi.LocalTarget("%s_%s" % (self.input().path, self.sampling_strategy))

    def run(self):
        dataset_dir = self.input().path

        train_normal_dir = "%s/train/NORMAL" % self.output().path
        train_pneumonia_dir = "%s/train/PNEUMONIA" % self.output().path
        val_normal_dir = "%s/val/NORMAL" % self.output().path
        val_pneumonia_dir = "%s/val/PNEUMONIA" % self.output().path
        test_normal_dir = "%s/test/NORMAL" % self.output().path
        test_pneumonia_dir = "%s/test/PNEUMONIA" % self.output().path

        pathlib.Path(train_normal_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(train_pneumonia_dir).mkdir(parents=True, exist_ok=True)
        _symlink_dir("%s/val/NORMAL" % dataset_dir, val_normal_dir)
        _symlink_dir("%s/val/PNEUMONIA" % dataset_dir, val_pneumonia_dir)
        _symlink_dir("%s/test/NORMAL" % dataset_dir, test_normal_dir)
        _symlink_dir("%s/test/PNEUMONIA" % dataset_dir, test_pneumonia_dir)

        normal_images: List[str] = glob("%s/train/NORMAL/*.jpeg" % dataset_dir)
        pneumonia_images: List[str] = glob("%s/train/PNEUMONIA/*.jpeg" % dataset_dir)

        images = [normal_images, pneumonia_images]

        X_normal = np.column_stack(
            (np.zeros(len(normal_images), dtype=int), np.arange(0, len(normal_images), dtype=int)))
        X_pneumonia = np.column_stack(
            (np.ones(len(pneumonia_images), dtype=int), np.arange(0, len(pneumonia_images), dtype=int)))

        X = np.concatenate((X_normal, X_pneumonia), axis=0)
        y = np.concatenate((np.zeros(len(normal_images)), np.ones(len(pneumonia_images))), axis=0)

        sampler = {
            "oversample": RandomOverSampler(random_state=42),
            "undersample": RandomUnderSampler(random_state=42),
        }

        random_sampler = sampler.get(self.sampling_strategy)

        X_res, y_res = random_sampler.fit_sample(X, y)

        for i, (x, y) in enumerate(zip(X_res, y_res)):
            image = images[x[0]][x[1]]
            if y == 1:
                os.symlink(os.path.abspath(image), "%s/%d.jpeg" % (train_pneumonia_dir, i))
            else:
                os.symlink(os.path.abspath(image), "%s/%d.jpeg" % (train_normal_dir, i))


class ResizeAndAugmentImages(luigi.Task):
    val_size = luigi.FloatParameter(default=0.2)
    sampling_strategy = luigi.ChoiceParameter(choices=["oversample", "undersample"], default="oversample")
    target_size = luigi.TupleParameter(default=(224, 224))
    interpolation = luigi.ChoiceParameter(choices=["nearest", "bilinear", "bicubic", "hamming", "box", "lanczos"],
                                          default="lanczos")
    keep_aspect_ratio = luigi.BoolParameter(default=True)
    keep_resized_for_training = luigi.BoolParameter(default=False)
    prior_augmentation = luigi.BoolParameter(default=False)
    prior_augmentation_times = luigi.IntParameter(default=1)
    verbose_augmentation = luigi.BoolParameter(default=False)

    def requires(self):
        return BalanceDataset(val_size=self.val_size, sampling_strategy=self.sampling_strategy)

    def output(self):
        if self.keep_aspect_ratio:
            identifier = "keep_aspect_ratio"
            if self.keep_resized_for_training:
                identifier += "+resize"
        else:
            identifier = "resize"
        if self.prior_augmentation:
            identifier += f"+augmented{self.prior_augmentation_times}x"

        return luigi.LocalTarget(
            "%s_%s_%s_%dx%d" % (self.input().path, self.interpolation, identifier,
                                self.target_size[0], self.target_size[1]))

    def run(self):
        dataset_dir = self.input().path

        train_normal_dir = "%s/train/NORMAL" % self.output().path
        train_pneumonia_dir = "%s/train/PNEUMONIA" % self.output().path
        val_normal_dir = "%s/val/NORMAL" % self.output().path
        val_pneumonia_dir = "%s/val/PNEUMONIA" % self.output().path
        test_normal_dir = "%s/test/NORMAL" % self.output().path
        test_pneumonia_dir = "%s/test/PNEUMONIA" % self.output().path

        pathlib.Path(train_normal_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(train_pneumonia_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(val_normal_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(val_pneumonia_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(test_normal_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(test_pneumonia_dir).mkdir(parents=True, exist_ok=True)

        interpolation: int = PIL_INTERPOLATION_METHODS[self.interpolation]

        _resize_images("%s/train/NORMAL" % dataset_dir, train_normal_dir, self.target_size,
                       interpolation, self.keep_aspect_ratio)
        _resize_images("%s/train/PNEUMONIA" % dataset_dir, train_pneumonia_dir, self.target_size,
                       interpolation, self.keep_aspect_ratio)
        if self.keep_aspect_ratio and self.keep_resized_for_training:
            _resize_images("%s/train/NORMAL" % dataset_dir, train_normal_dir, self.target_size,
                           interpolation, False)
            _resize_images("%s/train/PNEUMONIA" % dataset_dir, train_pneumonia_dir, self.target_size,
                           interpolation, False)
        _resize_images("%s/val/NORMAL" % dataset_dir, val_normal_dir, self.target_size,
                       interpolation, self.keep_aspect_ratio)
        _resize_images("%s/val/PNEUMONIA" % dataset_dir, val_pneumonia_dir, self.target_size,
                       interpolation, self.keep_aspect_ratio)
        _resize_images("%s/test/NORMAL" % dataset_dir, test_normal_dir, self.target_size,
                       interpolation, self.keep_aspect_ratio)
        _resize_images("%s/test/PNEUMONIA" % dataset_dir, test_pneumonia_dir, self.target_size,
                       interpolation, self.keep_aspect_ratio)

        if self.prior_augmentation:
            print("Prior augmentation enabled...")
            augment_folder(train_normal_dir, self.verbose_augmentation, self.prior_augmentation_times, seed=SEED)
            augment_folder(train_pneumonia_dir, self.verbose_augmentation, self.prior_augmentation_times, seed=SEED)
        else:
            print("Skipping prior augmentation...")


def _symlink_dir(src: str, dst: str):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        os.symlink(os.path.abspath(os.path.join(src, item)), os.path.join(dst, item))


def _resize_images(src: str, dst: str, target_size: Tuple[int, int], interpolation: int, keep_aspect_ratio: bool):
    for item in os.listdir(src):
        im = Image.open(os.path.join(src, item))
        if keep_aspect_ratio:
            old_size: Tuple[int, int] = im.size

            ratio = float(max(target_size)) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            im = im.resize(new_size, interpolation)

            delta_w = target_size[0] - new_size[0]
            delta_h = target_size[1] - new_size[1]
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            im = ImageOps.expand(im, padding)
        else:
            im = im.resize(target_size, interpolation)
        filename, ext = os.path.splitext(item)
        filename = "{name}_{uid}{ext}".format(name=filename,
                                              uid="aspect_ratio_kept" if keep_aspect_ratio else "resized",
                                              ext=ext)
        im.save(os.path.join(dst, filename), quality=100)
