"""
Modified dataset augmentation with added augmentation adapted from ``waifu2x``.

Original dataset augmentation can be found here:
https://github.com/yu45020/Waifu2x/blob/master/Dataloader.py

This file (as well as ``custom_training_script.py``) should be moved into the ``Waifu2x`` directory
when ready to use.

"""
import glob
from io import BytesIO
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor


class CustomImageDataWithExtraNoise(Dataset):
    """``ImageData`` unmodified - instead, only augmentation below is modified."""
    def __init__(
        self,
        img_folder,
        patch_size=96,
        shrink_size=2,
        noise_level=1,
        down_sample_method=None,
        color_mod='RGB',
        dummy_len=None,
    ):

        self.img_folder = img_folder
        all_img = glob.glob(self.img_folder + "/**", recursive=True)
        self.img = list(
            filter(lambda x: x.endswith('png') or x.endswith("jpg") or x.endswith("jpeg"), all_img)
        )
        self.total_img = len(self.img)
        self.dummy_len = dummy_len if dummy_len is not None else self.total_img
        self.random_cropper = RandomCrop(size=patch_size)
        self.color_mod = color_mod
        self.img_augmenter = CustomImageAugmentWithExtraNoise(shrink_size,
                                                              noise_level,
                                                              down_sample_method)

    def get_img_patches(self, img_file):
        """Unmodified ``get_img_patches``."""
        img_pil = Image.open(img_file).convert("RGB")
        img_patch = self.random_cropper(img_pil)
        lr_hr_patches = self.img_augmenter.process(img_patch)

        return lr_hr_patches

    def __len__(self):
        """Unmodified ``__len__``."""
        return self.dummy_len  # len(self.img)

    def __getitem__(self, index):
        """Unmodified ``__getitem__``."""
        idx = random.choice(range(0, self.total_img))
        img = self.img[idx]
        patch = self.get_img_patches(img)

        if self.color_mod == 'RGB':
            lr_img = patch[0].convert("RGB")
            hr_img = patch[1].convert("RGB")
        elif self.color_mod == 'YCbCr':
            lr_img, _, _ = patch[0].convert('YCbCr').split()
            hr_img, _, _ = patch[1].convert('YCbCr').split()
        else:
            raise KeyError('Either RGB or YCbCr')

        return to_tensor(lr_img), to_tensor(hr_img)


class CustomImageAugmentWithExtraNoise:
    """
    Original class with ``add_jpeg_noise`` modified to add in salt-and-pepper noise to
    ``lr_img``.

    """
    def __init__(self, shrink_size=2, noise_level=1, down_sample_method=None):
        # noise_level (int): 0: no noise; 1: 75-95% quality; 2:50-75%
        if noise_level == 0:
            self.noise_level = [0, 0]
            self.gaussian_scale = 0
        elif noise_level == 1:
            self.noise_level = [5, 25]
            self.gaussian_scale = 0.5
        elif noise_level == 2:
            self.noise_level = [25, 50]
            self.gaussian_scale = 1.25
        else:
            raise KeyError("Noise level should be either 0, 1, 2")

        self.shrink_size = shrink_size
        self.down_sample_method = down_sample_method

    def shrink_img(self, hr_img):
        """Unmodified ``shrink_img``."""
        if self.down_sample_method is None:
            resample_method = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        else:
            resample_method = self.down_sample_method

        img_w, img_h = tuple(map(lambda x: int(x / self.shrink_size), hr_img.size))
        lr_img = hr_img.resize((img_w, img_h), resample_method)

        return lr_img

    def add_jpeg_noise(self, hr_img):
        """Modified to add in salt-and-pepper noise to ``lr_img``."""
        quality = 100 - round(random.uniform(*self.noise_level))
        lr_img = BytesIO()
        hr_img.save(lr_img, format='JPEG', quality=quality)
        lr_img.seek(0)
        lr_img = Image.open(lr_img)

        # add salt-and-pepper noise
        lr_img_array = np.asarray(lr_img)

        gaussian = np.random.normal(
            loc=0.0, scale=self.gaussian_scale, size=(lr_img_array.shape[0], lr_img_array.shape[1])
        )
        noisy_image = np.zeros(lr_img_array.shape, np.float32)
        noisy_image[:, :, 0] = lr_img_array[:, :, 0] + gaussian
        noisy_image[:, :, 1] = lr_img_array[:, :, 1] + gaussian
        noisy_image[:, :, 2] = lr_img_array[:, :, 2] + gaussian

        lr_img = Image.fromarray(np.uint8(noisy_image))

        return lr_img

    def process(self, hr_patch_pil):
        """Unmodified ``process``."""
        lr_patch_pil = self.shrink_img(hr_patch_pil)
        if self.noise_level[1] > 0:
            lr_patch_pil = self.add_jpeg_noise(lr_patch_pil)

        return lr_patch_pil, hr_patch_pil

    def up_sample(self, img, resample):
        """Unmodified ``up_sample``."""
        width, height = img.size

        return img.resize((self.shrink_size * width, self.shrink_size * height), resample=resample)
