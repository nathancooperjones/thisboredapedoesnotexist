"""
Modified training script adapted from ``waifu2x``.

Original training script can be found here: https://github.com/yu45020/Waifu2x/blob/master/train.py

This script (as well as ``custom_dataset.py``) should be moved into the ``Waifu2x`` directory when
ready to use, and run from the terminal with something similar to:

```
python custom_training_script.py --max_epochs 10
```

"""
import os
import pathlib

import fire
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import trange

from custom_dataset import CustomImageDataWithExtraNoise
from Models import CARN_V2
from utils import image_quality
from utils.cls import CyclicLR


def training_loop(
    initial_model_checkpoint_path: str = 'model_check_points/CRAN_V2',
    train_folder: str = '/home/nathancooperjones/GitHub/apebase/superresolution/train',
    test_folder: str = '/home/nathancooperjones/GitHub/apebase/superresolution/test',
    model_save_path: str = 'ape_model',
    max_epochs: int = 12,
    train_batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    patch_size: int = 192,
    noise_level: int = 2,
    save_test_images: bool = False,
    verbose: bool = True,
):
    """
    Modified training script to train a super-resolution model. Major changes include:

    1) Train the model in FP32, not FP16.

    2) Use the custom dataset with extra noise added to it.

    3) Add in arguments via ``fire`` to change training loop parameters without having to modify the
    script.

    """
    (
        pathlib.Path(os.path.join(model_save_path, 'check_test_imgs/'))
        .mkdir(parents=True, exist_ok=True)
    )

    img_dataset = CustomImageDataWithExtraNoise(img_folder=train_folder,
                                                patch_size=patch_size,
                                                shrink_size=2,
                                                noise_level=noise_level,
                                                down_sample_method=None,
                                                color_mod='RGB')
    img_data = DataLoader(img_dataset, batch_size=train_batch_size, shuffle=True, num_workers=6)

    total_batch = len(img_data)
    print(len(img_dataset))

    test_dataset = CustomImageDataWithExtraNoise(img_folder=test_folder,
                                                 patch_size=patch_size,
                                                 shrink_size=2,
                                                 noise_level=noise_level,
                                                 down_sample_method=None,
                                                 color_mod='RGB')
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    criteria = nn.L1Loss()

    model = CARN_V2(color_channels=3,
                    mid_channels=64,
                    conv=nn.Conv2d,
                    single_conv_size=3,
                    single_conv_group=1,
                    scale=2,
                    activation=nn.LeakyReLU(0.1),
                    SEBlock=True,
                    repeat_blocks=3,
                    atrous=(1, 1, 1))

    model.total_parameters()

    # model.initialize_weights_xavier_uniform()

    model = model.cuda()
    model.load_state_dict(
        torch.load(os.path.join(initial_model_checkpoint_path, 'CARN_model_checkpoint.pt'))
    )
    model = model.float()  # move out of FP16, just in case

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay,
                           amsgrad=True)

    last_iter = torch.load(
        os.path.join(initial_model_checkpoint_path, 'CARN_scheduler_last_iter.pt')
    )
    scheduler = CyclicLR(optimizer,
                         base_lr=1e-4,
                         max_lr=1e-4,
                         step_size=3 * total_batch,
                         mode='triangular',
                         last_batch_iteration=last_iter)
    train_loss = []
    train_ssim = []
    train_psnr = []

    test_loss = []
    test_ssim = []
    test_psnr = []

    ibar = trange(max_epochs,
                  ascii=True,
                  maxinterval=1,
                  postfix={'avg_loss': 0, 'train_ssim': 0, 'test_ssim': 0})

    for i in ibar:
        for index, batch in enumerate(img_data):
            scheduler.batch_step()
            lr_img, hr_img = batch
            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda()

            # model.zero_grad()
            optimizer.zero_grad()

            outputs = model.forward(lr_img)
            outputs = outputs.float()

            loss = criteria(outputs, hr_img)

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            ssim = image_quality.msssim(outputs, hr_img).item()
            psnr = image_quality.psnr(outputs, hr_img).item()

            ibar.set_postfix(ratio=index / total_batch, loss=loss.item(),
                             ssim=ssim, batch=index,
                             psnr=psnr,
                             lr=scheduler.current_lr)
            train_loss.append(loss.item())
            train_ssim.append(ssim)
            train_psnr.append(psnr)

            # if verbose:
            #     print(f'Epoch {i} Train Loss: {train_loss[-1]}')
            #     print(f'Epoch {i} Train SSIM: {train_ssim[-1]}')
            #     print(f'Epoch {i} Train PNSR: {train_psnr[-1]}\n')

        # +++++++++++++++++++++++++++++++++++++
        #           End of One Epoch
        # -------------------------------------

        torch.save(model.state_dict(), os.path.join(model_save_path, 'CARN_model_checkpoint.pt'))
        torch.save(scheduler, os.path.join(model_save_path, 'CARN_scheduler_optim.pt'))
        torch.save(optimizer.state_dict(), os.path.join(model_save_path, 'CARN_adam_checkpoint.pt'))
        torch.save(scheduler.last_batch_iteration,
                   os.path.join(model_save_path, 'CARN_scheduler_last_iter.pt'))

        # +++++++++++++++++++++++++++++++++++++
        #           Test
        # -------------------------------------

        with torch.no_grad():
            ssim = []
            batch_loss = []
            psnr = []
            for index, test_batch in enumerate(test_data):
                lr_img, hr_img = test_batch
                lr_img = lr_img.cuda()
                hr_img = hr_img.cuda()

                lr_img_up = model(lr_img)
                lr_img_up = lr_img_up.float()
                loss = criteria(lr_img_up, hr_img)

                if save_test_images:
                    save_image([lr_img_up[0], hr_img[0]],
                               os.path.join(model_save_path, f'check_test_imgs/{index}.png'))
                batch_loss.append(loss.item())
                ssim.append(image_quality.msssim(lr_img_up, hr_img).item())
                psnr.append(image_quality.psnr(lr_img_up, hr_img).item())

            test_ssim.append(np.mean(ssim))
            test_loss.append(np.mean(batch_loss))
            test_psnr.append(np.mean(psnr))

            if verbose:
                print(f'Epoch {i} Test Loss: {train_loss[-1]}')
                print(f'Epoch {i} Test SSIM: {train_ssim[-1]}')
                print(f'Epoch {i} Test PNSR: {train_psnr[-1]}\n')
                print('-----\n')


if __name__ == '__main__':
    fire.Fire(training_loop)
