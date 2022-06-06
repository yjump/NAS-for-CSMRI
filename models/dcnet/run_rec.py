"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
sys.path.append('/home/shuo/yanjp/tsi_mri')

import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.args import Args
from common.subsample import MaskFunc
from common.utils import save_reconstructions
from data import transforms
from data.mri_data import SliceData
from models.dcnet.recnet import RecNet


class DataTransform:
    """
    Data Transformer for running U-Net models on a test dataset.
    """

    def __init__(self, resolution, which_challenge, mask_func=None):
        """
        Args:
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.mask_func = mask_func

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.Array): k-space measurements
            target (numpy.Array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object
            fname (pathlib.Path): Path to the input file
            slice (int): Serial number of the slice
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Normalized zero-filled input image
                mean (float): Mean of the zero-filled image
                std (float): Standard deviation of the zero-filled image
                fname (pathlib.Path): Path to the input file
                slice (int): Serial number of the slice
        """
        kspace = transforms.to_tensor(kspace)
        gt = transforms.ifft2(kspace)
        gt = transforms.complex_center_crop(gt, (self.resolution, self.resolution))
        kspace = transforms.fft2(gt)

        if self.mask_func is not None:
            seed = tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.ones(masked_kspace.shape)
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        masked_kspace = transforms.fft2_nshift(image)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image_mod = transforms.complex_abs(image).max()
        image_r = image[:, :, 0]*6/image_mod
        image_i = image[:, :, 1]*6/image_mod

        # image_r = image[:, :, 0]
        # image_i = image[:, :, 1]
        #
        # image_r, mean_r, std_r = transforms.normalize_instance(image_r, eps=1e-11)
        # image_r = image_r.clamp(-6, 6)
        #
        # image_i, mean_i, std_i = transforms.normalize_instance(image_i, eps=1e-11)
        # image_i = image_i.clamp(-6, 6)


        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input

        image = np.stack((image_r, image_i), axis=-1)
        image = image.transpose((2, 0, 1))
        image = transforms.to_tensor(image)

        image_mod = np.stack((image_mod, image_mod), axis=-1)
        mask = mask.expand(kspace.shape)
        mask = mask.transpose(0, 2).transpose(1, 2)
        mask = transforms.ifftshift(mask)
        masked_kspace = masked_kspace.transpose(0, 2).transpose(1, 2)

        # mean = np.stack((mean_r, mean_i), axis=0)
        # std = np.stack((std_r, std_i), axis=0)

        return image, mask, masked_kspace, image_mod, fname, slice
        # return image, mean, std, fname, slice


def create_data_loaders(args):
    mask_func = None
    if args.mask_kspace:
        mask_func = MaskFunc(args.center_fractions, args.accelerations)
    data = SliceData(
        root=args.data_path / f'{args.data_split}',
        transform=DataTransform(args.resolution, args.challenge, mask_func),
        sample_rate=1.,
        challenge=args.challenge
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = RecNet().to(args.device)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, mask, kspace, image_mod, fnames, slices) in data_loader:
        # for (input, mean, std, fnames, slices) in data_loader:
            input = input.to(args.device)
            mask = mask.to(args.device)
            kspace = kspace.to(args.device)
            image_mod = image_mod.unsqueeze(2).unsqueeze(3).to(args.device)

            recons = model(input, kspace, mask, image_mod).to('cpu')
            image_mod = image_mod.to('cpu')
            # mean = mean.unsqueeze(-1).unsqueeze(-1).to('cpu')
            # std = std.unsqueeze(-1).unsqueeze(-1).to('cpu')

            for i in range(recons.shape[0]):
                recons[i] = recons[i] * image_mod[i] / 6
                # recons[i] = recons[i] * std[i] + mean[i]
                recons_i = recons[i].numpy()
                recons_i = np.sqrt((recons_i ** 2).sum(axis=0))
                reconstructions[fnames[i]].append((slices[i].numpy(), recons_i))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = Args()
    parser.add_argument('--mask-kspace', action='store_true',
                        help='Whether to apply a mask (set to True for val data and False '
                             'for test data')
    parser.add_argument('--data-split', choices=['val', 'test'], required=True,
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
