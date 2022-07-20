"""
    GraphCSPN: Geometry-Aware Depth Completion via Dynamic GCNs

    European Conference on Computer Vision (ECCV) 2022

    This part of code is modified based on https://github.com/zzangjinsun/NLSPN_ECCV20
"""


import os
import warnings
import numpy as np
import json
import h5py

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore", category=UserWarning)

"""
NYUDepthV2 json file has a following format:

{
    "train": [
        {
            "filename": "train/bedroom_0078/00066.h5"
        }, ...
    ],
    "val": [
        {
            "filename": "train/study_0008/00351.h5"
        }, ...
    ],
    "test": [
        {
            "filename": "val/official/00001.h5"
        }, ...
    ]
}

Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""

class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)


class NYU(BaseDataset):
    def __init__(self, args, mode):
        super(NYU, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        # For NYUDepthV2, crop size is fixed
        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_file = os.path.join(self.args.dir_data,
                                 self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')

        if self.augment and self.mode == 'train':
            _scale = 1
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor(),
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            dep = dep / _scale

        else:
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

        dep_sp = self.get_sparse_depth(dep, self.args.num_sample)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep}

        return output

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp
