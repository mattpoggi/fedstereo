import numpy as np
import torch

import os
import math
import cv2
import tarfile
import random

import webdataset as wds
from PIL import Image
import io

def gt_decoder(value):
    return np.expand_dims(cv2.imdecode(np.fromstring(value, np.uint8), -1), -1).transpose(2, 0, 1).astype(np.float32) / 256.

def proxy_decoder(value):
    return np.expand_dims(cv2.imdecode(np.fromstring(value, np.uint8), -1), -1).transpose(2, 0, 1).astype(np.float32)

def jpg_decoder(value):
    return np.array(Image.open(io.BytesIO(value))).transpose(2, 0, 1).astype(np.float32) / 255.


kitti_raw_lut = {'city': ['2011_09_26_drive_0001_sync',
                        '2011_09_26_drive_0002_sync',
                        '2011_09_26_drive_0005_sync',
                        '2011_09_26_drive_0009_sync',
                        '2011_09_26_drive_0011_sync',
                        '2011_09_26_drive_0013_sync',
                        '2011_09_26_drive_0014_sync',
                        '2011_09_26_drive_0017_sync',
                        '2011_09_26_drive_0018_sync',
                        '2011_09_26_drive_0048_sync',
                        '2011_09_26_drive_0051_sync',
                        '2011_09_26_drive_0056_sync',
                        '2011_09_26_drive_0057_sync',
                        '2011_09_26_drive_0059_sync',
                        '2011_09_26_drive_0060_sync',
                        '2011_09_26_drive_0084_sync',
                        '2011_09_26_drive_0091_sync',
                        '2011_09_26_drive_0093_sync',
                        '2011_09_26_drive_0095_sync',
                        '2011_09_26_drive_0096_sync',
                        '2011_09_26_drive_0104_sync',
                        '2011_09_26_drive_0106_sync',
                        '2011_09_26_drive_0113_sync',
                        '2011_09_26_drive_0117_sync',
                        '2011_09_28_drive_0001_sync',
                        '2011_09_28_drive_0002_sync',
                        '2011_09_29_drive_0026_sync',
                        '2011_09_29_drive_0071_sync'], 
                'residential': ['2011_09_26_drive_0019_sync',
                        '2011_09_26_drive_0020_sync',
                        '2011_09_26_drive_0022_sync',
                        '2011_09_26_drive_0023_sync',
                        '2011_09_26_drive_0035_sync',
                        '2011_09_26_drive_0036_sync',
                        '2011_09_26_drive_0039_sync',
                        '2011_09_26_drive_0046_sync',
                        '2011_09_26_drive_0061_sync',
                        '2011_09_26_drive_0064_sync',
                        '2011_09_26_drive_0079_sync',
                        '2011_09_26_drive_0086_sync',
                        '2011_09_26_drive_0087_sync',
                        '2011_09_30_drive_0018_sync',
                        '2011_09_30_drive_0020_sync',
                        '2011_09_30_drive_0027_sync',
                        '2011_09_30_drive_0028_sync',
                        '2011_09_30_drive_0033_sync',
                        '2011_09_30_drive_0034_sync',
                        '2011_10_03_drive_0027_sync',
                        '2011_10_03_drive_0034_sync'], 
                'campus2': ['2011_09_28_drive_0016_sync',
                        '2011_09_28_drive_0021_sync',
                        '2011_09_28_drive_0034_sync',
                        '2011_09_28_drive_0035_sync',
                        '2011_09_28_drive_0037_sync',
                        '2011_09_28_drive_0038_sync',
                        '2011_09_28_drive_0039_sync',
                        '2011_09_28_drive_0043_sync',
                        '2011_09_28_drive_0045_sync',
                        '2011_09_28_drive_0047_sync',
                        # second round
                        '2011_09_28_drive_0016_sync',
                        '2011_09_28_drive_0021_sync',
                        '2011_09_28_drive_0034_sync',
                        '2011_09_28_drive_0035_sync',
                        '2011_09_28_drive_0037_sync',
                        '2011_09_28_drive_0038_sync',
                        '2011_09_28_drive_0039_sync',
                        '2011_09_28_drive_0043_sync',
                        '2011_09_28_drive_0045_sync',
                        '2011_09_28_drive_0047_sync'],
                'road': ['2011_09_26_drive_0015_sync',
                        '2011_09_26_drive_0027_sync',
                        '2011_09_26_drive_0028_sync',
                        '2011_09_26_drive_0029_sync',
                        '2011_09_26_drive_0032_sync',
                        '2011_09_26_drive_0052_sync',
                        '2011_09_26_drive_0070_sync',
                        '2011_09_26_drive_0101_sync',
                        '2011_09_29_drive_0004_sync',
                        '2011_09_30_drive_0016_sync',
                        '2011_10_03_drive_0042_sync',
                        '2011_10_03_drive_0047_sync']}


drivingstereo_lut = {'rainy': ['2018-08-17-09-45'], 
                    'dusky': ['2018-10-11-17-08'],
                    'cloudy': ['2018-10-15-11-43'], 
                    'rainy2': ['2018-10-17-14-35'],
                    'rainy3': ['2018-10-22-10-44'],
                    'rainy4': ['2018-10-25-07-37'],
                    'dusky2': ['2018-10-16-07-40',
                            '2018-10-16-11-13',
                            '2018-10-16-11-43',
                            '2018-10-24-11-01'],
                    'cloudy2': ['2018-10-17-14-35',
                            '2018-10-17-15-38',
                            '2018-10-18-10-39',
                            '2018-10-18-15-04',
                            '2018-10-19-10-33']
                    }
dsec_lut = {'night': ['zurich_city_03_a'],
            'night2': ['zurich_city_09_a'],
            'night3': ['zurich_city_10_a'],
            'night4': ['zurich_city_10_b'],

            'night5': ['zurich_city_09_b'],
            'night6': ['zurich_city_09_c'],
            'night7': ['zurich_city_09_d'],
            'night8': ['zurich_city_09_e'],
                    }

def fetch_single_dataloader(dataset,datapath,domain,subs,proxy16=False):

    if dataset == 'kitti_raw':
        lut = kitti_raw_lut
        lut['all'] = []
        for key in ['city', 'residential', 'campus2', 'road']:
                lut['all'] += kitti_raw_lut[key]
    elif dataset == 'drivingstereo':
        lut = drivingstereo_lut
    elif dataset == 'dsec':
        lut = dsec_lut

    samples = 0
    if subs == -1:
        sequences = [ '%s/%s.tar'%(datapath,s) for s in lut[domain]]
    else:
        sequences = [ '%s/%s.tar'%(datapath,s) for s in random.sample(lut[domain],subs)]   

    dataset = wds.WebDataset(sequences[0]).decode(
        wds.handle_extension("image_02.jpg", jpg_decoder),
        wds.handle_extension("image_03.jpg", jpg_decoder),
        wds.handle_extension("groundtruth.png", gt_decoder),
        wds.handle_extension("proxy.png", proxy_decoder) if not proxy16 else wds.handle_extension("proxy.png", gt_decoder),
        wds.imagehandler("torchrgb"))
    
    with tarfile.open(sequences[0]) as archive:
        samples += (sum(1 for name in archive.getnames() if 'image_02' in name))

    for s in sequences[1:]:
        dataset +=  wds.WebDataset(s).decode(
        wds.handle_extension("image_02.jpg", jpg_decoder),
        wds.handle_extension("image_03.jpg", jpg_decoder),
        wds.handle_extension("groundtruth.png", gt_decoder),
        wds.handle_extension("proxy.png", proxy_decoder) if not proxy16 else wds.handle_extension("proxy.png", gt_decoder),
        wds.imagehandler("torchrgb"))

        with tarfile.open(s) as archive:
            samples += (sum(1 for name in archive.getnames() if 'image_02' in name))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, persistent_workers=True,
                    pin_memory=False, shuffle=False, num_workers=1, drop_last=True)
    loader.__len__ = int(samples)
    return loader

