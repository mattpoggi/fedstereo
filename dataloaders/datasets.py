# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import yaml
import cv2
import random

import frame_utils
from augmentor import DisparityAugmentor

class MiddleburyDataset(data.Dataset):
    def __init__(self, datapath, subs=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, '*/im0.png')))
        for i in range(len(image_list)):
            if (not self.is_test and ('ArtL' not in image_list[i] and 'Teddy' not in image_list[i])) or self.is_test:
                self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('im0', 'im1')] ]
                self.extra_info += [ image_list[i].split('/')[-2] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][1])
            data['im3'] = frame_utils.read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'] = np.expand_dims( frame_utils.readPFM(self.image_list[index][0].replace('im0.png','disp0GT.pfm')), -1)
            data['validgt'] = data['gt'] < 5000

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][1])
        data['im3'] = frame_utils.read_gen(self.image_list[index][2])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'] = np.expand_dims( frame_utils.readPFM(self.image_list[index][0].replace('im0.png','disp0GT.pfm')), -1)
        data['validgt'] = data['gt'] < 5000

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)

class KITTIStereoDataset(data.Dataset):
    def __init__(self, datapath, subs=None, proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, 'image_2/*_10.png')))
        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('image_2', 'image_3')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][1])
            data['im3'] = frame_utils.read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('image_2','disp_occ_0'))

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][1])
        data['im3'] = frame_utils.read_gen(self.image_list[index][2])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('image_2','disp_occ_0'))

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)

class FlyingThingsDataset(data.Dataset):
    def __init__(self, datapath, subs=None, proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, 'image_clean/left/*.png')))
        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('left', 'right')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][1])
            data['im3'] = frame_utils.read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            data['gt'] = -frame_utils.readPFM(self.image_list[index][0].replace('image_clean','disparity').replace('png','pfm'))
            data['validgt'] = (data['gt']>0)

            data['dxy'] = np.load(self.image_list[index][0].replace('image_clean','slant_window').replace('png','npy'))

            data['gt'] = np.expand_dims( np.array(data['gt']).astype(np.float32), -1)
            data['validgt'] = np.expand_dims( np.array(data['validgt']).astype(np.float32), -1)

            data['dxy'] = np.expand_dims( np.array(data['dxy']).astype(np.float32), -1)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['dxy'] = torch.from_numpy(data['dxy']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][1])
        data['im3'] = frame_utils.read_gen(self.image_list[index][2])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        data['gt'] = -frame_utils.readPFM(self.image_list[index][0].replace('image_clean','disparity').replace('png','pfm'))
        data['validgt'] = (data['gt']>0)

        data['dxy'] = np.load(self.image_list[index][0].replace('image_clean','slant_window').replace('png','npy'))

        data['gt'] = np.expand_dims( np.array(data['gt']).astype(np.float32), -1)
        data['validgt'] = np.expand_dims( np.array(data['validgt']).astype(np.float32), -1)

        #data['dxy'] = np.expand_dims( np.array(data['dxy']).astype(np.float32), -1)
        data['dxy'] = np.transpose(data['dxy'], (1,2,0))

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'], data['dxy'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)


class HuaweiDataset(data.Dataset):
    def __init__(self, datapath, subs=None, aug_params=None, test=False, overfit=False, usemask=False, segment=-1, scale=1):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.scale = scale
        self.usemask = usemask
        self.segment = segment

        # Glue code between persefone data and my shitty format
        scene_list = sorted(glob(osp.join(datapath)+'/*'))
        for i in scene_list:
#            if 'Pillow' not in i:
#                continue
            ill = sorted(glob('%s/camera_00/*png'%i))
            for j in ill:
#                if 'im4' not in j:
#                    continue
                im = j.split('/')[-1].split('.')[0]
                left_img = "%s/camera_00/%s.png"%(i,im)
                right_img = left_img.replace('camera_00','camera_02')
                disp_npy = "%s/disp_00.npy"%(i)
                reliable = "%s/mask_00.png"%(i)
                self.image_list += [ [left_img, right_img, disp_npy, reliable] ]
                self.extra_info += [ i.split('/')[-1]+'_%s'%im ] # scene and frame_id

        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][0])
            data['im3'] = frame_utils.read_gen(self.image_list[index][1])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'] = np.expand_dims( np.load(self.image_list[index][2]), -1)
            data['validgt'] = data['gt'] > 0

            if self.usemask:
                mask = np.expand_dims(cv2.imread(self.image_list[index][3], -1), -1) > 0
                data['validgt'] *= mask
            
            if self.segment > -1:
                mask = np.expand_dims(cv2.imread(self.image_list[index][3].replace('mask_00','mask_cat'), -1), -1) == self.segment
                data['validgt'] *= mask

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][0])
        data['im3'] = frame_utils.read_gen(self.image_list[index][1])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'] = np.expand_dims( np.load(self.image_list[index][2]), -1)
        data['validgt'] = data['gt'] > 0

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        if self.scale != 1:
            h, w = data['im2'].shape[0]//self.scale, data['im2'].shape[1]//self.scale
            data['im2'] = cv2.resize(data['im2'], (w, h), interpolation=cv2.INTER_NEAREST)
            data['im3'] = cv2.resize(data['im3'], (w, h), interpolation=cv2.INTER_NEAREST)

            data['gt'] = np.expand_dims( cv2.resize(data['gt'], (w, h), interpolation=cv2.INTER_NEAREST) / self.scale, -1)
            data['validgt'] = np.expand_dims( cv2.resize(data['validgt'], (w, h), interpolation=cv2.INTER_NEAREST), -1)


        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)









class HuaweiUnbalancedDataset(data.Dataset):
    def __init__(self, datapath, subs=None, aug_params=None, test=False, overfit=False, segment=-1, usemask=False, scale=1):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.scale = scale
        self.usemask = usemask
        self.segment = segment

        # Glue code between persefone data and my shitty format
        scene_list = sorted(glob(osp.join(datapath)+'/*'))
        for i in scene_list:
#            if 'Pillow' not in i:
#                continue
            ill = sorted(glob('%s/camera_00/*png'%i))
            for j in ill:
#                if 'im4' not in j:
#                    continue
                im = j.split('/')[-1].split('.')[0]
                left_img = "%s/camera_00/%s.png"%(i,im)
                right_img = left_img.replace('camera_00','camera_01')
                disp_npy = "%s/disp_00.npy"%(i)
                reliable = "%s/mask_00.png"%(i)
                self.image_list += [ [left_img, right_img, disp_npy, reliable] ]
                self.extra_info += [ i.split('/')[-1]+'_%s'%im ] # scene and frame_id

        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][0])
            data['im3'] = frame_utils.read_gen(self.image_list[index][1])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            data['im2'] = cv2.resize(data['im2'], (data['im3'].shape[1], data['im3'].shape[0]))

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'] = np.expand_dims( np.load(self.image_list[index][2]), -1)
            data['validgt'] = data['gt'] > 0

            if self.usemask:
                mask = np.expand_dims(cv2.imread(self.image_list[index][3], -1), -1) > 0
                data['validgt'] *= mask

            if self.segment > -1:
                mask = np.expand_dims(cv2.imread(self.image_list[index][3].replace('mask_00','warped_mask_cat'), -1), -1) == self.segment
                data['validgt'] *= mask


            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][0])
        data['im3'] = frame_utils.read_gen(self.image_list[index][1])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['im3'] = cv2.resize(data['im3'], (data['im2'].shape[1], data['im2'].shape[0]))

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'] = np.expand_dims( np.load(self.image_list[index][2]), -1)
        data['validgt'] = data['gt'] > 0

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        if self.scale != 1:
            h, w = data['im2'].shape[0]//self.scale, data['im2'].shape[1]//self.scale
            data['im2'] = cv2.resize(data['im2'], (w, h), interpolation=cv2.INTER_NEAREST)
            data['im3'] = cv2.resize(data['im3'], (w, h), interpolation=cv2.INTER_NEAREST)

            data['gt'] = np.expand_dims( cv2.resize(data['gt'], (w, h), interpolation=cv2.INTER_NEAREST) / self.scale, -1)
            data['validgt'] = np.expand_dims( cv2.resize(data['validgt'], (w, h), interpolation=cv2.INTER_NEAREST), -1)


        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)














class MSDataset(data.Dataset):
    def __init__(self, datapath, subs=None, aug_params=None, test=False, overfit=False, segment=-1, usemask=False, scale=1, color_ch=-1, ms_ch=-1):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.scale = scale
        self.usemask = usemask
        self.segment = segment

        # Glue code between persefone data and my shitty format
        scene_list = open('/home/eve/Projects/ProjectH/RGB-MS/split/huawei_offline/validation_all-bands.txt').readlines()
        for i in scene_list:
            im = i.split(' ')[0]
            left_img = '/media/data4/Huawei/rgb-ms_pp/'+i.split(' ')[0]
            right_img = '/media/data4/Huawei/rgb-ms_pp/'+i.split(' ')[1]
            disp_npy = '/media/data4/Huawei/rgb-ms_pp/'+i.split(' ')[2]
            reliable = '/media/data4/Huawei/rgb-ms_pp/'+i.split(' ')[2]
            self.image_list += [ [left_img, right_img, disp_npy, reliable] ]
            self.extra_info += [ i.split('/')[-1]+'_%s'%im ] # scene and frame_id

        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index, color=-1, ms=-1):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][0])
            data['im3'] = np.load(self.image_list[index][1])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = ( data['im3'] - data['im3'].min() ) / (data['im3'].max()-data['im3'].min())

            data['im2'] = cv2.resize(data['im2'], (data['im3'].shape[1], data['im3'].shape[0]))

            if color == -1:
                data['im2'] = data['im2'].mean(-1)
            else:
                data['im2'] = data['im2'][Ellipsis,color]

            if color == -1:
                data['im3'] = data['im3'].mean(-1)
            else:
                data['im3'] = data['im3'][Ellipsis,ms]

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'] = np.expand_dims( np.load(self.image_list[index][2]), -1)
            data['validgt'] = data['gt'] > 0

            if self.usemask:
                mask = np.expand_dims(cv2.imread(self.image_list[index][3], -1), -1) > 0
                data['validgt'] *= mask

            if self.segment > -1:
                mask = np.expand_dims(cv2.imread(self.image_list[index][3].replace('mask_00','warped_mask_cat'), -1), -1) == self.segment
                data['validgt'] *= mask


            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][0])
        data['im3'] = frame_utils.read_gen(self.image_list[index][1])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['im3'] = cv2.resize(data['im3'], (data['im2'].shape[1], data['im2'].shape[0]))

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'] = np.expand_dims( np.load(self.image_list[index][2]), -1)
        data['validgt'] = data['gt'] > 0

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        if self.scale != 1:
            h, w = data['im2'].shape[0]//self.scale, data['im2'].shape[1]//self.scale
            data['im2'] = cv2.resize(data['im2'], (w, h), interpolation=cv2.INTER_NEAREST)
            data['im3'] = cv2.resize(data['im3'], (w, h), interpolation=cv2.INTER_NEAREST)

            data['gt'] = np.expand_dims( cv2.resize(data['gt'], (w, h), interpolation=cv2.INTER_NEAREST) / self.scale, -1)
            data['validgt'] = np.expand_dims( cv2.resize(data['validgt'], (w, h), interpolation=cv2.INTER_NEAREST), -1)


        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)


class KITTIGodardDataset(data.Dataset):
    def __init__(self, datapath, subs="all", proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.sequences = {'city': ['2011_09_26/2011_09_26_drive_0001_sync',
                                '2011_09_26/2011_09_26_drive_0002_sync',
                                '2011_09_26/2011_09_26_drive_0005_sync',
                                '2011_09_26/2011_09_26_drive_0009_sync',
                                '2011_09_26/2011_09_26_drive_0011_sync',
                                '2011_09_26/2011_09_26_drive_0013_sync',
                                '2011_09_26/2011_09_26_drive_0014_sync',
                                '2011_09_26/2011_09_26_drive_0017_sync',
                                '2011_09_26/2011_09_26_drive_0018_sync',
                                '2011_09_26/2011_09_26_drive_0048_sync',
                                '2011_09_26/2011_09_26_drive_0051_sync',
                                '2011_09_26/2011_09_26_drive_0056_sync',
                                '2011_09_26/2011_09_26_drive_0057_sync',
                                '2011_09_26/2011_09_26_drive_0059_sync',
                                '2011_09_26/2011_09_26_drive_0060_sync',
                                '2011_09_26/2011_09_26_drive_0084_sync',
                                '2011_09_26/2011_09_26_drive_0091_sync',
                                '2011_09_26/2011_09_26_drive_0093_sync',
                                '2011_09_26/2011_09_26_drive_0095_sync',
                                '2011_09_26/2011_09_26_drive_0096_sync',
                                '2011_09_26/2011_09_26_drive_0104_sync',
                                '2011_09_26/2011_09_26_drive_0106_sync',
                                '2011_09_26/2011_09_26_drive_0113_sync',
                                '2011_09_26/2011_09_26_drive_0117_sync',
                                '2011_09_28/2011_09_28_drive_0001_sync',
                                '2011_09_28/2011_09_28_drive_0002_sync',
                                '2011_09_29/2011_09_29_drive_0026_sync',
                                '2011_09_29/2011_09_29_drive_0071_sync'], 
                        'residential': ['2011_09_26/2011_09_26_drive_0019_sync',
                                '2011_09_26/2011_09_26_drive_0020_sync',
                                '2011_09_26/2011_09_26_drive_0022_sync',
                                '2011_09_26/2011_09_26_drive_0023_sync',
                                '2011_09_26/2011_09_26_drive_0035_sync',
                                '2011_09_26/2011_09_26_drive_0036_sync',
                                '2011_09_26/2011_09_26_drive_0039_sync',
                                '2011_09_26/2011_09_26_drive_0046_sync',
                                '2011_09_26/2011_09_26_drive_0061_sync',
                                '2011_09_26/2011_09_26_drive_0064_sync',
                                '2011_09_26/2011_09_26_drive_0079_sync',
                                '2011_09_26/2011_09_26_drive_0086_sync',
                                '2011_09_26/2011_09_26_drive_0087_sync',
                                '2011_09_30/2011_09_30_drive_0018_sync',
                                '2011_09_30/2011_09_30_drive_0020_sync',
                                '2011_09_30/2011_09_30_drive_0027_sync',
                                '2011_09_30/2011_09_30_drive_0028_sync',
                                '2011_09_30/2011_09_30_drive_0033_sync',
                                '2011_09_30/2011_09_30_drive_0034_sync',
                                '2011_10_03/2011_10_03_drive_0027_sync',
                                '2011_10_03/2011_10_03_drive_0034_sync'], 
                        'campus': ['2011_09_28/2011_09_28_drive_0016_sync',
                                '2011_09_28/2011_09_28_drive_0021_sync',
                                '2011_09_28/2011_09_28_drive_0034_sync',
                                '2011_09_28/2011_09_28_drive_0035_sync',
                                '2011_09_28/2011_09_28_drive_0037_sync',
                                '2011_09_28/2011_09_28_drive_0038_sync',
                                '2011_09_28/2011_09_28_drive_0039_sync',
                                '2011_09_28/2011_09_28_drive_0043_sync',
                                '2011_09_28/2011_09_28_drive_0045_sync',
                                '2011_09_28/2011_09_28_drive_0047_sync'], 
                        'road': ['2011_09_26/2011_09_26_drive_0015_sync',
                                '2011_09_26/2011_09_26_drive_0027_sync',
                                '2011_09_26/2011_09_26_drive_0028_sync',
                                '2011_09_26/2011_09_26_drive_0029_sync',
                                '2011_09_26/2011_09_26_drive_0032_sync',
                                '2011_09_26/2011_09_26_drive_0052_sync',
                                '2011_09_26/2011_09_26_drive_0070_sync',
                                '2011_09_26/2011_09_26_drive_0101_sync',
                                '2011_09_29/2011_09_29_drive_0004_sync',
                                '2011_09_30/2011_09_30_drive_0016_sync',
                                '2011_10_03/2011_10_03_drive_0042_sync',
                                '2011_10_03/2011_10_03_drive_0047_sync']}

        if subs not in ['all', 'city', 'residential', 'road', 'campus']:
            self.sequences[subs] = [subs]

        # Glue code between persefone data and my shitty format        
        image_list = []
        samples = []
        if subs == 'all':
            for key in self.sequences:
                samples += self.sequences[key]
        else:
            samples = self.sequences[subs]

        #sorted(glob(osp.join(datapath, 'image_2/*_10.png')))
        for seq in samples:
            image_list += sorted(glob(osp.join(datapath, seq, 'image_02/data/*.jpg')))

        for i in range(len(image_list)):
            self.image_list += [ [image_list[i].replace('image_02/data', 'image_02/data').replace('png', 'jpg'), 
                                image_list[i].replace('image_02/data', 'image_02/data').replace('png', 'jpg'), 
                                image_list[i].replace('image_02/data', 'image_03/data').replace('png', 'jpg')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
          
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][1])
            data['im3'] = frame_utils.read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            """
            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]            
            data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('image_02/data','proj_disp/groundtruth/image_02').replace('.jpg','.png'))

            data['pr'], data['validpr'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('KITTI_RTSA','KITTI_RTSA/sgm').replace('.jpg','.png'))
            if data['pr'].shape[1] != data['im2'].shape[1]:
                data['pr'] = data['pr'][:,:-(data['pr'].shape[1] - data['im2'].shape[1])]
                data['validpr'] = data['validpr'][:,:-(data['validpr'].shape[1] - data['im2'].shape[1])]

            if data['pr'].max() < 1:
                data['pr'] *= 256.

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()


            data['pr'] = np.array(data['pr']).astype(np.float32)
            data['validpr'] = np.array(data['validpr']).astype(np.float32)
            
            data['pr'] = torch.from_numpy(data['pr']).permute(2, 0, 1).float()
            data['validpr'] = torch.from_numpy(data['validpr']).permute(2, 0, 1).float()
            """


            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

    def __len__(self):
        return len(self.image_list)




class KITTIRawDataset(data.Dataset):
    def __init__(self, datapath, subs="all", mode='sequence', nr_samples=1, proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.datapath = datapath
        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.sequences = {'city': ['2011_09_26/2011_09_26_drive_0001_sync',
                                '2011_09_26/2011_09_26_drive_0002_sync',
                                '2011_09_26/2011_09_26_drive_0005_sync',
                                '2011_09_26/2011_09_26_drive_0009_sync',
                                '2011_09_26/2011_09_26_drive_0011_sync',
                                '2011_09_26/2011_09_26_drive_0013_sync',
                                '2011_09_26/2011_09_26_drive_0014_sync',
                                '2011_09_26/2011_09_26_drive_0017_sync',
                                '2011_09_26/2011_09_26_drive_0018_sync',
                                '2011_09_26/2011_09_26_drive_0048_sync',
                                '2011_09_26/2011_09_26_drive_0051_sync',
                                '2011_09_26/2011_09_26_drive_0056_sync',
                                '2011_09_26/2011_09_26_drive_0057_sync',
                                '2011_09_26/2011_09_26_drive_0059_sync',
                                '2011_09_26/2011_09_26_drive_0060_sync',
                                '2011_09_26/2011_09_26_drive_0084_sync',
                                '2011_09_26/2011_09_26_drive_0091_sync',
                                '2011_09_26/2011_09_26_drive_0093_sync',
                                '2011_09_26/2011_09_26_drive_0095_sync',
                                '2011_09_26/2011_09_26_drive_0096_sync',
                                '2011_09_26/2011_09_26_drive_0104_sync',
                                '2011_09_26/2011_09_26_drive_0106_sync',
                                '2011_09_26/2011_09_26_drive_0113_sync',
                                '2011_09_26/2011_09_26_drive_0117_sync',
                                '2011_09_28/2011_09_28_drive_0001_sync',
                                '2011_09_28/2011_09_28_drive_0002_sync',
                                '2011_09_29/2011_09_29_drive_0026_sync',
                                '2011_09_29/2011_09_29_drive_0071_sync'], 
                        'residential': ['2011_09_26/2011_09_26_drive_0019_sync',
                                '2011_09_26/2011_09_26_drive_0020_sync',
                                '2011_09_26/2011_09_26_drive_0022_sync',
                                '2011_09_26/2011_09_26_drive_0023_sync',
                                '2011_09_26/2011_09_26_drive_0035_sync',
                                '2011_09_26/2011_09_26_drive_0036_sync',
                                '2011_09_26/2011_09_26_drive_0039_sync',
                                '2011_09_26/2011_09_26_drive_0046_sync',
                                '2011_09_26/2011_09_26_drive_0061_sync',
                                '2011_09_26/2011_09_26_drive_0064_sync',
                                '2011_09_26/2011_09_26_drive_0079_sync',
                                '2011_09_26/2011_09_26_drive_0086_sync',
                                '2011_09_26/2011_09_26_drive_0087_sync',
                                '2011_09_30/2011_09_30_drive_0018_sync',
                                '2011_09_30/2011_09_30_drive_0020_sync',
                                '2011_09_30/2011_09_30_drive_0027_sync',
                                '2011_09_30/2011_09_30_drive_0028_sync',
                                '2011_09_30/2011_09_30_drive_0033_sync',
                                '2011_09_30/2011_09_30_drive_0034_sync',
                                '2011_10_03/2011_10_03_drive_0027_sync',
                                '2011_10_03/2011_10_03_drive_0034_sync'], 
                        'campus': ['2011_09_28/2011_09_28_drive_0016_sync',
                                '2011_09_28/2011_09_28_drive_0021_sync',
                                '2011_09_28/2011_09_28_drive_0034_sync',
                                '2011_09_28/2011_09_28_drive_0035_sync',
                                '2011_09_28/2011_09_28_drive_0037_sync',
                                '2011_09_28/2011_09_28_drive_0038_sync',
                                '2011_09_28/2011_09_28_drive_0039_sync',
                                '2011_09_28/2011_09_28_drive_0043_sync',
                                '2011_09_28/2011_09_28_drive_0045_sync',
                                '2011_09_28/2011_09_28_drive_0047_sync'],
                        'campus2': ['2011_09_28/2011_09_28_drive_0016_sync',
                                '2011_09_28/2011_09_28_drive_0021_sync',
                                '2011_09_28/2011_09_28_drive_0034_sync',
                                '2011_09_28/2011_09_28_drive_0035_sync',
                                '2011_09_28/2011_09_28_drive_0037_sync',
                                '2011_09_28/2011_09_28_drive_0038_sync',
                                '2011_09_28/2011_09_28_drive_0039_sync',
                                '2011_09_28/2011_09_28_drive_0043_sync',
                                '2011_09_28/2011_09_28_drive_0045_sync',
                                '2011_09_28/2011_09_28_drive_0047_sync',
                                '2011_09_28/2011_09_28_drive_0016_sync',
                                '2011_09_28/2011_09_28_drive_0021_sync',
                                '2011_09_28/2011_09_28_drive_0034_sync',
                                '2011_09_28/2011_09_28_drive_0035_sync',
                                '2011_09_28/2011_09_28_drive_0037_sync',
                                '2011_09_28/2011_09_28_drive_0038_sync',
                                '2011_09_28/2011_09_28_drive_0039_sync',
                                '2011_09_28/2011_09_28_drive_0043_sync',
                                '2011_09_28/2011_09_28_drive_0045_sync',
                                '2011_09_28/2011_09_28_drive_0047_sync'], 
                        'road': ['2011_09_26/2011_09_26_drive_0015_sync',
                                '2011_09_26/2011_09_26_drive_0027_sync',
                                '2011_09_26/2011_09_26_drive_0028_sync',
                                '2011_09_26/2011_09_26_drive_0029_sync',
                                '2011_09_26/2011_09_26_drive_0032_sync',
                                '2011_09_26/2011_09_26_drive_0052_sync',
                                '2011_09_26/2011_09_26_drive_0070_sync',
                                '2011_09_26/2011_09_26_drive_0101_sync',
                                '2011_09_29/2011_09_29_drive_0004_sync',
                                '2011_09_30/2011_09_30_drive_0016_sync',
                                '2011_10_03/2011_10_03_drive_0042_sync',
                                '2011_10_03/2011_10_03_drive_0047_sync']}

        """
        if subs not in ['all', 'city', 'residential', 'road', 'campus']:
            #self.sequences[subs] = [subs]
            name, ids, ide = subs.split('_')
            ids = int(ids)
            ide = int(ide)
            if name == 'all':
                if ids != 0 and ide != 0:
                    samples = self.sequences['city'][ids:ide]+self.sequences['residential'][ids:ide]+self.sequences['road'][ids:ide]+self.sequences['campus'][ids:ide]
                else:
                    samples = self.sequences['city']+self.sequences['residential']+self.sequences['road']+self.sequences['campus']
            else:
                samples = self.sequences[name]
                if ids != 0 and ide != 0:
                    samples = samples[ids:ide]
        else:
            self.sequences[subs] = [subs]
        """

        # Glue code between persefone data and my shitty format        
        image_list = []
        samples = []
        if subs == 'all':
            for key in self.sequences:
                samples += self.sequences[key]
        else:
            samples = self.sequences[subs]

        if mode == 'random_sample':
            samples = random.sample(samples,nr_samples)

        #sorted(glob(osp.join(datapath, 'image_2/*_10.png')))
        for seq in samples:
            image_list += sorted(glob(osp.join(datapath, seq, 'proj_disp/groundtruth/image_02/*.png')))

        for i in range(len(image_list)):
            self.image_list += [ [image_list[i].replace('proj_disp/groundtruth/image_02', 'image_02/data').replace('png', 'jpg'), 
                                image_list[i].replace('proj_disp/groundtruth/image_02', 'image_02/data').replace('png', 'jpg'), 
                                image_list[i].replace('proj_disp/groundtruth/image_02', 'image_03/data').replace('png', 'jpg')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
          
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][1])
            data['im3'] = frame_utils.read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]            
            data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('image_02/data','proj_disp/groundtruth/image_02').replace('.jpg','.png'))

            data['pr'], data['validpr'] = frame_utils.readDispKITTI(self.image_list[index][0].replace(self.datapath,self.datapath+'/sgm/').replace('.jpg','.png'))
            if data['pr'].shape[1] != data['im2'].shape[1]:
                data['pr'] = data['pr'][:,:-(data['pr'].shape[1] - data['im2'].shape[1])]
                data['validpr'] = data['validpr'][:,:-(data['validpr'].shape[1] - data['im2'].shape[1])]

            if data['pr'].max() < 1:
                data['pr'] *= 256.

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()


            data['pr'] = np.array(data['pr']).astype(np.float32)
            data['validpr'] = np.array(data['validpr']).astype(np.float32)
            
            data['pr'] = torch.from_numpy(data['pr']).permute(2, 0, 1).float()
            data['validpr'] = torch.from_numpy(data['validpr']).permute(2, 0, 1).float()



            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][1])
        data['im3'] = frame_utils.read_gen(self.image_list[index][2])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('image_2','disp_occ_0'))

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)


class DSECDataset(data.Dataset):
    def __init__(self, datapath, subs="all", proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.sequences = {'day': ['zurich_city_00_a',
                                    'zurich_city_00_b',
                                    'zurich_city_01_a',
                                    'zurich_city_01_b',
                                    'zurich_city_01_c',
                                    'zurich_city_01_d',
                                    ], 
                        'night': ['zurich_city_03_a'],
                        'night2': ['zurich_city_09_a'],
                        'night3': ['zurich_city_10_a'],
                        'night4': ['zurich_city_10_b'],

                        'night5': ['zurich_city_09_b'],
                        'night6': ['zurich_city_09_c'],
                        'night7': ['zurich_city_09_d'],
                        'night8': ['zurich_city_09_e'],
                        }

        # Glue code between persefone data and my shitty format        
        image_list = []
        samples = []
        # if subs == 'all':
        #     for key in self.sequences:
        #         samples = self.sequences[key]
        # else:
        #     name, ids, ide = subs.split('_')
        #     ids = int(ids)
        #     ide = int(ide)
        #     if ids != 0 or ide != 0:
        #         samples = [f for f in self.sequences[name][ids:ide]]

        if subs == 'all':
            for key in self.sequences:
                samples += self.sequences[key]
        else:
            samples = self.sequences[subs]
            
        #sorted(glob(osp.join(datapath, 'image_2/*_10.png')))
        for seq in samples:
            image_list += sorted(glob(osp.join(datapath, seq, 'sgm/images/*.png')))

        for i in range(len(image_list)):
            self.image_list += [ [image_list[i].replace('sgm/images/', 'images/left/rectified/'), 
                                image_list[i].replace('sgm/images/', 'images/left/rectified/'), 
                                image_list[i].replace('sgm/images/', 'images/right/rectified/')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
          
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][1])
            data['im3'] = frame_utils.read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

#            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1] 
            if os.path.exists(self.image_list[index][0].replace('images/left/rectified/','disparity/image/').replace('.jpg','.png')):  
                data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('images/left/rectified/','disparity/image/').replace('.jpg','.png'))

                data['gt'] = np.array(data['gt']).astype(np.float32)
                data['validgt'] = np.array(data['validgt']).astype(np.float32)
                
                data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()[:,384:-256,256:]
                data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()[:,384:-256,256:]

            data['pr'], data['validpr'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('images/left/rectified/','sgm/images/').replace('.jpg','.png'))
            if data['pr'].shape[1] != data['im2'].shape[1]:
                data['pr'] = data['pr'][:,:-(data['pr'].shape[1] - data['im2'].shape[1])]
                data['validpr'] = data['validpr'][:,:-(data['validpr'].shape[1] - data['im2'].shape[1])]

            if data['pr'].max() < 1:
                data['pr'] *= 256.

            data['pr'] = np.array(data['pr']).astype(np.float32)
            data['validpr'] = np.array(data['validpr']).astype(np.float32)
            
            data['pr'] = torch.from_numpy(data['pr']).permute(2, 0, 1).float()[:,384:-256,256:]
            data['validpr'] = torch.from_numpy(data['validpr']).permute(2, 0, 1).float()[:,384:-256,256:]

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()[:,384:-256,256:]
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()[:,384:-256,256:]
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][1])
        data['im3'] = frame_utils.read_gen(self.image_list[index][2])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('image_2','disp_occ_0'))

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)







class ARGODataset(data.Dataset):
    def __init__(self, datapath, subs="all", proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.sequences = {'day': ['0ef28d5c-ae34-370b-99e7-6709e1c4b929',
                                    #'zurich_city_11_c',
                                    ], 
                        'night0': ['033669d3-3d6b-3d3d-bd93-7985d86653ea'],
                        'night1': ['53037376-5303-5303-5303-553038557184'],
                        'night2': ['5c251c22-11b2-3278-835c-0cf3cdee3f44'],
                        }

        # Glue code between persefone data and my shitty format        
        image_list_left, image_list_right, image_list_gt = [], [], []
        samples = []
        if subs == 'all':
            for key in self.sequences:
                samples += self.sequences[key]
        else:
            samples = self.sequences[subs]

        #sorted(glob(osp.join(datapath, 'image_2/*_10.png')))
        for seq in samples:
            image_list_left += sorted(glob(osp.join(datapath, 'rectified_stereo_images_v1.1/*/', seq, 'stereo_front_left_rect/*.jpg')))
            image_list_right += sorted(glob(osp.join(datapath, 'rectified_stereo_images_v1.1/*/', seq, 'stereo_front_right_rect/*.jpg')))
            image_list_gt += sorted(glob(osp.join(datapath, 'disparity_maps_v1.1/*/', seq, 'stereo_front_left_rect_disparity/*.png')))
            #image_list_pr += sorted(glob(osp.join(datapath, 'rectified_stereo_images_v1.1/*/', seq, 'sgm/*.png')))

        for i in range(len(image_list_left)):
            self.image_list += [ [image_list_left[i], 
                                image_list_right[i], 
                                image_list_gt[i]] ]
            self.extra_info += [ image_list_left[i].split('/')[-1] ] # scene and frame_id
          
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][0])
            data['im3'] = frame_utils.read_gen(self.image_list[index][1])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]            
            data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][2])

            data['pr'], data['validpr'] = frame_utils.readDispKITTI(self.image_list[index][2])
            if data['pr'].shape[1] != data['im2'].shape[1]:
                data['pr'] = data['pr'][:,:-(data['pr'].shape[1] - data['im2'].shape[1])]
                data['validpr'] = data['validpr'][:,:-(data['validpr'].shape[1] - data['im2'].shape[1])]

            if data['pr'].max() < 1:
                data['pr'] *= 256.

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()[:,512:]#-256,256:]
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()[:,512:]#-256,256:]


            data['pr'] = np.array(data['pr']).astype(np.float32)
            data['validpr'] = np.array(data['validpr']).astype(np.float32)
            
            data['pr'] = torch.from_numpy(data['pr']).permute(2, 0, 1).float()[:,512:]#-256,256:]
            data['validpr'] = torch.from_numpy(data['validpr']).permute(2, 0, 1).float()[:,512:]#-256,256:]



            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()[:,512:]#-256,256:]
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()[:,512:]#-256,256:]
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][0])
        data['im3'] = frame_utils.read_gen(self.image_list[index][1])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][2])

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)




class DrivingStereoDataset(data.Dataset):
    def __init__(self, datapath, subs="all", proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.sequences = {
                        'rainy': ['2018-08-17-09-45'],
                        'dusky': ['2018-10-11-17-08'],
                        'cloudy': ['2018-10-15-11-43'],
                        }

        # Glue code between persefone data and my shitty format        
        image_list = []
        samples = []
        if subs == 'all':
            for key in self.sequences:
                samples += self.sequences[key]
        else:
            samples = self.sequences[subs]

        #sorted(glob(osp.join(datapath, 'image_2/*_10.png')))
        for seq in samples:
            image_list += sorted(glob(osp.join(datapath, 'train-left/%s*'%seq)))

        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], 
                                image_list[i].replace('train-left', 'train-right'), 
                                image_list[i].replace('train-left', 'train-disparity')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
          
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][0])
            data['im3'] = frame_utils.read_gen(self.image_list[index][1])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]            
            data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('train-left','train-disparity').replace('jpg','png'))

            data['pr'], data['validpr'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('train-left','train-sgm-left').replace('jpg','png'))
            if data['pr'].shape[1] != data['im2'].shape[1]:
                data['pr'] = data['pr'][:,:-(data['pr'].shape[1] - data['im2'].shape[1])]
                data['validpr'] = data['validpr'][:,:-(data['validpr'].shape[1] - data['im2'].shape[1])]

            if data['pr'].max() < 1:
                data['pr'] *= 256.

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()


            data['pr'] = np.array(data['pr']).astype(np.float32)
            data['validpr'] = np.array(data['validpr']).astype(np.float32)
            
            data['pr'] = torch.from_numpy(data['pr']).permute(2, 0, 1).float()
            data['validpr'] = torch.from_numpy(data['validpr']).permute(2, 0, 1).float()



            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        data['im2'] = frame_utils.read_gen(self.image_list[index][1])
        data['im3'] = frame_utils.read_gen(self.image_list[index][2])

        data['im2'] = np.array(data['im2']).astype(np.uint8)
        data['im3'] = np.array(data['im3']).astype(np.uint8)

        data['gt'], data['validgt'] = [None]*2
        frm = self.image_list[index][0].split('.')[-1]
        data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('image_2','disp_occ_0'))

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.float32)

        # grayscale images
        if len(data['im2'].shape) == 2:
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im2'] = data['im2'][..., :3]                

        if len(data['im3'].shape) == 2:
            data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
        else:
            data['im3'] = data['im3'][..., :3]

        if self.augmentor is not None:
            augm_data = self.augmentor(data['im2'], data['im3'], data['gt'], data['validgt'])
        else:
            augm_data = data

        augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])
        augm_data['im3f'] = np.ascontiguousarray(augm_data['im3'][:,::-1])

        for k in augm_data:
            if augm_data[k] is not None:
                augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
        
        augm_data['extra_info'] = self.extra_info[index]

        return augm_data

    def __len__(self):
        return len(self.image_list)




class RobotCarDataset(data.Dataset):
    def __init__(self, datapath, subs="all", proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []
        self.sequences = {
                        'night0': ['2014-11-14-16-34-33'],
                        'cloudy': ['2018-10-11-17-08'],
                        'country': ['2018-10-15-11-43'],
                        }

        # Glue code between persefone data and my shitty format        
        image_list = []
        samples = []
        if subs == 'all':
            for key in self.sequences:
                samples += self.sequences[key]
        else:
            samples = self.sequences[subs]

        #sorted(glob(osp.join(datapath, 'image_2/*_10.png')))
        for seq in samples:
            image_list += sorted(glob(osp.join(datapath, seq, 'stereo-jpg/left/*')))[:1000]

        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], 
                                image_list[i].replace('stereo-jpg/left/', 'stereo-jpg/right/'), 
                                image_list[i].replace('stereo-jpg/left/', 'sgm')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
          
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][0])
            data['im3'] = frame_utils.read_gen(self.image_list[index][1])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            #data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]  
            """
            data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('train-left','train-disparity').replace('jpg','png'))
            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()
            """

            data['pr'], data['validpr'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('stereo-jpg/left/','sgm/').replace('jpg','png'))
            if data['pr'].shape[1] != data['im2'].shape[1]:
                data['pr'] = data['pr'][:,:-(data['pr'].shape[1] - data['im2'].shape[1])]
                data['validpr'] = data['validpr'][:,:-(data['validpr'].shape[1] - data['im2'].shape[1])]

            if data['pr'].max() < 1:
                data['pr'] *= 256.

            data['pr'] = np.array(data['pr']).astype(np.float32)
            data['validpr'] = np.array(data['validpr']).astype(np.float32)
            
            data['pr'] = torch.from_numpy(data['pr']).permute(2, 0, 1).float()
            data['validpr'] = torch.from_numpy(data['validpr']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data

    def __len__(self):
        return len(self.image_list)















class ZEDDataset(data.Dataset):
    def __init__(self, datapath, subs=None, proxies=None, aug_params=None, sparse=False, test=False, overfit=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            self.augmentor = DisparityAugmentor(**aug_params)

        self.is_test = test
        self.init_seed = False
        self.proxies = proxies
        self.disp_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

        # Glue code between persefone data and my shitty format
        image_list = sorted(glob(osp.join(datapath, 'left/*.jpg')))
        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('left', 'right')] ]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
                    
        if overfit:
            self.image_list = self.image_list[0:1]
            self.extra_info = self.extra_info[0:1]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            data['im2'] = frame_utils.read_gen(self.image_list[index][1])
            data['im3'] = frame_utils.read_gen(self.image_list[index][2])
            
            data['im2'] = np.array(data['im2']).astype(np.uint8)/ 255.0
            data['im3'] = np.array(data['im3']).astype(np.uint8)/ 255.0

            # grayscale images
            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]                

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][...,None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            
            data['pr'], data['validpr'] = frame_utils.readDispKITTI(self.image_list[index][0].replace('left','disp_sgm').replace('.jpg','.png'))
            #if data['pr'].max() < 1:
            #    data['pr'] *= 256.

            data['pr'] = np.array(data['pr']).astype(np.float32)
            data['validpr'] = np.array(data['validpr']).astype(np.float32)
            
            data['pr'] = torch.from_numpy(data['pr']).permute(2, 0, 1).float()
            data['validpr'] = torch.from_numpy(data['validpr']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            data['extra_info'] = self.extra_info[index]
            
            return data


    def __len__(self):
        return len(self.image_list)














def worker_init_fn(worker_id):                                                          
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'flyingthings':
        if args.test:
            dataset = FlyingThingsDataset(args.datapath,args.subs,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = FlyingThingsDataset(args.datapath,None,None,aug_params,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True, worker_init_fn = worker_init_fn)
            print('Training with %d image pairs' % len(dataset))

    elif args.dataset == 'kitti_stereo':
        if args.test:
            dataset = KITTIStereoDataset(args.datapath,args.subs,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = KITTIStereoDataset(args.datapath,args.subs,aug_params=aug_params,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))


    elif args.dataset == 'zed':
        if args.test:
            dataset = ZEDDataset(args.datapath,args.subs,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = ZEDDataset(args.datapath,args.subs,aug_params=aug_params,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))


    elif args.dataset == 'kitti_godard':
        if args.test:
            dataset = KITTIGodardDataset(args.datapath,args.subs,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = KITTIGodardDataset(args.datapath,args.subs,aug_params,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))


    elif args.dataset == 'kitti_raw':
        if args.test:
            dataset = KITTIRawDataset(args.datapath,args.subs,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = KITTIRawDataset(args.datapath,args.subs,aug_params,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))


    elif args.dataset == 'dsec':
        if args.test:
            dataset = DSECDataset(args.datapath,args.subs,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = DSECDataset(args.datapath,args.subs,aug_params,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))

    
    elif args.dataset == 'argo':
        if args.test:
            dataset = ARGODataset(args.datapath,args.subs,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = ARGODataset(args.datapath,args.subs,aug_params,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))



    elif args.dataset == 'middlebury':
        if args.test:
            dataset = MiddleburyDataset(args.datapath,args.subs,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = MiddleburyDataset(args.datapath,args.subs,aug_params,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))

    elif args.dataset == 'huawei':
        if args.test:
            dataset = HuaweiDataset(args.datapath,args.subs,usemask=args.mask,segment=args.segment,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = HuaweiDataset(args.datapath,args.subs,aug_params,overfit=args.overfit,scale=args.scale)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))

    elif args.dataset == 'multispectral':
        if args.test:
            dataset = MSDataset(args.datapath,args.subs,usemask=args.mask,segment=args.segment,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            print('Training not supported, sorry :(')
            # aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            # dataset = HuaweiDataset(args.datapath,args.subs,aug_params,overfit=args.overfit,scale=args.scale)
            # loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
            #     pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            # print('Training with %d image pairs' % len(dataset))


    elif args.dataset == 'unbalanced':
        if args.test:
            dataset = HuaweiUnbalancedDataset(args.datapath,args.subs,usemask=args.mask,segment=args.segment,test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            dataset = HuaweiUnbalancedDataset(args.datapath,args.subs,aug_params,overfit=args.overfit,scale=4)+HuaweiUnbalancedDataset(args.datapath,args.subs,aug_params,overfit=args.overfit,scale=2)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))



    elif args.dataset == 'mixed':
        if args.test:
            print("Mixed dataset not implemented for testing")
            exit()
        else:
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 0.1, 'do_flip': True}
            booster_full = HuaweiDataset(args.datapath.split(' ')[0],args.subs,aug_params,overfit=args.overfit,scale=1)
            booster_half = HuaweiDataset(args.datapath.split(' ')[0],args.subs,aug_params,overfit=args.overfit,scale=2)
            booster_quarter = HuaweiDataset(args.datapath.split(' ')[0],args.subs,aug_params,overfit=args.overfit,scale=4)
            #middlebury_full = MiddleburyDataset(args.datapath.split(' ')[1],args.subs,aug_params,overfit=args.overfit)
            #middlebury_half = MiddleburyDataset(args.datapath.split(' ')[1].replace('F', 'H'),args.subs,aug_params,overfit=args.overfit)
            #middlebury_quarter = MiddleburyDataset(args.datapath.split(' ')[1].replace('F', 'Q'),args.subs,aug_params,overfit=args.overfit)
            booster = booster_quarter + booster_half #+ booster_full
            #middlebury = middlebury_full + middlebury_half #+ middlebury_quarter
            #repeat = len(booster)//len(middlebury)
            dataset = booster
            #for i in range(repeat):
            #    dataset = dataset + middlebury
            loader = data.DataLoader(dataset, batch_size=args.batch_size,
                pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
            print('Training with %d image pairs' % len(dataset))


    return loader



















def fetch_single_dataloader(dataset, datapath, subs):
    """ Create the data loader for the corresponding trainign set """

    if dataset == 'kitti_raw':
        dataset = KITTIRawDataset(datapath,subs,test=True)
        loader = data.DataLoader(dataset, batch_size=1, persistent_workers=True,
            pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
        print('Testing with %d image pairs' % len(dataset))

    elif dataset == 'dsec':
        dataset = DSECDataset(datapath,subs,test=True)
        loader = data.DataLoader(dataset, batch_size=1, persistent_workers=True,
            pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
        print('Testing with %d image pairs' % len(dataset))
    
    elif dataset == 'argo':
        dataset = ARGODataset(datapath,subs,test=True)
        loader = data.DataLoader(dataset, batch_size=1, persistent_workers=True,
            pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
        print('Testing with %d image pairs' % len(dataset))

    elif dataset == 'drivingstereo':
        dataset = DrivingStereoDataset(datapath,subs,test=True)
        loader = data.DataLoader(dataset, batch_size=1, persistent_workers=True,
            pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
        print('Testing with %d image pairs' % len(dataset))

    elif dataset == 'robotcar':
        dataset = RobotCarDataset(datapath,subs,test=True)
        loader = data.DataLoader(dataset, batch_size=1, persistent_workers=True,
            pin_memory=False, shuffle=False, num_workers=8, drop_last=True)
        print('Testing with %d image pairs' % len(dataset))

    
    return loader
