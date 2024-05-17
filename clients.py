import threading
import configparser
from contextlib import nullcontext
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

import copy

from models import *
import sys
sys.path.append('dataloaders')
import tar_datasets as datasets

import tqdm

class StereoClient(threading.Thread):

    def __init__(self, cfg, args, idx, server=None):
        
        threading.Thread.__init__(self)
        config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        config.read(cfg)
        self.idx = idx
        self.gpu = int(config['adaptation']['gpu'])
        self.server = server
        
        self.args=args
        self.adapt_mode = config['adaptation']['adapt_mode']
        self.sample_mode = config['adaptation']['sample_mode']

        self.model = config['network']['model']
        
        self.sender = config['federated'].getboolean('sender')
        self.listener = config['federated'].getboolean('listener')
        self.bootstrap_countdown = 0 if self.server is None else self.server.bootstrap_countdown 
        self.bootstrapped = False if (self.listener and self.server is not None and self.bootstrap_countdown != 0) else True

        self.runs = []
        for i in range(len(config['environment'].getlist('dataset'))):
            dataset = config['environment'].getlist('dataset')[i]
            datapath = config['environment'].getlist('datapath')[i]
            domain = config['environment'].getlist('domain')[i]
            subs = int(config['environment'].getlist('subs')[i])
            proxy16 = config['environment'].getboolean('proxy16')
        
            self.runs.append( {'loader': datasets.fetch_single_dataloader(dataset,datapath,domain,subs,proxy16),
                                'dataset':dataset, 
                                'domain':domain, 
            })

        self.net = models_lut[self.model](args)
        self.net = nn.DataParallel(self.net)
        self.net.to('cuda:%d'%self.gpu)
        self.net.load_state_dict(torch.load(config['network']['checkpoint'], torch.device('cuda:%d'%self.gpu))['state_dict'])
        self.net = self.net.module

        self.optimizer = optim.Adam(self.net.parameters(), lr=float(config['adaptation']['lr']), betas=(0.9, 0.999))

        self.current_run = self.runs[0]
        self.loader = self.current_run['loader']
        self.accumulator = {}

    def run(self):

        args=self.args

        if self.listener or self.args.verbose:
            self.pbar = tqdm.tqdm(total=self.current_run['loader'].__len__, file=sys.stdout)
        
        while not self.bootstrapped:
            time.sleep(0.01)

        self.net.eval()
        with torch.no_grad() if (self.adapt_mode == 'none') else nullcontext():
            
            for batch_idx, data in enumerate(self.loader):

                if self.server is not None:
                    ret = self.server.gpu_locks[self.gpu].acquire()
                    while not ret:
                        time.sleep(0.01)
                        ret = self.server.gpu_locks[self.gpu].acquire()

                if self.adapt_mode != 'none':
                    self.optimizer.zero_grad()
                
                data['image_02.jpg'], data['image_03.jpg'] = data['image_02.jpg'].to('cuda:%d'%self.gpu), data['image_03.jpg'].to('cuda:%d'%self.gpu)
                
                if 'proxy.png' in data:
                    data['validpr'] = (data['proxy.png']>0).float()
                    data['proxy.png'], data['validpr'] = data['proxy.png'].to('cuda:%d'%self.gpu), data['validpr'].to('cuda:%d'%self.gpu)
                
                if data['image_02.jpg'].shape[-1] != data['proxy.png'].shape[-1]:
                    data['proxy.png'] = data['proxy.png'][...,:data['image_02.jpg'].shape[-1]]
                    data['validpr'] = data['validpr'][...,:data['image_02.jpg'].shape[-1]]
                
                # pad images
                ht, wt = data['image_02.jpg'].shape[-2], data['image_02.jpg'].shape[-1]
                pad_ht = (((ht // 128) + 1) * 128 - ht) % 128
                pad_wd = (((wt // 128) + 1) * 128 - wt) % 128
                _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
                data['image_02.jpg'] = F.pad(data['image_02.jpg'], _pad, mode='replicate')
                data['image_03.jpg'] = F.pad(data['image_03.jpg'], _pad, mode='replicate')
                
                pred_disps = self.net(data['image_02.jpg'], data['image_03.jpg'], mad = 'mad' in self.adapt_mode)
                
                # upsample and remove padding for final prediction
                pred_disp = F.interpolate( pred_disps[0], scale_factor=4., mode='bilinear')[0]*-20.
                ht, wd = pred_disp.shape[-2:]
                c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
                pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]

                # upsample and remove padding from all predictions (if needed for adaptation)
                if self.adapt_mode != 'none':
                    pred_disps = [F.interpolate( pred_disps[i], scale_factor=2**(i+2))*-20. for i in range(len(pred_disps))]                
                    pred_disps = [pred_disps[i][..., c[0]:c[1], c[2]:c[3]] for i in range(len(pred_disps))] 
                    data['image_02.jpg'] = data['image_02.jpg'][..., c[0]:c[1], c[2]:c[3]]
                    data['image_03.jpg'] = data['image_03.jpg'][..., c[0]:c[1], c[2]:c[3]]
                
                if self.adapt_mode != 'none':
                    block = self.net.sample_block(self.sample_mode, seed=batch_idx) if ('mad' in self.adapt_mode) else self.net.sample_all()
                    loss = self.net.compute_loss(data['image_02.jpg'], data['image_03.jpg'], pred_disps, data['proxy.png'], data['validpr'], adapt_mode=self.adapt_mode, idx=block)
                    loss.backward()
                    self.optimizer.step()
            
                pred_disp = pred_disp.detach()

                result = {}
                if 'groundtruth.png' in data:
                    data['validgt'] = (data['groundtruth.png'] > 0).float()
                    result = kitti_metrics(pred_disp.cpu().numpy(), data['groundtruth.png'].numpy(), data['validgt'].numpy())
                result['disp'] = pred_disp
                   
                for k in result:
                    if k != 'disp' and k!= 'errormap':
                        if k not in self.accumulator:
                            self.accumulator[k] = []
                        self.accumulator[k].append(result[k])
                
                if self.listener or self.args.verbose:
                    self.pbar.set_description("Thread %d, Seq: %s/%s, Frame %s, bad3: %2.2f"%(self.idx, self.current_run['dataset'], self.current_run['domain'], data['__key__'][0], result['bad 3'] if 'bad 3' in result else np.nan))
                    self.pbar.update(1)

                if self.server is not None and len(self.server._listening_clients) == 0:
                    self.server.gpu_locks[self.gpu].release()
                    break
                
                if self.sender and self.server is not None:

                    if (batch_idx > 0 and batch_idx % self.server.interval == 0): 

                        ret = self.server.pushing_model.acquire(blocking=False)
                        while not ret:

                            time.sleep(0.01)
                            ret = self.server.pushing_model.acquire(blocking=False)
                        block = -1 

                        if self.server.fed_mode == 'fedmad':
                            block = self.net.get_block_to_send(seed=self.idx+(batch_idx//self.server.interval))

                        self.server.push_model(self.idx, self.net, block)
                        self.server.pushing_model.release()

                if self.server is not None:
                    self.server.gpu_locks[self.gpu].release()

        if self.listener or self.args.verbose:
            self.pbar.close()

        if not self.listener and len(self.server._listening_clients) != 0:
            self.run()

        if self.server is None:
            self.print_stats()
        elif self.server is not None and self.listener:
            self.print_stats()
            self.server.remove_listening_client(self)

    def print_stats(self):
        metrs = ''
        for k in self.accumulator:
            metrs += '& %.2f '%np.array(self.accumulator[k]).mean()

        print("\nThread %d results on Seq %s:\\\\ \n%s \\\\"%(self.idx,self.current_run['domain'],str(metrs)))