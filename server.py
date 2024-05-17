import threading
import configparser

import torch.nn as nn
import torch
import time
import copy

from models import *

class StereoServer(threading.Thread):

    def __init__(self, cfg, args):
        
        config = configparser.ConfigParser()
        config.read(cfg)
        self.args=args
        self.gpu = int(config['adaptation']['gpu'])
        self.pushing_model = threading.Lock()

        self.gpu_locks = [threading.Lock() for i in range(4)]

        self.model = config['network']['model']
        self.net = models_lut[self.model](args)
        self.net = nn.DataParallel(self.net)
        self.net.load_state_dict(torch.load(config['network']['checkpoint'], torch.device('cuda:%d'%self.gpu))['state_dict'])
        self.net = self.net.module

        self.current_weights = self.net.state_dict()

        self.fed_mode = config['federated']['mode']
        self.interval = int(config['federated']['interval'])
        self.bootstrap_countdown = int(config['federated']['bootstrap'])

        self.models = []
        self._listening_clients = []
        self._sending_clients = []
        threading.Thread.__init__(self)

        self.blocks = {0: ['feature_extraction.block1', 'feature_extraction.block2', 'decoder2'],
                        1: ['feature_extraction.block3', 'decoder3'],
                        2: ['feature_extraction.block4', 'decoder4'],
                        3: ['feature_extraction.block5', 'decoder5'],
                        4: ['feature_extraction.block6', 'decoder6'] }

    def link_listening_client(self,client):
        self._listening_clients.append(client)

    def link_sending_client(self,client):
        self._sending_clients.append(client)

    def remove_listening_client(self,client):
        self._listening_clients = [c for c in self._listening_clients if c != client]

    def remove_sending_client(self,client):
        self._sending_clients = [c for c in self._sending_clients if c != client]

    def __average_weights(self,models):
        w_avg = {}
        w = [i['weights'] for i in models]

        for key in self.current_weights.keys():

            avg_k = [n[key].to(device="cuda:%d"%self.gpu) for n in w if key in n.keys()]
            w_avg[key] = self.current_weights[key] if len(avg_k) == 0 else torch.mean(torch.stack(avg_k,0),dim=0)
            
        self.current_weights = w_avg
        return w_avg

    def push_model(self, idx, model, block=-1):
        state_dict = {} 
        for key in model.state_dict().keys():
            
            if block==-1 or any(x in key for x in self.blocks[block]):

                state_dict[key] = model.state_dict()[key].detach().clone()

        if idx not in [m['client_id'] for m in self.models]:

            self.models.append({'client_id':idx, 'weights':state_dict})
        else:

            self.models = [m for m in self.models if m['client_id'] != idx]
            self.models.append({'client_id':idx, 'weights':state_dict})
        
    def push_model_easy(self, idx, model, block=-1, gpu=0):
        if idx not in [m['client_id'] for m in self.models]:

            self.models.append({'client_id':idx, 'weights':copy.deepcopy(model.to('cuda:%d'%self.gpu).state_dict())})
        else:

            self.models = [m for m in self.models if m['client_id'] != idx]
            self.models.append({'client_id':idx, 'weights':copy.deepcopy(model.to('cuda:%d'%self.gpu).state_dict())})
        
        model.to('cuda:%d'%gpu)

    def update_model(self):
        res = self.__average_weights(self.models)
        self.models = []
        return res

    def run(self):
        while True:

            time.sleep(0.01)
            if len(self.models) > 0 and len(self.models) == len(self._sending_clients):

                new_model = self.update_model()
                for t in self._listening_clients:
                    
                    t.net.load_state_dict(new_model)
                    self.bootstrap_countdown -= 1
                    t.bootstrapped = True if self.bootstrap_countdown <= 0 else False
                
            if len(self._listening_clients) == 0:
                break