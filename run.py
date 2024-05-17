import threading
import configparser
import random

from clients import StereoClient
from server import StereoServer

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='FedStereo')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--nodelist', type=str, default='cfgs/kitti_raw/madnet_federated/list_clients.ini')
parser.add_argument('--server', type=str, default=None)
parser.add_argument('--seed', type=int, default=1234)
args = parser.parse_args()

def main():

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.nodelist) as f:
        clients_file = [s.strip() for s in f.readlines() ]
        clients_ids = range(len(clients_file))

    server = None
    print('Clients:\n%s'%str(clients_file))
    if args.server is not None:
        print('Server:\n%s'%args.server)
        server = StereoServer(args.server,args)

    threads = [StereoClient(i,args,j,server=server) for i,j in zip(clients_file,clients_ids)]
    for t in threads:
        if t.listener and server is not None:
            server.link_listening_client(t)

        if t.sender and server is not None:
            server.link_sending_client(t)
    
    # Avvio dei thread
    if server is not None:
        server.start()
    for i in range(len(threads)):
        threads[i].start()
    
if __name__ == '__main__':
   main()
