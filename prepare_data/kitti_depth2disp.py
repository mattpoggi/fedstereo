import cv2
import numpy as np
import argparse

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1226] = 707.0912
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

parser = argparse.ArgumentParser(description='Depth2Disp')
parser.add_argument('--src', type=str, required=True)
parser.add_argument('--tgt', type=str, required=True)
args = parser.parse_args()

depth = cv2.imread(args.src, -1) / 256.
disp = width_to_focal[depth.shape[1]] * 0.54 / depth
disp[depth==0] = 0
cv2.imwrite(args.tgt, (disp*256).astype(np.uint16))