import torch
import torch.nn as nn
import torch.nn.functional as F
from .corr import CorrBlock1D
from .submodule import *
from ..losses import *

class MADNet2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.feature_extraction = feature_extraction()
        self.decoder6 = disparity_decoder(5+192)
        self.decoder5 = disparity_decoder(5+128+1)
        self.decoder4 = disparity_decoder(5+96+1)
        self.decoder3 = disparity_decoder(5+64+1)
        self.decoder2 = disparity_decoder(5+32+1)

        self.sample_distribution = torch.zeros(5,requires_grad=False) 
        self.softmax = torch.nn.Softmax()
        self.loss_t1, self.loss_t2 = 0, 0
        self.last_trained_blocks = []
        self.updates_histogram = torch.zeros(5,requires_grad=False)
        self.accumulated_loss = torch.zeros(5,requires_grad=False)
        self.loss_weights = [1, 1, 1, 1, 1]

    @torch.no_grad()
    def coords_grid(self, batch, ht, wd):
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    @torch.no_grad()
    def sample_block(self, sample_mode='prob', seed=0):
        if sample_mode == 'prob':
            prob = self.softmax(self.sample_distribution)
            block = np.random.choice(range(self.sample_distribution.shape[0]),size=1,p=prob.numpy())[0]
        else:
            block = 0 
        self.updates_histogram[block] += 1
        return block

    @torch.no_grad()  
    def sample_all(self):
        self.updates_histogram += 1
        return -1

    @torch.no_grad()   
    def get_block_to_send(self, sample_mode='prob', seed=0):
        if sample_mode == 'prob':
            prob = (self.softmax(self.updates_histogram))
            block = np.random.choice(range(self.updates_histogram.shape[0]),size=1,p=prob.numpy())[0]
            self.updates_histogram[block] *= 0.9
            self.accumulated_loss *= 0
        else:
            block = 0
        
        return block       

    @torch.no_grad()
    def update_sample_distribution(self, block, new_loss, mode='mad'):
        if self.loss_t1 == 0 and self.loss_t2 == 0:
            self.loss_t1 = new_loss
            self.loss_t2 = new_loss
            
        expected_loss = 2 * self.loss_t1 - self.loss_t2	
        gain_loss = expected_loss - new_loss
        self.sample_distribution = 0.99 * self.sample_distribution
        for i in self.last_trained_blocks:
            self.sample_distribution[i] += 0.01 * gain_loss

        self.last_trained_blocks = [block]
        self.loss_t2 = self.loss_t1
        self.loss_t1 = new_loss

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = self.coords_grid(N, H, W).to(img.device)
        coords1 = self.coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def forward(self, image2, image3, mad=False):
        """ Estimate optical flow between pair of frames """

        im2_fea = self.feature_extraction(image2, mad)
        im3_fea = self.feature_extraction(image3, mad)

        corr_block = CorrBlock1D
           
        corr_fn6 = corr_block(im2_fea[6], im3_fea[6], radius=2, num_levels=1)
        corr_fn5 = corr_block(im2_fea[5], im3_fea[5], radius=2, num_levels=1)
        corr_fn4 = corr_block(im2_fea[4], im3_fea[4], radius=2, num_levels=1)
        corr_fn3 = corr_block(im2_fea[3], im3_fea[3], radius=2, num_levels=1)
        corr_fn2 = corr_block(im2_fea[2], im3_fea[2], radius=2, num_levels=1)

        coords0, coords1_6 = self.initialize_flow(im2_fea[6])
        coords0, coords1_5 = self.initialize_flow(im2_fea[5])
        coords0, coords1_4 = self.initialize_flow(im2_fea[4])
        coords0, coords1_3 = self.initialize_flow(im2_fea[3])
        coords0, coords1_2 = self.initialize_flow(im2_fea[2])
        
        
        corr6 = corr_fn6(coords1_6)
        disp6 = self.decoder6(torch.cat((im2_fea[6],corr6), 1))
        disp6_u = F.interpolate(disp6 if not mad else disp6.detach(), scale_factor=2)*20./32

        coords1_5 = coords1_5 + disp6_u
        corr5 = corr_fn5(coords1_5)
        disp5 = self.decoder5(torch.cat((im2_fea[5],corr5,disp6_u), 1))
        disp5_u = F.interpolate(disp5 if not mad else disp5.detach(), scale_factor=2)*20./16

        coords1_4 = coords1_4 + disp5_u
        corr4 = corr_fn4(coords1_4)
        disp4 = self.decoder4(torch.cat((im2_fea[4],corr4,disp5_u), 1))
        disp4_u = F.interpolate(disp4 if not mad else disp4.detach(), scale_factor=2)*20./8

        coords1_3 = coords1_3 + disp4_u
        corr3 = corr_fn3(coords1_3)
        disp3 = self.decoder3(torch.cat((im2_fea[3],corr3,disp4_u), 1))
        disp3_u = F.interpolate(disp3 if not mad else disp3.detach(), scale_factor=2)*20./4

        coords1_2 = coords1_2 + disp3_u
        corr2 = corr_fn2(coords1_2)
        disp2 = self.decoder2(torch.cat((im2_fea[2],corr2,disp3_u), 1))

        return disp2, disp3, disp4, disp5, disp6

    def training_loss(self, pred_disps, gt_disp):
        # From MADNet original paper
        # pred_disps are disp2, disp3, disp4, disp5, disp6 at their original resolution
        # gt_disp is groundtruth disparity at full resolution (no validity mask required, since SceneFlow is dense)
        loss =  0.005*F.l1_loss( pred_disps[0], -F.interpolate(gt_disp, scale_factor=1/4., mode='nearest')/20., reduction='sum') + \
            0.01*F.l1_loss( pred_disps[1], -F.interpolate(gt_disp, scale_factor=1/8., mode='nearest')/20., reduction='sum') + \
            0.02*F.l1_loss( pred_disps[2], -F.interpolate(gt_disp, scale_factor=1/16., mode='nearest')/20., reduction='sum') + \
            0.08*F.l1_loss( pred_disps[3], -F.interpolate(gt_disp, scale_factor=1/32., mode='nearest')/20., reduction='sum') 
        return loss


    def compute_loss(self, image2, image3, predictions, gt, validgt, adapt_mode='full', idx=-1):

        if adapt_mode == 'full':
            loss =  [self_supervised_loss(predictions[0], image2, image3),
                    self_supervised_loss(predictions[1], image2, image3),
                    self_supervised_loss(predictions[2], image2, image3),
                    self_supervised_loss(predictions[3], image2, image3),
                    self_supervised_loss(predictions[4], image2, image3)]
            self.accumulated_loss += torch.stack([loss[i] * self.loss_weights[i] for i in range(len(loss))],0).detach().cpu()
            loss = sum(loss).mean()

        elif adapt_mode == 'full++':   
            # legacy from original MADNet training (classical average reduction without any weights gives almost identical results)
            loss =  [0.001*F.l1_loss(predictions[0][validgt>0], gt[validgt>0], reduction='sum') / 20., 
                    0.001*F.l1_loss(predictions[1][validgt>0], gt[validgt>0], reduction='sum') / 20., 
                    0.001*F.l1_loss(predictions[2][validgt>0], gt[validgt>0], reduction='sum') / 20., 
                    0.001*F.l1_loss(predictions[3][validgt>0], gt[validgt>0], reduction='sum') / 20.,
                    0.001*F.l1_loss(predictions[4][validgt>0], gt[validgt>0], reduction='sum') / 20.]
            self.accumulated_loss += torch.stack([loss[i] * self.loss_weights[i] for i in range(len(loss))],0).detach().cpu()
            loss = sum(loss).mean()
        
        elif adapt_mode == 'mad':
            loss = self_supervised_loss(predictions[idx], image2, image3)

        elif adapt_mode == 'mad++':
            loss = F.l1_loss(predictions[idx][validgt>0], gt[validgt>0]) 

        if 'mad' in adapt_mode:
            self.update_sample_distribution(idx,loss.cpu(),adapt_mode)

        return loss