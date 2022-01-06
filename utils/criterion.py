# reference: https://github.com/irfanICMLL/structure_knowledge_distillation

import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import sim_dis_compute

def DCF_losses(args,
               model_rgb,
               model_depth,
               model_discriminator,
               model_estimator,
               model,
               imgs,
               gts,
               depths):
    bce = torch.nn.BCEWithLogitsLoss()
    tml = torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean')

    # RGB Stream
    att_rgb, det_rgb, x3_r, x4_r, x5_r = model_rgb(imgs)
    loss1_rgb = bce(att_rgb, gts)
    loss2_rgb = bce(det_rgb, gts)
    loss_rgb = (loss1_rgb + loss2_rgb) / 2.0

    # Depth Calibration module
    with torch.no_grad():
        score = model_discriminator(depths)
        score = torch.softmax(score, dim=1)
        x3_, x4_, x5_ = x3_r.detach(), x4_r.detach(), x5_r.detach()
        pred_depth = model_estimator(imgs, x3_, x4_, x5_)
        depth_calibrated = torch.mul(depths, score[:, 0].view(-1, 1, 1, 1).expand(-1, 1, args.img_size, args.img_size)) \
                           + torch.mul(pred_depth,
                                       score[:, 1].view(-1, 1, 1, 1).expand(-1, 1, args.img_size, args.img_size))
    # Depth Stream
    depths = depth_calibrated
    depths = torch.cat([depths, depths, depths], dim=1)
    att_depth, det_depth, x3_d, x4_d, x5_d = model_depth(depths)
    loss1_depth = bce(att_depth, gts)
    loss2_depth = bce(det_depth, gts)
    loss_depth = (loss1_depth + loss2_depth) / 2.0

    # Fusion Stream
    x3_rd, x4_rd, x5_rd = x3_r.detach(), x4_r.detach(), x5_r.detach()
    x3_dd, x4_dd, x5_dd = x3_d.detach(), x4_d.detach(), x5_d.detach()
    att, pred, x3, x4, x5 = model(x3_rd, x4_rd, x5_rd, x3_dd, x4_dd, x5_dd)

    loss1 = bce(att, gts)
    loss2 = bce(pred, gts)
    loss_sal = (loss1 + loss2) / 2.0

    loss_tml1 = tml(x3, gts * x3, (1 - gts) * x3)
    loss_tml2 = tml(x4, gts * x4, (1 - gts) * x4)
    loss_tml3 = tml(x5, gts * x5, (1 - gts) * x5)
    loss_triplet = (loss_tml1 + loss_tml2 + loss_tml3) / 3.0

    loss_fusion = loss_sal + args.lambda_triplet * loss_triplet

    return loss_rgb, loss_depth, loss_sal, loss_triplet, loss_fusion, det_rgb, det_rgb, depth_calibrated, pred

def multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(reduction='mean')

    d0 = torch.sigmoid(d0)
    d1 = torch.sigmoid(d1)
    d2 = torch.sigmoid(d2)
    d3 = torch.sigmoid(d3)
    d4 = torch.sigmoid(d4)
    d5 = torch.sigmoid(d5)
    d6 = torch.sigmoid(d6)

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

def KD_KLDivLoss(Stu_output, Tea_output, temperature):
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(Stu_output/T, dim=1), F.softmax(Tea_output/T, dim=1))
    KD_loss = KD_loss * T * T
    return KD_loss