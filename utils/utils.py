import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torchmetrics import IoU
from torchmetrics.functional import fbeta, mean_absolute_error


def fix_seed_torch(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = False  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1) #1x1x4

    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2) / ((f_T.shape[-1]*f_T.shape[-2])**2) / f_T.shape[0]
    sim_dis = sim_err.sum()

    return sim_dis

def cal_iou(preds, gts, threshold=0.5, device='cpu'):
    preds = thresholding_pred(preds, threshold)

    iou = IoU(num_classes=2).to(device)

    return iou(preds.long(), gts.long())

def thresholding_pred(preds, threshold=0.5):
    ma = torch.max(preds)
    mi = torch.min(preds)

    preds = (preds-mi)/(ma-mi)
    preds[preds >= threshold] = 1.
    preds[preds < threshold] = 0.

    return preds

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

def cal_fbeta(preds, gts):
    return fbeta(preds.flatten().long(), gts.flatten().long())

def cal_mae(preds, gts):
    return mean_absolute_error(preds.flatten().long(), gts.flatten().long())

def _object(preds, gts):
    temp = preds[gts == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    return score

def _S_object(preds, gts):
    fg = torch.where(gts==0, torch.zeros_like(preds), preds)
    bg = torch.where(gts==1, torch.zeros_like(preds), 1-preds)
    o_fg = _object(fg, gts)
    o_bg = _object(bg, 1-gts)
    u = gts.mean()
    Q = u * o_fg + (1-u) * o_bg
    return Q

def _centroid(gts, device):
    rows, cols = gts.size()[-2:]
    gts = gts.view(rows, cols)
    if gts.sum() == 0:
        X = torch.eye(1) * round(cols / 2).to(device)
        Y = torch.eye(1) * round(rows / 2).to(device)
    else:
        total = gts.sum()
        i = torch.from_numpy(np.arange(0,cols)).float().to(device)
        j = torch.from_numpy(np.arange(0,rows)).float().to(device)
        X = torch.round((gts.sum(dim=0)*i).sum() / total)
        Y = torch.round((gts.sum(dim=1)*j).sum() / total)
    return X.long(), Y.long()

def _ssim(preds, gts):
    gts = gts.float()
    h, w = preds.size()[-2:]
    N = h * w
    x = preds.mean()
    y = gts.mean()
    sigma_x2 = ((preds - x) * (preds - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gts - y) * (gts - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((preds - x) * (gts - y)).sum() / (N - 1 + 1e-20)

    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + 1e-20)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q

def _S_region(preds, gts, device):
    X, Y = _centroid(gts, device)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gts, X, Y)
    p1, p2, p3, p4 = _dividePrediction(preds, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4

    return Q

def _dividePrediction(preds, X, Y):
    h, w = preds.size()[-2:]
    preds = preds.view(h, w)
    LT = preds[:Y, :X]
    RT = preds[:Y, X:w]
    LB = preds[Y:h, :X]
    RB = preds[Y:h, X:w]
    return LT, RT, LB, RB

def _divideGT(gts, X, Y):
    h, w = gts.size()[-2:]
    area = h*w
    gts = gts.view(h, w)
    LT = gts[:Y, :X]
    RT = gts[:Y, X:w]
    LB = gts[Y:h, :X]
    RB = gts[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def cal_Smeasure(preds, gts, alpha=0.5, device='cpu'):
    y = gts.mean()
    if y == 0:
        x = preds.mean()
        Q = 1.0 - x
    elif y == 1:
        x = preds.mean()
        Q = x

    Q = alpha * _S_object(preds, gts) + (1-alpha) * _S_region(preds, gts, device)

    if Q.item() < 0:
        Q = torch.FloatTensor([0.0])

    return Q

def cal_Emeasure(y_pred, y, num=255):

    score = torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)

    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)

    return score.max()
