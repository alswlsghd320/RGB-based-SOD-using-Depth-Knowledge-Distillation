# https://github.com/PanoAsh/Evaluation-on-salient-object-detection/blob/master/evaluator.py
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader

from datasets.datasets import TestDataset
from models.BASNet.BASNet import BASNet
from models.PoolNet.poolnet import PoolNet, extra_layer, vgg16_locate
from models.u2net.u2net import U2NETP
from utils.utils import fix_seed_torch, thresholding_pred, cal_mae, cal_fbeta, cal_Emeasure, cal_Smeasure

fix_seed_torch(42)

def test(args):
    device = 'cuda' if torch.cuda.is_available() and args.is_cuda else 'cpu'

    test_ds = TestDataset(img_size=args.img_size, RGBD=False, dataset=args.dataset)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    kd = 'kd' in args.path

    # Define models
    if args.model == 'BASNet':
        model = BASNet(n_channels=3, kd=kd).to(device)
    elif args.model == 'PoolNet':
        model = PoolNet('vgg', *extra_layer('vgg', vgg16_locate()), kd=kd).to(device)
    elif args.model == 'u2net':
        model = U2NETP(3, 1, kd=kd).to(device)
    else:
        AssertionError('Student model must be in BASNet or PoolNet or u2net')

    model.load_state_dict(torch.load(args.path, map_location=device))
    model.eval()

    mae = 0
    F = 0
    E = 0
    S = 0

    for imgs, gts in tqdm.tqdm(test_dl, total=len(test_dl), mininterval=0.01):
        imgs = imgs.to(device)
        gts = gts.to(device)

        with torch.no_grad():
            if kd:
                if args.model == 'PoolNet':
                    preds, att_1, att_2, att_3 = model(imgs)
                else:
                    preds, d1, d2, d3, d4, d5, d6, att_1, att_2, att_3 = model(imgs)
            else:
                if args.model == 'PoolNet':
                    preds = model(imgs)
                else:
                    preds, d1, d2, d3, d4, d5, d6 = model(imgs)

            preds = thresholding_pred(preds)

            mae += cal_mae(preds, gts).item()
            F += cal_fbeta(preds, gts).item()
            E += cal_Emeasure(preds, gts).item()
            S += cal_Smeasure(preds, gts, device=device).item()

    mae /= len(test_dl)
    F /= len(test_dl)
    E /= len(test_dl)
    S /= len(test_dl)

    print(f'Test {args.model}')
    print(f'mae: {mae:.4f} | F measure: {F:.4f} | E measure: {E:.4f} | S measure: {S:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="The path of model.pth")
    parser.add_argument("--model", type=str, default='PoolNet', help="BASNet or PoolNet or u2net")
    parser.add_argument("--dataset", type=str, default='DUT-RGBD/test_data',
                        help="DUT-RGBD/test_data, LFSD, SSD, STEREO-1000")
    parser.add_argument("--is_cuda", type=bool, default=True, help="Whether to use cuda or not")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--threshold", type=float, default=0.5, help="")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of cpu workers")

    args = parser.parse_args()
    test(args)



