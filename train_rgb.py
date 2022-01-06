import os
import tqdm
import argparse
from datetime import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import SalObjDataset
from models.BASNet.BASNet import BASNet
from models.PoolNet.poolnet import PoolNet, extra_layer, vgg16_locate
from models.u2net.u2net import U2NETP
from utils.utils import fix_seed_torch, cal_iou
from utils.criterion import multi_bce_loss_fusion

fix_seed_torch(42)

def train_rgb(args):
    device = 'cuda' if torch.cuda.is_available() and args.is_cuda else 'cpu'

    t = dt.today().strftime("%Y%m%d%H%M")
    tensorboard_path = os.path.join('run', 'rgb', t)
    save_path = os.path.join('run', 'rgb', t, 'save_models')

    # Define Tensorboard
    writer = SummaryWriter(tensorboard_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Define Dataset, Dataloader
    train_ds = SalObjDataset(img_size=args.img_size, RGBD=False, mode=0)
    val_ds = SalObjDataset(img_size=args.img_size, RGBD=False, mode=1)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define models
    if args.model == 'BASNet':
        model = BASNet(n_channels=3).to(device)
    elif args.model == 'PoolNet':
        model = PoolNet('vgg', *extra_layer('vgg', vgg16_locate())).to(device)
    elif args.model == 'u2net':
        model = U2NETP(3, 1).to(device)
    else:
        AssertionError('Student model must be in BASNet or PoolNet or u2net')

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay_gamma)

    try:
        iter_num = 0
        for epoc in range(args.epochs):
            model.train()

            losses = 0
            iou = 0

            for imgs, gts in tqdm.tqdm(train_dl, total=len(train_dl), mininterval=0.01):
                imgs = imgs.to(device)
                gts = gts.to(device)

                optimizer.zero_grad()

                if args.model == 'PoolNet':
                    preds = model(imgs)
                    loss = nn.BCEWithLogitsLoss()(preds, gts)
                else:
                    preds, d1, d2, d3, d4, d5, d6 = model(imgs)
                    loss0, loss = multi_bce_loss_fusion(preds, d1, d2, d3, d4, d5, d6, gts)

                mean_iou = cal_iou(preds.sigmoid(), gts, args.threshold, device)

                losses += loss
                iou += mean_iou

                loss.backward()
                optimizer.step()
                scheduler.step()

                iter_num += 1
                if (iter_num % 10) == 0:
                    writer.add_scalar('Train/loss', loss.item(), iter_num)
                    writer.add_scalar('Train/Loss/IoU', mean_iou.item(), iter_num)
                    writer.add_images('Train/Pred', preds.sigmoid(), iter_num)
                    writer.add_images('Train/Images', imgs.sigmoid(), iter_num)
                    writer.add_images('Train/Ground_truths', gts.sigmoid(), iter_num)
                    writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], iter_num)

            losses /= len(train_dl)
            iou /= len(train_dl)

            model.eval()

            val_loss = 0
            val_iter_num = 0
            val_mean_iou = 0

            for imgs, gts in val_dl:
                imgs = imgs.to(device)
                gts = gts.to(device)

                imgs.requires_grad = False
                gts.requires_grad = False

                with torch.no_grad():
                    if args.model == 'PoolNet':
                        preds = model(imgs)
                        loss = nn.BCEWithLogitsLoss()(preds, gts)
                    else:
                        preds, d1, d2, d3, d4, d5, d6 = model(imgs)
                        loss0, loss = multi_bce_loss_fusion(preds, d1, d2, d3, d4, d5, d6, gts)

                    val_loss += loss
                    val_mean_iou += cal_iou(preds.sigmoid(), gts, args.threshold, device)

                val_iter_num += 1

                if (val_iter_num % len(val_dl)) == 0:
                    writer.add_images('Eval/Pred', preds.sigmoid(), epoc)
                    writer.add_images('Eval/Images', imgs.sigmoid(), epoc)
                    writer.add_images('Eval/Ground_truths', gts.sigmoid(), epoc)

            val_loss /= len(val_dl)
            val_mean_iou /= len(val_dl)
            writer.add_scalar('Eval/loss', val_loss.item(), epoc)
            writer.add_scalar('Eval/Loss/IoU', val_mean_iou.item(), epoc)

            print(f'epoch: {epoc} | train_loss: {losses:.4f}, | train_mean_iou: {iou.item():.4f} | '
                  f'val_loss: {val_loss.item():.4f} | val_mean_iou: {val_mean_iou.item():.4f}')

            if (epoc + 1) % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_path, f'{args.model}_{epoc + 1}.pth'))

        if args.save_model:
            torch.save(model.state_dict(), os.path.join(save_path, f'{args.model}.pth'))

        writer.close()

    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(save_path, f'{args.model}_{epoc + 1}.pth'))
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="The number of epochs of training")
    parser.add_argument("--model", type=str, default='PoolNet', help="BASNet or PoolNet or u2net")
    parser.add_argument("--is_cuda", type=bool, default=True, help="Whether to use cuda or not")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of cpu workers")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--threshold', type=float, default=0.5, help='Mean IOU Threshold')
    parser.add_argument('--decay_gamma', type=float, default=0.5, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=300, help='every n steps decays learning rate')
    parser.add_argument("--save_model", type=bool, default=True, help="Save model")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval between saving model checkpoints")

    args = parser.parse_args()

    train_rgb(args)
