import os
import tqdm
import argparse
from datetime import datetime as dt

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import SalObjDataset
from models.DCF.DCF_ResNet_models import DCF_ResNet
from models.DCF.depth_calibration_models import discriminator, depth_estimator
from models.DCF.fusion import fusion
from utils.utils import fix_seed_torch, clip_gradient, cal_iou
from utils.criterion import DCF_losses

fix_seed_torch(42)

def train_rgbd(args):
    device = 'cuda' if torch.cuda.is_available() and args.is_cuda else 'cpu'

    t = dt.today().strftime("%Y%m%d%H%M")
    tensorboard_path = os.path.join('run', 'rgbd', t)
    save_path = os.path.join('run', 'rgbd', t, 'save_models')

    # Define Tensorboard
    writer = SummaryWriter(tensorboard_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Define Dataset, Dataloader
    train_ds = SalObjDataset(img_size=args.img_size, RGBD=True, mode=0)
    val_ds = SalObjDataset(img_size=args.img_size, RGBD=True, mode=1)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define models
    model_rgb = DCF_ResNet().to(device)
    model_depth = DCF_ResNet().to(device)
    model_discriminator = discriminator(n_class=2).to(device)
    model_estimator = depth_estimator().to(device)
    model = fusion().to(device)

    #Load

    opt_rgb = torch.optim.Adam(model_rgb.parameters(), args.lr)
    opt_depth = torch.optim.Adam(model_depth.parameters(), args.lr)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    scheduler_rgb = StepLR(opt_rgb, step_size=args.decay_epoch, gamma=args.decay_gamma)
    scheduler_depth = StepLR(opt_depth, step_size=args.decay_epoch, gamma=args.decay_gamma)
    scheduler_opt = StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay_gamma)

    try:
        iter_num = 0
        for epoc in range(args.epochs):
            model_rgb.train()
            model_depth.train()
            model_discriminator.train()
            model_estimator.train()
            model.train()

            losses = []

            for imgs, gts, depths in tqdm.tqdm(train_dl, total=len(train_dl), mininterval=0.01):
                imgs = imgs.to(device)
                gts = gts.to(device)
                depths = depths.to(device)
                _depths = depths

                opt_rgb.zero_grad()
                opt_depth.zero_grad()
                optimizer.zero_grad()

                loss_rgb, loss_depth, loss_sal, \
                loss_triplet, loss_fusion, det_rgb, \
                det_depth, depth_calibrated, pred = DCF_losses(args,
                                                               model_rgb,
                                                               model_depth,
                                                               model_discriminator,
                                                               model_estimator,
                                                               model,
                                                               imgs, gts, depths)
                loss_rgb.backward()
                clip_gradient(opt_rgb, args.clip)
                opt_rgb.step()
                scheduler_rgb.step()

                loss_depth.backward()
                clip_gradient(opt_depth, args.clip)
                opt_depth.step()
                scheduler_depth.step()

                loss_fusion.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()
                scheduler_opt.step()

                iter_num += 1
                if (iter_num % 10) == 0:
                    mean_iou = cal_iou(pred.sigmoid(), gts, args.threshold, device)

                    writer.add_scalar('Train/Loss/rgb', loss_rgb.item(), iter_num)
                    writer.add_scalar('Train/Loss/depth', loss_depth.item(), iter_num)
                    writer.add_scalar('Train/Loss/sal', loss_sal.item(), iter_num)
                    writer.add_scalar('Train/Loss/triplet', loss_triplet.item(), iter_num)
                    writer.add_scalar('Train/Loss/fusion', loss_fusion.item(), iter_num)
                    writer.add_scalar('Train/Loss/IoU', mean_iou.item(), iter_num)
                    writer.add_images('Train/Results/Images', imgs.sigmoid(), iter_num)
                    writer.add_images('Train/Results/Ground_Truth', gts.sigmoid(), iter_num)
                    writer.add_images('Train/Results/rgb', det_rgb.sigmoid(), iter_num)
                    writer.add_images('Train/Results/depth_map', _depths, iter_num)
                    writer.add_images('Train/Results/calibrated_depth', depth_calibrated, iter_num)
                    writer.add_images('Train/Results/depth', det_depth.sigmoid(), iter_num)
                    writer.add_images('Train/Results/Pred', pred.sigmoid(), iter_num)
                    writer.add_scalar('Train/LR/opt_rgb', opt_rgb.param_groups[0]['lr'], iter_num)
                    writer.add_scalar('Train/LR/opt_depth', opt_depth.param_groups[0]['lr'], iter_num)
                    writer.add_scalar('Train/LR/optimizer', optimizer.param_groups[0]['lr'], iter_num)

                if (iter_num % len(train_dl)) == 0:
                    losses.append(loss_rgb.item())
                    losses.append(loss_depth.item())
                    losses.append(loss_fusion.item())
                    losses.append(mean_iou.item())

            model_rgb.eval()
            model_depth.eval()
            model_discriminator.eval()
            model_estimator.eval()
            model.eval()

            val_loss_rgb = 0
            val_loss_depth = 0
            val_loss_fusion = 0
            val_iter_num = 0
            val_mean_iou = 0

            for imgs, gts, depths in val_dl:
                imgs = imgs.to(device)
                gts = gts.to(device)
                depths = depths.to(device)
                _depths = depths

                imgs.requires_grad = False
                gts.requires_grad = False
                depths.requires_grad = False

                with torch.no_grad():
                    loss_rgb, loss_depth, loss_sal, \
                    loss_triplet, loss_fusion, det_rgb, \
                    det_depth, depth_calibrated, pred = DCF_losses(args,
                                                                   model_rgb,
                                                                   model_depth,
                                                                   model_discriminator,
                                                                   model_estimator,
                                                                   model,
                                                                   imgs, gts, depths)
                    val_loss_rgb += loss_rgb
                    val_loss_depth += loss_depth
                    val_loss_fusion += loss_fusion
                    val_mean_iou += cal_iou(pred.sigmoid(), gts, args.threshold, device)

                val_iter_num += 1
                if (val_iter_num % len(val_dl)) == 0:
                    writer.add_images('Eval/Results/rgb', det_rgb.sigmoid(), val_iter_num)
                    writer.add_images('Eval/Results/depth_map', _depths, val_iter_num)
                    writer.add_images('Eval/Results/calibrated_depth', depth_calibrated, val_iter_num)
                    writer.add_images('Eval/Results/depth', det_depth.sigmoid(), val_iter_num)
                    writer.add_images('Eval/Results/Pred', pred.sigmoid(), val_iter_num)
                    writer.add_images('Eval/Results/Images', imgs.sigmoid(), val_iter_num)
                    writer.add_images('Eval/Results/Ground_Truth', gts.sigmoid(), val_iter_num)

            val_loss_rgb /= len(val_dl)
            val_loss_depth /= len(val_dl)
            val_loss_fusion /= len(val_dl)
            val_mean_iou /= len(val_dl)
            writer.add_scalar('Eval/Loss/rgb', val_loss_rgb.item(), epoc)
            writer.add_scalar('Eval/Loss/depth', val_loss_depth.item(), epoc)
            writer.add_scalar('Eval/Loss/fusion', val_loss_fusion.item(), epoc)
            writer.add_scalar('Eval/Loss/IoU', val_mean_iou.item(), epoc)

            print(f'epoch: {epoc}, \ntrain_loss_rgb: {losses[0]:.4f}, |'
                  f'train_loss_depth: {losses[1]:.4f}, |'
                  f'train_loss_fusion: {losses[2]:.4f}, |'
                  f'train_mean_iou: {losses[3]:.4f}\n'
                  f'val_loss_rgb: {val_loss_rgb.item():.4f},   |'
                  f'val_loss_depth: {val_loss_depth.item():.4f},   |'
                  f'val_loss_fusion: {val_loss_fusion.item():.4f}   |'
                  f'val_mean_iou: {val_mean_iou.item():.4f}')

            if (epoc + 1) % args.save_interval == 0:
                torch.save(model_rgb.state_dict(), os.path.join(save_path, f'DCF_rgb_{epoc + 1}.pth'))
                torch.save(model_depth.state_dict(), os.path.join(save_path, f'DCF_depth_{epoc + 1}.pth'))
                torch.save(model_discriminator.state_dict(),
                           os.path.join(save_path, f'DCF_discriminator_{epoc + 1}.pth'))
                torch.save(model_estimator.state_dict(), os.path.join(save_path, f'DCF_estimator_{epoc + 1}.pth'))
                torch.save(model.state_dict(), os.path.join(save_path, f'DCF_{epoc + 1}.pth'))

        if args.save_model:
            torch.save(model_rgb.state_dict(), os.path.join(save_path, f'DCF_rgb.pth'))
            torch.save(model_depth.state_dict(), os.path.join(save_path, f'DCF_depth.pth'))
            torch.save(model_discriminator.state_dict(), os.path.join(save_path, f'DCF_discriminator.pth'))
            torch.save(model_estimator.state_dict(), os.path.join(save_path, f'DCF_estimator.pth'))
            torch.save(model.state_dict(), os.path.join(save_path, f'DCF.pth'))

        writer.close()

    except KeyboardInterrupt:
        torch.save(model_rgb.state_dict(), os.path.join(save_path, f'DCF_rgb_{epoc + 1}.pth'))
        torch.save(model_depth.state_dict(), os.path.join(save_path, f'DCF_depth_{epoc + 1}.pth'))
        torch.save(model_discriminator.state_dict(), os.path.join(save_path, f'DCF_discriminator_{epoc + 1}.pth'))
        torch.save(model_estimator.state_dict(), os.path.join(save_path, f'DCF_estimator_{epoc + 1}.pth'))
        torch.save(model.state_dict(), os.path.join(save_path, f'DCF_{epoc + 1}.pth'))
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60, help="The number of epochs of training")
    parser.add_argument("--dataset", type=str, default='NJU2K', help="NJU2K or SIP")
    parser.add_argument("--is_cuda", type=bool, default=True, help="Whether to use cuda or not")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of cpu workers")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lambda_triplet', type=float, default=0.2, help='learning rate')
    parser.add_argument('--threshold', type=float, default=0.5, help='Mean IOU Threshold')
    parser.add_argument('--decay_gamma', type=float, default=0.5, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=300, help='every n steps decays learning rate')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument("--save_model", type=bool, default=True, help="Save model")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval between saving model checkpoints")

    args = parser.parse_args()

    train_rgbd(args)

