import os
import tqdm
import argparse
from datetime import datetime as dt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import SalObjDataset
from models.DCF.DCF_ResNet_models import DCF_ResNet
from models.DCF.depth_calibration_models import discriminator, depth_estimator
from models.DCF.fusion import fusion
from models.BASNet.BASNet import BASNet
from models.PoolNet.poolnet import PoolNet, extra_layer, vgg16_locate
from models.u2net.u2net import U2NETP
from utils.utils import fix_seed_torch, cal_iou
from utils.criterion import DCF_losses, multi_bce_loss_fusion, KD_KLDivLoss
import optuna
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="The number of epochs of training")
parser.add_argument("--model", type=str, default='u2net', help="BASNet or PoolNet or u2net")
parser.add_argument("--img_size", type=int, default=256, help="Image size")
parser.add_argument("--batch_size", type=int, default=32, help="Size of the batches")
parser.add_argument("--num_workers", type=int, default=4, help="The number of cpu workers")
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--threshold', type=float, default=0.5, help='Mean IOU Threshold')
parser.add_argument("--is_cuda", type=bool, default=True, help="Whether to use cuda or not")
parser.add_argument('--lambda_triplet', type=float, default=0.2, help='learning rate')
parser.add_argument("--use_lamdba", type=bool, default=True, help="if False, lambda value is changed dynamically")
parser.add_argument('--lamdba_kld', type=float, default=0.2, help='ratio of loss_kld')
parser.add_argument('--decay_gamma', type=float, default=0.5, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=300, help='every n steps decays learning rate')
parser.add_argument("--teacher_path", type=str, default='save_models/teacher', help="the model folder path for teacher network")
parser.add_argument("--save_model", type=bool, default=True, help="Save model")
parser.add_argument("--save_interval", type=int, default=10, help="Interval between saving model checkpoints")
args = parser.parse_args()
fix_seed_torch(42)

def train(args):
    device = 'cuda' if torch.cuda.is_available() and args.is_cuda else 'cpu'

    t = dt.today().strftime("%Y%m%d%H%M")
    tensorboard_path = os.path.join('run', 'kd', t)
    save_path = os.path.join('run', 'kd', t, 'save_models')

    # Define Tensorboard
    writer = SummaryWriter(tensorboard_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Define Dataset, Dataloader
    train_ds = SalObjDataset(img_size=args.img_size, RGBD=True, mode=0)
    val_ds = SalObjDataset(img_size=args.img_size, RGBD=True, mode=1)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    final_iou = 0

    # Define Teacher Network(DCF)
    T_model_rgb = DCF_ResNet().to(device).requires_grad_(False)
    T_model_depth = DCF_ResNet().to(device).requires_grad_(False)
    T_model = fusion().to(device).requires_grad_(False)
    T_model_discriminator = discriminator(n_class=2).to(device).requires_grad_(False)
    T_model_estimator = depth_estimator().to(device).requires_grad_(False)

    T_model_rgb.load_state_dict(torch.load('./save_models/teacher/DCF_rgb.pth', map_location=device))
    T_model_depth.load_state_dict(torch.load('./save_models/teacher/DCF_depth.pth', map_location=device))
    T_model.load_state_dict(torch.load('./save_models/teacher/DCF.pth', map_location=device))
    T_model_discriminator.load_state_dict(
        torch.load('./save_models/teacher/DCF_discriminator.pth', map_location=device))
    T_model_estimator.load_state_dict(torch.load('./save_models/teacher/DCF_estimator.pth', map_location=device))

    T_model_rgb.eval()
    T_model_depth.eval()
    T_model.eval()
    T_model_discriminator.eval()
    T_model_estimator.eval()

    # Define Student Network
    if args.model == 'BASNet':
        S_model = BASNet(n_channels=3, kd=True).to(device)
    elif args.model == 'PoolNet':
        S_model = PoolNet('vgg', *extra_layer('vgg', vgg16_locate()), kd=True).to(device)
    elif args.model == 'u2net':
        S_model = U2NETP(3, 1, kd=True).to(device)
    else:
        AssertionError('Student model must be in BASNet or PoolNet or u2net')

    opt_G = torch.optim.Adam(S_model.parameters(), args.lr)

    scheduler_G = StepLR(opt_G, step_size=args.decay_epoch, gamma=args.decay_gamma)

    bce = nn.BCEWithLogitsLoss(reduction='mean') if args.model == 'PoolNet' else multi_bce_loss_fusion
    kld = KD_KLDivLoss
    depth_loss = nn.BCEWithLogitsLoss(reduction='mean')

    try:
        iter_num = 0
        for epoc in range(args.epochs):
            S_model.train()

            losses = 0
            iou = 0

            for imgs, gts, depths in tqdm.tqdm(train_dl, total=len(train_dl), mininterval=0.01):
                imgs = imgs.to(device)
                gts = gts.to(device)
                depths = depths.to(device)

                opt_G.zero_grad()

                with torch.no_grad():
                    tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, det_depth, tmp6, preds_T = DCF_losses(args,
                                                                                              T_model_rgb,
                                                                                              T_model_depth,
                                                                                              T_model_discriminator,
                                                                                              T_model_estimator,
                                                                                              T_model,
                                                                                              imgs, gts, depths)
                # BCE loss
                if args.model == 'PoolNet':
                    preds_S, att_1, att_2, att_3 = S_model(imgs)
                    loss_bce = bce(preds_S, gts)
                else:
                    preds_S, d1, d2, d3, d4, d5, d6, att_1, att_2, att_3 = S_model(imgs)
                    loss0, loss_bce = bce(preds_S, d1, d2, d3, d4, d5, d6, gts)

                # Attention loss
                dets_depth = det_depth.sigmoid()
                loss_att = depth_loss(att_1, dets_depth.detach()) + depth_loss(att_2, dets_depth.detach()) + depth_loss(
                    att_3, dets_depth.detach())

                # KLD loss
                loss_kld = kld(preds_S, preds_T.detach(), temperature=20)

                mean_iou = cal_iou(preds_S.sigmoid(), gts, args.threshold, device)

                if not args.use_lamdba:
                    lambda_kld = mean_iou.item()
                    loss = lambda_kld * loss_kld + loss_att + loss_bce
                else:
                    lambda_kld = args.lambda_kld
                    loss = lambda_kld * loss_kld + loss_att + loss_bce

                losses += loss.item()
                iou += mean_iou.item()

                loss.backward()
                opt_G.step()
                scheduler_G.step()

                iter_num += 1

                if (iter_num % 10) == 0:
                    writer.add_scalar('Train/Loss/loss_bce', loss_bce.item(), iter_num)
                    writer.add_scalar('Train/Loss/loss_att', loss_att.item(), iter_num)
                    writer.add_scalar('Train/Loss/loss_kld', loss_kld.item(), iter_num)
                    writer.add_scalar('Train/Loss/Total_loss', loss.item(), iter_num)
                    writer.add_scalar('Train/Loss/IoU', mean_iou.item(), iter_num)
                    writer.add_images('Train/Results/Images', imgs.sigmoid(), iter_num)
                    writer.add_images('Train/Results/Ground_Truth', gts.sigmoid(), iter_num)
                    writer.add_images('Train/Results/Teacher_Pred', preds_T.sigmoid(), iter_num)
                    writer.add_images('Train/Results/Student_Pred', preds_S.sigmoid(), iter_num)
                    writer.add_images('Train/Results/Teacher_Depth', det_depth.sigmoid(), iter_num)
                    writer.add_scalar('Train/LR/opt_G', opt_G.param_groups[0]['lr'], iter_num)

            losses /= len(train_dl)
            iou /= len(train_dl)

            S_model.eval()

            val_iter_num = 0
            val_loss = 0
            val_mean_iou = 0

            for imgs, gts, depths in val_dl:
                imgs = imgs.to(device)
                gts = gts.to(device)
                depths = depths.to(device)

                imgs.requires_grad = False
                gts.requires_grad = False
                depths.requires_grad = False

                with torch.no_grad():
                    tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, det_depth, tmp6, preds_T = DCF_losses(args,
                                                                                              T_model_rgb,
                                                                                              T_model_depth,
                                                                                              T_model_discriminator,
                                                                                              T_model_estimator,
                                                                                              T_model,
                                                                                              imgs, gts, depths,
                                                                                              )

                    if args.model == 'PoolNet':
                        preds_S, att_1, att_2, att_3 = S_model(imgs)
                        loss_bce = bce(preds_S, gts)
                    else:
                        preds_S, d1, d2, d3, d4, d5, d6, att_1, att_2, att_3 = S_model(imgs)
                        loss0, loss_bce = bce(preds_S, d1, d2, d3, d4, d5, d6, gts)

                    # Attention loss
                    dets_depth = det_depth.sigmoid()
                    loss_att = depth_loss(att_1, dets_depth.detach()) + depth_loss(att_2, dets_depth.detach()) + \
                               depth_loss(att_3, dets_depth.detach())

                    # KLD loss
                    loss_kld = kld(preds_S, preds_T.detach(), temperature=20)

                    mean_iou = cal_iou(preds_S.sigmoid(), gts, args.threshold, device)

                    if not args.use_lamdba:
                        lambda_kld = mean_iou.item()
                        loss = lambda_kld * loss_kld + loss_att + loss_bce
                    else:
                        lambda_kld = args.lambda_kld
                        loss = lambda_kld * loss_kld + loss_att + loss_bce

                val_loss += loss.item()
                val_mean_iou += mean_iou.item()

                val_iter_num += 1
                if (val_iter_num % len(val_dl)) == 0:
                    writer.add_scalar('Eval/Loss/loss_bce', loss_bce.item(), epoc)
                    writer.add_scalar('Eval/Loss/loss_att', loss_att.item(), epoc)
                    writer.add_scalar('Eval/Loss/loss_kld', loss_kld.item(), epoc)
                    writer.add_scalar('Eval/Loss/Total_loss', loss.item(), epoc)
                    writer.add_images('Eval/Results/Images', imgs.sigmoid(), epoc)
                    writer.add_images('Eval/Results/Ground_Truth', gts.sigmoid(), epoc)
                    writer.add_images('Eval/Results/Teacher_Pred', preds_T.sigmoid(), epoc)
                    writer.add_images('Eval/Results/Student_Pred', preds_S.sigmoid(), epoc)
                    writer.add_images('Eval/Results/Teacher_Depth', det_depth.sigmoid(), epoc)

            val_loss /= len(val_dl)
            val_mean_iou /= len(val_dl)
            writer.add_scalar('Eval/Loss/IoU', val_mean_iou, epoc)

            final_iou = val_mean_iou

            print(f'epoch: {epoc} | train_loss: {losses:.4f}, | train_mean_iou: {iou:.4f} | '
                  f'val_loss: {val_loss:.4f} | val_mean_iou: {val_mean_iou:.4f}')

            if (epoc + 1) % args.save_interval == 0:
                torch.save(S_model.state_dict(), os.path.join(save_path, f'Student_{args.model}_{epoc + 1}.pth'))

        if args.save_model:
            torch.save(S_model.state_dict(), os.path.join(save_path, f'Student_{args.model}.pth'))

        writer.close()

    except KeyboardInterrupt:
        torch.save(S_model.state_dict(), os.path.join(save_path, f'Student_{args.model}_{epoc + 1}.pth'))
        writer.close()

    return final_iou

def train_optuna(trial):

    cfg = {'alpha' : trial.suggest_uniform('alpha', 0.1, 0.99)}
    args.lambda_kld = cfg['alpha']
    iou_score = train(args)

    return iou_score

if __name__ == '__main__':

    train(args)
    # sampler = optuna.samplers.TPESampler()
    # study = optuna.create_study(sampler=sampler)
    # study.optimize(train_optuna, n_trials=25)
    # joblib.dump(study, './u2net_best_alpha.pkl')
    #
    # # study = joblib.load('/content/gdrive/My Drive/Colab_Data/studies/mnist_optuna.pkl')
    # # df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'system_attrs'], axis=1)
    # # df.head(3)

    # plot_optimization_history(study)