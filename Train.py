import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, get_coef,cal_ual
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim
from lib.Network import Network_interFA_noSpade_noEdge_ODE_slot_channel4

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def train(train_loader, model, optimizer, epoch, save_path, writer):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edge) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda(device=device_ids[0])
            gts = gts.cuda(device=device_ids[0])


            preds = model(images)

            ual_coef = get_coef(iter_percentage=i/total_step, method='cos')
            ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
            ual_loss *= ual_coef

            loss_init = structure_loss(preds[0], gts)*0.0625 + structure_loss(preds[1], gts)*0.125 + structure_loss(preds[2], gts)*0.25 + \
                        structure_loss(preds[3], gts)*0.5
            loss_final = structure_loss(preds[4], gts)



            loss = loss_init + loss_final + 2*ual_loss

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                        format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,'Loss_total': loss.data},
                                   global_step=step)




        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    mae_sum = 0

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda(device=device_ids[0])

            result = model(image)

            # 注意：PyTorch 1.10+ 建议用 interpolate 替代 upsample
            res = F.interpolate(
                result[4],
                size=gt.shape,
                mode='bilinear',
                align_corners=False
            )
            res = res.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.abs(res - gt).mean()

    mae = mae_sum / test_loader.size
    writer.add_scalar('MAE', mae, global_step=epoch)
    print(f'Epoch: {epoch}, MAE: {mae:.6f}, bestMAE: {best_mae:.6f}, bestEpoch: {best_epoch}.')

    # —— 每轮都保存 ——
    os.makedirs(save_path, exist_ok=True)
    epoch_save_path = os.path.join(save_path, f'Net_epoch_{epoch}.pth')
    torch.save(model.state_dict(), epoch_save_path)
    print(f'Saved model for epoch {epoch} -> {epoch_save_path}')

    # —— 保留“保存最佳”逻辑 ——
    if epoch == 1 or mae < best_mae:
        best_mae = mae
        best_epoch = epoch
        best_save_path = os.path.join(save_path, 'Net_epoch_best.pth')
        torch.save(model.state_dict(), best_save_path)
        print(f'New best model (epoch {epoch}, MAE {mae:.6f}) saved -> {best_save_path}')

    logging.info(f'[Val Info]: Epoch:{epoch} MAE:{mae:.6f} bestEpoch:{best_epoch} bestMAE:{best_mae:.6f}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,default='',help='the path to save model and log')
    opt = parser.parse_args()



    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print('USE GPU 2,3')
    cudnn.benchmark = True

    # build the model
    device_ids = [0, 1]
    # network
    model = torch.nn.DataParallel(Network_interFA_noSpade_noEdge_ODE_slot_channel4(channels=96), device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=0)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)


    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0
    
    # learning rate schedule
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print("Start train...")
    for epoch in range(1, opt.epoch):

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)

