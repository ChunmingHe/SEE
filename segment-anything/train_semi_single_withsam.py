import argparse
import os
from random import random
from sam_refiner import sam_refiner
from PIL import Image
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.transforms as T
import random
from lib.Network import Network_interFA_noSpade_noEdge_ODE_slot_channel4
import yaml
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import zip_longest


import utils
from datetime import datetime
from statistics import mean
import logging
import torch
import torch.distributed as dist
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, get_coef, cal_ual

# torch.distributed.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
local_rank = 0


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


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4

# 定义图像增强操作
class StochasticAugmentation:
    def __init__(self):
        self.flip = [T.RandomHorizontalFlip(p=1), T.RandomVerticalFlip(p=1)]
        self.rotations = [T.RandomRotation(degrees=0), T.RandomRotation(degrees=90),
                          T.RandomRotation(degrees=180), T.RandomRotation(degrees=270)]
        self.scalings = [T.Resize((int(1024 * 0.5), int(1024 * 0.5))),
                         T.Resize((1024 , 1024 )),
                         T.Resize((int(1024 * 2), int(1024 *2)))]

    def apply(self, image):
        # 随机选择增强操作
        flip_type = random.choice(self.flip)
        rotation_type = random.choice(self.rotations)
        scaling_type = random.choice(self.scalings)

        # 顺序应用增强
        augmented_image = flip_type(image)
        augmented_image = rotation_type(augmented_image)
        augmented_image = scaling_type(augmented_image)

        return augmented_image


def multi_augmentation_fusion(image, teacher_model, gts2, k=12, device=torch.device('cpu')):
    """
    image: 已搬到 GPU 的 Tensor
    teacher_model: EMA 模型，已在 eval() 下
    gts2: 已搬到 GPU 的标注
    返回：
        fused_mask: 在 CPU 上的 Tensor [B,1,H,W]
        augmented_images: 在 CPU 上的 list of Tensor [B,3,H,W]
    """
    augmented_masks = []
    augmented_images = []
    stochastic_aug = StochasticAugmentation()

    # K 次随机增强
    for _ in range(k):
        aug = stochastic_aug.apply(image)  # on GPU
        augmented_images.append(aug.cpu())

        # 前向生成 mask
        with torch.cuda.amp.autocast():
            m = teacher_model(aug)[4].sigmoid()
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        augmented_masks.append(m.cpu())

    # 对齐到原始大小
    _, _, H, W = image.shape
    resized_masks = [
        F.interpolate(m.to(device), size=(H, W), mode='bilinear', align_corners=False)
        .cpu() for m in augmented_masks
    ]
    resized_images = [
        F.interpolate(aug.to(device), size=(H, W), mode='bilinear', align_corners=False)
        .cpu() for aug in augmented_images
    ]

    # 平均融合
    fused_mask = torch.mean(torch.cat(resized_masks, dim=1), dim=1, keepdim=True)
    return fused_mask, resized_images


def compute_entropy(self, pseudo_label):
    eps = 1e-10  # Prevent log(0)
    return -pseudo_label * torch.log2(pseudo_label + eps) - (1 - pseudo_label) * torch.log2(1 - pseudo_label + eps)


class PseudoLabelPool:
    def __init__(self, B=3):
        """
        初始化伪标签池 B，用于存储每张图像的最优伪标签
        """
        self.B = B  # 池大小
        self.pool = {}  # 每个图像的伪标签池字典
        self.metrics_pool = {}  # 每个图像的度量指标字典
        self.weighted_map_pool = {}  # 每个图像的加权图字典

    def compute_entropy(self, pseudo_label):
        eps = 1e-10  # Prevent log(0)
        return -pseudo_label * torch.log2(pseudo_label + eps) - (1 - pseudo_label) * torch.log2(1 - pseudo_label + eps)

    def evaluate_uncertainty(self, pseudo_label, previous_label):
        a = torch.min(torch.abs(pseudo_label - previous_label))
        b = torch.max(torch.abs(pseudo_label - previous_label))
        residual_uncertainty = self.compute_entropy(torch.abs(pseudo_label - previous_label))
        return residual_uncertainty

    def add_or_replace(self, image_id, pseudo_label, previous_label, metrics, weighted_map):
        """
        评估新生成的伪标签与池中现有伪标签的质量，决定是否替换。
        metrics 包含 Ua, Ur 和 Ud
        """
        Ua, Ur, Ud = metrics
        pseudo_label = pseudo_label.detach()
        weighted_map = weighted_map.detach()
        # 检查是否已经为该图像创建了池，如果没有则创建
        if image_id not in self.pool:
            self.pool[image_id] = []  # 为新图像初始化伪标签池
            self.metrics_pool[image_id] = []
            self.weighted_map_pool[image_id] = []  # 初始化加权图池

        # 如果池中尚未满，就直接添加新伪标签和加权图
        if len(self.pool[image_id]) < self.B:
            self.pool[image_id].append(pseudo_label)
            self.metrics_pool[image_id].append((Ua, Ur, Ud))  # 存储度量指标
            self.weighted_map_pool[image_id].append(weighted_map)  # 存储加权图
            # print(f"Length of weighted_map_pool for image_id {image_id}: {len(self.weighted_map_pool[image_id])}")
            return True

        # 如果池已满，比较新伪标签与池中最优伪标签的度量
        for idx, (best_label, (best_Ua, best_Ur, best_Ud)) in enumerate(
                zip(self.pool[image_id], self.metrics_pool[image_id])):
            count_better_metrics = sum([Ua < best_Ua, Ur < best_Ur, Ud < best_Ud])

            if count_better_metrics >= 2:
                # 如果新伪标签在至少两个度量上优于池中的伪标签，则替换
                self.pool[image_id][idx] = pseudo_label
                self.metrics_pool[image_id][idx] = (Ua, Ur, Ud)  # 更新度量
                self.weighted_map_pool[image_id][idx] = weighted_map  # 更新加权图
                print(f"Length of weighted_map_pool for image_id {image_id}: {len(self.weighted_map_pool[image_id])}")
                return True

        # 如果没有任何替换发生，返回False
        return False

    def get_best_labels(self, image_id):
        """
        获取指定图像的最优伪标签
        """
        return self.pool.get(image_id, []), self.weighted_map_pool.get(image_id, [])

    def compute_uncertainty_metrics(self, pseudo_label, previous_label):

        entropy_map = self.compute_entropy(pseudo_label)
        high_uncertainty_pixels = (entropy_map > 0.9).float()
        Ua = high_uncertainty_pixels.mean()

        low_uncertainty_pixels = (entropy_map <= 0.9).float()


        relative_uncertainty = high_uncertainty_pixels.sum() / (low_uncertainty_pixels.sum() + 1e-10)
        Ur = relative_uncertainty

        residual_uncertainty = self.evaluate_uncertainty(pseudo_label, previous_label)
        Ud = residual_uncertainty.mean()

        return Ua, Ur, Ud


def image_selection_indicator(Ua, Ur, Ua_threshold=0.1, Ur_threshold=0.5):
    """
    选择伪标签的指示函数 I。
    如果 Ua 和 Ur 都小于阈值，则 I=1，否则 I=0。
    """
    if Ua < Ua_threshold and Ur < Ur_threshold:
        return 1  # 伪标签可靠，选择该伪标签
    else:
        return 0  # 伪标签不可靠，丢弃该伪标签


def compute_weighted_map(pseudo_label, pseudo_label_pool, Ua, Ur, Ua_threshold=0.1, Ur_threshold=0.5):
    """
    计算加权图，首先根据选择标准过滤伪标签，然后计算不确定性加权。
    """
    # 计算伪标签的不确定性图
    entropy_map = pseudo_label_pool.compute_entropy(pseudo_label)

    # 计算指示函数 I，决定是否选择该伪标签
    I = image_selection_indicator(Ua, Ur, Ua_threshold, Ur_threshold)

    # 计算最终的加权图
    weighted_map = (1 - entropy_map) * I
    return weighted_map


def manage_sam_model(should_use, current_sam):
    """
    根据 should_use 参数来管理 SAM 模型。
    如果 should_use 为 True 且 current_sam 为 None，则加载模型；
    如果 should_use 为 False 且 current_sam 不为 None，则释放模型。
    返回最新的 current_sam。
    """
    if should_use:
        if current_sam is None:
            current_sam = sam_model_registry["vit_b"](
                checkpoint="/hpc/home/ch594/SAM-Adapter-PyTorch/sam_vit_b_01ec64.pth").cuda()
            for p in current_sam.parameters():
                p.requires_grad = False
        return current_sam
    else:
        if current_sam is not None:
            del current_sam
            torch.cuda.empty_cache()
        return None


# 更新训练损失部分
def compute_loss(i, total_step, predictions, pseudo_labels, weighted_maps, B=3):
    loss_all = 0
    for b in range(len(weighted_maps)):
        # 获取伪标签池中的第b个伪标签
        pseudo_label = pseudo_labels[b]

        ual_coef2 = get_coef(iter_percentage=i / total_step, method='cos')
        ual_loss2 = cal_ual(seg_logits=predictions, seg_gts=pseudo_label)
        ual_loss2 *= ual_coef2

        loss_final2 = structure_loss(predictions, pseudo_label)

        weighted_ce_loss = (weighted_maps[b] * ual_loss2).mean()  # 使用 .mean() 来将每个元素的损失合并为一个标量
        weighted_iou_loss = (weighted_maps[b] * loss_final2).mean()  # 同样地，对 IoU 损失应用加权并求平均

        # 计算总损失
        loss_all += weighted_ce_loss + weighted_iou_loss

    return loss_all / B  # 返回平均损失


def update_ema(model_teacher, model_student, alpha=0.99):
    """
    更新教师模型的 EMA 权重。

    Parameters:
    model_teacher (nn.Module): 教师模型
    model_student (nn.Module): 学生模型
    alpha (float): EMA 衰减系数，通常设置为接近 1（例如 0.99）
    """
    # 获取教师模型和学生模型的所有参数
    teacher_params = model_teacher.parameters()
    student_params = model_student.parameters()

    # 更新每一层的参数
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_params, student_params):
            # 使用 EMA 更新教师模型的参数
            teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data


def train(train_loader, trainu_loader, model2, args, optimizer, epoch, writer, pseudo_label_pool, model3,
          previous_predictions):
    global step
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model2.train()
    model3.eval()
    loss_all = 0
    epoch_step = 0

    # 只在需要时加载 SAM 到 GPU
    sam = manage_sam_model(True, None).to(device)

    for i, ((images, gts, edges), (images2, gts2, edges2)) \
            in enumerate(zip_longest(train_loader, trainu_loader, fillvalue=(None, None, None)), start=1):

        total_step = len(trainu_loader)

        # —— 有监督部分 ——
        if images is None or images.size(0) <= 1:
            sup_loss = 0
        else:
            # 仅在前向时搬到 GPU
            images = images.to(device)
            gts = gts.to(device)
            edges = edges.to(device)

            preds = model2(images)

            ual_coef = get_coef(iter_percentage=i / total_step, method='cos')
            ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts) * ual_coef

            loss_init = (structure_loss(preds[0], gts) * 0.0625
                         + structure_loss(preds[1], gts) * 0.125
                         + structure_loss(preds[2], gts) * 0.25
                         + structure_loss(preds[3], gts) * 0.5).cpu()
            loss_final = structure_loss(preds[4], gts).cpu()


            sup_loss = loss_init + loss_final + 2 * ual_loss

            # 计算完毕后，释放显存，搬回 CPU
            images, gts, edges = images.cpu(), gts.cpu(), edges.cpu()

        if images2.size(0) == 1:
            unsup_loss = 0
        else:

            # 前向时搬到 GPU
            images2_gpu = images2.to(device)
            gts2_gpu = gts2.to(device)


            preds2 = model2(images2_gpu)
            preds2_mask = preds2[4].sigmoid()
            preds2_mask = (preds2_mask - preds2_mask.min()) / (
                        preds2_mask.max() - preds2_mask.min() + 1e-8)

            # multi-augmentation 融合（内部做了按需搬运）
            fused_mask, augmented_images = multi_augmentation_fusion(
                images2_gpu, model3, gts2_gpu, k=3, device=device
            )

            # 搬回 CPU，后续处理全在 CPU
            images2, gts2 = images2.cpu(), gts2.cpu()
            fused_mask = fused_mask.cpu()
            augmented_images = [img.cpu() for img in augmented_images]

            # 生成多组伪标签并取平均
            all_preds2_gt = []
            for aug_img in augmented_images:
                processed = []
                for j in range(aug_img.size(0)):
                    # 只有这里需要用到 GPU，SAM-refiner
                    img_j = aug_img[j].to(device)
                    mask_j = fused_mask[j].squeeze(0).detach().cpu().numpy()


                    with torch.no_grad():
                        _, logits = sam_refiner(img_j, [mask_j], sam)
                    sam_logits = logits.sigmoid()
                    sam_logits = (sam_logits - sam_logits.min()) / (sam_logits.max() - sam_logits.min() + 1e-8)

                    processed.append(torch.tensor(sam_logits))
                    # 及时释放
                    del logits, sam_logits
                    torch.cuda.empty_cache()

                preds2_gt = torch.stack(processed).to(device)
                preds2_gt = F.interpolate(preds2_gt, size=(1024, 1024),
                                          mode='bilinear', align_corners=False)

                all_preds2_gt.append(preds2_gt)
                del processed, preds2_gt
                torch.cuda.empty_cache()

            average_preds2_gt = torch.mean(torch.cat(all_preds2_gt, dim=1).float(), dim=1, keepdim=True)
            del all_preds2_gt
            torch.cuda.empty_cache()

            pseudo_label_new = average_preds2_gt.cpu()
            # 获取上一轮的预测结果。对于第一轮，使用当前预测初始化
            if epoch == 0 or i not in previous_predictions:
                previous_label = preds2_mask
            else:
                previous_label = previous_predictions[i]

            # 更新上一轮预测存储：用于下一轮比较
            previous_predictions[i] = preds2_mask
            Ua, Ur, Ud = pseudo_label_pool.compute_uncertainty_metrics(pseudo_label_new, previous_label.cpu())

            weighted_map = compute_weighted_map(pseudo_label_new, pseudo_label_pool, Ua, Ur)

            # Update pseudo-label pool
            pseudo_label_pool.add_or_replace(image_id=i, pseudo_label=pseudo_label_new, previous_label=previous_label,
                                             metrics=(Ua, Ur, Ud), weighted_map=weighted_map)

            # Get the best pseudo-labels from the pool
            best_pseudo_labels, weighted_maps = pseudo_label_pool.get_best_labels(i)



            unsup_loss = compute_loss(i, total_step, predictions=preds2_mask.cpu(), pseudo_labels=best_pseudo_labels,
                                      weighted_maps=weighted_maps, B=3).cpu()


        loss = sup_loss + unsup_loss
        optimizer.zero_grad()
        if not isinstance(loss, int):
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
        else:
            print("loss 为 int 类型，跳过反向传播")

        step += 1
        epoch_step += 1
        # 更新教师模型
        update_ema(model3, model2, alpha=0.99)

        # 累加 loss 时检查类型
        if isinstance(loss, int):
            loss_all += loss
        else:
            loss_all += loss.data

        if i % 20 == 0 or i == total_step or i == 1:
            if isinstance(unsup_loss, int):
                # 如果 unsup_loss 为 0，只打印 Total_loss
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}]'.
                    format(datetime.now(), epoch, args.epoch, i, total_step))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}]'.
                    format(epoch, args.epoch, i, total_step))
                writer.add_scalars('Loss_Statistics',
                                   {'loss_all': loss.data if not isinstance(loss, int) else loss},
                                   global_step=step)
            else:
                if isinstance(sup_loss, int):
                    print(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss2: {:0.4f} '.
                        format(datetime.now(), epoch, args.epoch, i, total_step, loss.data, unsup_loss.data))
                    logging.info(
                        '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss2: {:0.4f} '.
                        format(epoch, args.epoch, i, total_step, loss.data, unsup_loss.data))
                    writer.add_scalars('Loss_Statistics',
                                       {'unsup_loss': unsup_loss.data, 'loss_all': loss.data},
                                       global_step=step)
                else:
                    print(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                        format(datetime.now(), epoch, args.epoch, i, total_step, loss.data, sup_loss.data,
                               unsup_loss.data))
                    logging.info(
                        '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} '.
                        format(epoch, args.epoch, i, total_step, loss.data, sup_loss.data, unsup_loss.data))
                    writer.add_scalars('Loss_Statistics',
                                       {'sup_loss': sup_loss.data, 'unsup_loss': unsup_loss.data,
                                        'loss_all': loss.data},
                                       global_step=step)
        # # 清理中间变量
        # del images, gts, edges, images2, gts2, edges2, preds, preds2
        # torch.cuda.empty_cache()
    epoch_step = 1
    loss_all /= epoch_step
    logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, args.epoch, loss_all))
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

    # for batch in train_loader:
    #     for k, v in batch.items():
    #         batch[k] = v.to('cuda')  # 将数据加载到GPU
    #     inp = batch['inp']
    #     gt = batch['gt']
    #     model.set_input(inp, gt)
    #     model.optimize_parameters()
    #
    #     # 在单卡下直接使用 loss_G
    #     batch_loss = model.loss_G.item()  # 获取损失值
    #     loss_list.append(batch_loss)  # 将损失值添加到列表

    # loss = [i.item() for i in loss_list]
    # 使用完后释放SAM

    return loss_all

    # 记得在每轮末尾清理 SAM
    manage_sam_model(False, sam)
    del sam
    torch.cuda.empty_cache()
    return loss_all


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        mae_sum_edge = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            result = model(image)

            res = F.upsample(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


def main(config_, save_path, args):
    global config, step
    step = 0
    config = config_
    # log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # train_loader, val_loader = make_data_loaders()
    # trainu_loader, val_loader = make_datau_loaders()

    train_loader = get_loader(image_root=args.train_root + 'Imgs/',
                              gt_root=args.train_root + 'GT/',
                              edge_root=args.train_root + 'Edge/',
                              batchsize=args.batchsize,
                              trainsize=args.trainsize,
                              num_workers=0)

    trainu_loader = get_loader(image_root=args.trainu_root + 'Imgs/',
                               gt_root=args.trainu_root + 'GT/',
                               edge_root=args.trainu_root + 'Edge/',
                               batchsize=args.batchsize,
                               trainsize=args.trainsize,
                               num_workers=0)

    # total_step = len(train_loader)

    val_loader = test_dataset(image_root=args.val_root + 'Imgs/',
                              gt_root=args.val_root + 'GT/',
                              testsize=args.trainsize)

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }



    model_s = Network_interFA_noSpade_noEdge_ODE_slot_channel4().cuda()
    model_t = Network_interFA_noSpade_noEdge_ODE_slot_channel4().cuda()


    model_t.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('/hpc/home/ch594/snapshot/model_new/Net_epoch_best.pth').items()})

    optimizer = torch.optim.Adam(model_s.parameters(), args.lr)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)


    # step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    pseudo_label_pool = PseudoLabelPool(B=3)
    previous_predictions = {}
    for epoch in range(1, args.epoch):
        # train_loader.sampler.set_epoch(epoch)
        # t_epoch_start = timer.t()
        cur_lr = adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        train(train_loader, trainu_loader, model_s, args, optimizer, epoch, writer, pseudo_label_pool, model_t,
              previous_predictions)
        val(val_loader, model_s, epoch, save_path, writer)




def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_root', type=str, default='',
                        help='the training rgb images root')
    parser.add_argument('--trainu_root', type=str, default='',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='',
                        help='the test rgb images root')
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
    parser.add_argument('--save_path', type=str, default='./see/model_new/', help='the path to save model and log')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    args = parser.parse_args()

    step = 0
    best_mae = 1
    best_epoch = 0

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')


    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(args.epoch, args.lr, args.batchsize, args.trainsize, args.clip,
                                                         args.decay_rate, args.load, save_path, args.decay_epoch))

    main(config, save_path, args=args)
