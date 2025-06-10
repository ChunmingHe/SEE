import argparse
import os
from random import random
from sam_refiner import sam_refiner
from PIL import Image
from torch import optim

from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.transforms as T
import random
from lib.Network import Network
import yaml
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import zip_longest
import datasets

import utils
from datetime import datetime
from statistics import mean
import logging
import torch
import torch.distributed as dist
from util.data_val import get_loader, test_dataset
from util.utils import clip_gradient, adjust_lr, get_coef,cal_ual
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

# def make_data_loader(spec, tag=''):
#     if spec is None:
#         return None
#
#     dataset = datasets.make(spec['dataset'])
#     dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
#     if local_rank == 0:
#         log('{} dataset: size={}'.format(tag, len(dataset)))
#         for k, v in dataset[0].items():
#             log('  {}: shape={}'.format(k, tuple(v.shape)))
#
#     # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#     loader = DataLoader(dataset, batch_size=spec['batch_size'],
#         shuffle=False, num_workers=8, pin_memory=True)
#     return loader


# def make_data_loaders():
#     train_loader = make_data_loader(config.get('train_dataset'), tag='train')
#     val_loader = make_data_loader(config.get('val_dataset'), tag='val')
#     return train_loader, val_loader
#
# def make_datau_loaders():
#     train_loader = make_data_loader(config.get('trainu_dataset'), tag='train')
#     val_loader = make_data_loader(config.get('val_dataset'), tag='val')
#     return train_loader, val_loader

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

# def update_ema(model_teacher, model, alpha_teacher, iteration):
#     with torch.no_grad():
#         alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
#         for ema_param, param in zip(model_teacher.parameters(), model.parameters()):
#             ema_param.data = alpha_teacher * ema_param.data + (1.0 - alpha_teacher) * param.data

# def prepare_training():
#     if config.get('resume') is not None:
#         model = models.make(config['model']).cuda()
#         optimizer = utils.make_optimizer(
#             model.parameters(), config['optimizer'])
#         epoch_start = config.get('resume') + 1
#     else:
#         model = models.make(config['model']).cuda()
#         # model2 =
#         optimizer = utils.make_optimizer(
#             model.parameters(), config['optimizer'])
#         epoch_start = 1
#     max_epoch = config.get('epoch_max')
#     lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
#     # if local_rank == 0:
#     #     log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
#     return model, optimizer, epoch_start, lr_scheduler


# 定义图像增强操作
class StochasticAugmentation:
    def __init__(self):
        self.flip = [T.RandomHorizontalFlip(p=1), T.RandomVerticalFlip(p=1)]
        self.rotations = [T.RandomRotation(degrees=0), T.RandomRotation(degrees=90),
                          T.RandomRotation(degrees=180), T.RandomRotation(degrees=270)]
        # self.scalings = [  # 缩放 0.5x
        #                  T.Resize((384, 384)),  # 不缩放
        #                  ]  # 缩放 2x
        self.scalings = [T.Resize((int(384 * 0.5), int(384 * 0.5))),  # 缩放 0.5x
                         T.Resize((384, 384)),  # 不缩放
                         ]  # 缩放 2x

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


# 定义多增强结果融合的过程
def multi_augmentation_fusion(image, teacher_model, gts2, k=12):
    augmented_images = []
    augmented_masks = []
    stochastic_augmentation = StochasticAugmentation()
    # 执行K次随机增强
    for _ in range(k):
        augmented_image = image
        augmented_image = stochastic_augmentation.apply(augmented_image)

        augmented_images.append(augmented_image)

        # 通过教师模型生成分割掩膜
        # augmented_image_tensor = augmented_image.permute(2,1,0).cpu()
        # augmented_image_tensor = np.array(augmented_image_tensor).astype(np.uint8)
        # augmented_image_tensor = torch.from_numpy(augmented_image_tensor).cuda()


        mask= teacher_model(augmented_image)[4].sigmoid()
        mask = (mask - mask.min()) / (
                    mask.max() - mask.min() + 1e-8)

        mask = torch.tensor(mask) if isinstance(mask, np.ndarray) else mask
        augmented_masks.append(mask)

    # 对增强后的掩膜进行逆变换使其与原图大小对齐
    original_size = image.size()
    height, width = original_size[-2], original_size[-1]
    resized_masks = [
        F.interpolate(mask, size=(height, width), mode='bilinear', align_corners=False)
        for mask in augmented_masks]
    mae1 = torch.mean(torch.abs(resized_masks[0] - gts2))
    mae2 = torch.mean(torch.abs(resized_masks[1] - gts2))

    augmented_images = [
        F.interpolate(image, size=(height, width), mode='bilinear', align_corners=False)
        for image in augmented_images]
    # 对所有生成的掩膜进行融合（通过平均）
    fused_mask = torch.mean(torch.cat(resized_masks, dim=1), dim=1,keepdim=True)
    mae3 = torch.mean(torch.abs(fused_mask - gts2))
    # 对K个掩膜进行平均

    # 输出融合后的掩膜
    return fused_mask,augmented_images

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
        # 打印 pseudo_label 的最小值和最大值
        # print(f"Min pseudo_label: {torch.min(pseudo_label).item():.4f}")
        # print(f"Max pseudo_label: {torch.max(pseudo_label).item():.4f}")
        entropy_map = self.compute_entropy(pseudo_label)
        # 打印最大值和最小值
        entropy_min = torch.min(entropy_map)
        entropy_max = torch.max(entropy_map)
        previous_label_min = torch.min(previous_label)
        previous_label_max = torch.max(previous_label)
        print(f"Minimum entropy: {entropy_min.item():.4f}")
        print(f"Maximum entropy: {entropy_max.item():.4f}")
        high_uncertainty_pixels = (entropy_map > 0.9).float()
        Ua = high_uncertainty_pixels.mean()

        low_uncertainty_pixels = (entropy_map <= 0.9).float()
        # foreground_mask = (pseudo_label > 0.5).float()
        # low_uncertainty_foreground = low_uncertainty_pixels * foreground_mask

        relative_uncertainty = high_uncertainty_pixels.sum() / (low_uncertainty_pixels.sum()+1e-10)
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


def compute_weighted_map(pseudo_label,  pseudo_label_pool, Ua, Ur, Ua_threshold=0.1, Ur_threshold=0.5):
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
            current_sam = sam_model_registry["vit_b"](checkpoint="/home/zrh/segment-anything-main/sam_vit_b_01ec64.pth").cuda()
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

        ual_coef2 = get_coef(iter_percentage= i / total_step, method='cos')
        ual_loss2 = cal_ual(seg_logits=predictions, seg_gts=pseudo_label)
        ual_loss2 *= ual_coef2
        # loss_init = structure_loss(preds[0], gts) * 0.0625 + structure_loss(preds[1], gts) * 0.125 + structure_loss(
        #     preds[2], gts) * 0.25 + \
        #             structure_loss(preds[3], gts) * 0.5
        loss_final2 = structure_loss(predictions, pseudo_label)
        # loss_edge = dice_loss(predictions, edges) * 0.125 + dice_loss(predictions, edges) * 0.25 + \
        #             dice_loss(predictions, edges) * 0.5
        # sup_loss = loss_init + loss_final + 2 * ual_loss + loss_edge

        # # 计算交叉熵损失
        # ce_loss = Lce(predictions, pseudo_label)
        #
        # # 计算IoU损失
        # iou_loss = LIoU(predictions, pseudo_label)
        #
        weighted_ce_loss = (weighted_maps[b] * ual_loss2).mean()  # 使用 .mean() 来将每个元素的损失合并为一个标量
        weighted_iou_loss = (weighted_maps[b] * loss_final2).mean()  # 同样地，对 IoU 损失应用加权并求平均

        # 计算总损失
        loss_all += weighted_ce_loss +  weighted_iou_loss

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

def train(train_loader, trainu_loader, model2, args, optimizer, epoch, writer, pseudo_label_pool, model3, previous_predictions,model_small):
    global step
    model2.train()
    model3.eval()
    loss_all = 0
    epoch_step = 0
    # step = 0
    # pseudo_label_pool = PseudoLabelPool(B=3)

    for i, ((images, gts, edges), (images2, gts2, edges2)) in enumerate(zip_longest(train_loader, trainu_loader, fillvalue=(None, None, None)), start=1):
        # if i >20:
        #     break
        total_step = len(trainu_loader)
        # 将数据移到 GPU
        # images, gts, edges = images.cuda(), gts.cuda(), edges.cuda()
        images2, gts2, edges2 = images2.cuda(), gts2.cuda(), edges2.cuda()

        if images is None or images.shape[0] == 1:
            sup_loss = 0

        else:
            images, gts, edges = images.cuda(), gts.cuda(), edges.cuda()
            preds = model2(images)
            ual_coef = get_coef(iter_percentage=i / total_step, method='cos')
            ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
            ual_loss *= ual_coef
            loss_init = structure_loss(preds[0], gts) * 0.0625 + structure_loss(preds[1], gts) * 0.125 + structure_loss(
                preds[2], gts) * 0.25 + \
                        structure_loss(preds[3], gts) * 0.5
            loss_final = structure_loss(preds[4], gts)
            loss_edge = dice_loss(preds[6], edges) * 0.125 + dice_loss(preds[7], edges) * 0.25 + \
                        dice_loss(preds[8], edges) * 0.5
            sup_loss = loss_init + loss_final + 2 * ual_loss + loss_edge
        if images2.size(0) == 1:
            # 你也可以直接 continue，跳过整个当前 iteration
            continue
        preds2 = model2(images2)
        preds2_mask = preds2[4].sigmoid()
        preds2_mask = (preds2_mask - preds2_mask.min()) / (
                    preds2_mask.max() - preds2_mask.min() + 1e-8)


        fused_mask, augmented_images = multi_augmentation_fusion(images2, model3, gts2, k=3)
        min_value = fused_mask.min()
        max_value = fused_mask.max()
        # all_preds2_gt = []
        #
        # # 外层循环遍历 augmented_images
        # for augmented_image in augmented_images:
        #     # 获取 augmented_image 对应的 images2（可以根据你的数据结构进行调整）
        #     sam_masks_logits = model_small(augmented_image)[4]
        #     sam_masks_logits = sam_masks_logits.sigmoid()
        #     min_value = sam_masks_logits.min()
        #     max_value = sam_masks_logits.max()
        #     sam_masks_logits = (sam_masks_logits - sam_masks_logits.min()) / (sam_masks_logits.max() - sam_masks_logits.min() + 1e-8)
        #     min_value = sam_masks_logits.min()
        #     max_value = sam_masks_logits.max()
        #     all_preds2_gt.append(sam_masks_logits)
        #     mae = torch.mean(torch.abs(sam_masks_logits - gts2))
        # average_preds2_gt = torch.mean(torch.cat(all_preds2_gt, dim=1).float(), dim=1, keepdim=True)

        # 例如，将输入值缩放到 [0, 1] 之间
        # preds2_gt = (average_preds2_gt - average_preds2_gt.min()) / (average_preds2_gt.max() - average_preds2_gt.min())
        pseudo_label_new = fused_mask
        mae = torch.mean(torch.abs(pseudo_label_new - gts2))
        print("MAE:", mae.item())
        # 获取 preds2_gt 的最小值和最大值
        min_value = pseudo_label_new.min()
        max_value =pseudo_label_new.max()

        # print(f"Minimum value: {min_value}")
        # print(f"Maximum value: {max_value}")

        # previous_label = torch.sigmoid(preds2[4])
        # min_value = preds2[4].min()
        # max_value = preds2[4].max()
        # 获取上一轮的预测结果。对于第一轮，使用当前预测初始化
        if epoch == 0 or i not in previous_predictions:
            previous_label = preds2_mask
        else:
            previous_label = previous_predictions[i]

        # 更新上一轮预测存储：用于下一轮比较
        previous_predictions[i] = preds2_mask
        Ua, Ur, Ud = pseudo_label_pool.compute_uncertainty_metrics(pseudo_label_new, previous_label)

        weighted_map = compute_weighted_map(pseudo_label_new, pseudo_label_pool, Ua, Ur)


        # Update pseudo-label pool
        pseudo_label_pool.add_or_replace(image_id=i, pseudo_label=pseudo_label_new, previous_label=previous_label, metrics=(Ua, Ur, Ud), weighted_map=weighted_map)


        # Get the best pseudo-labels from the pool
        best_pseudo_labels, weighted_maps = pseudo_label_pool.get_best_labels(i)

        # # Pixel-level weighting for selected pseudo-labels
        # if image_selection(pseudo_label_new, pseudo_label_pool, i):
        #     weighted_map = pixel_level_weighting(pseudo_label_new,pseudo_label_pool, i)
        # else:
        #     weighted_map = torch.ones_like(pseudo_label_new)  # Use all ones if not selected

        unsup_loss = compute_loss(i, total_step, predictions=preds2_mask, pseudo_labels=best_pseudo_labels, weighted_maps=weighted_maps, B=3)

        # if i >=1:
        #     i = 1
        #     break





        # print(preds2_gt.shape)
        # ual_coef2 = get_coef(iter_percentage=i / total_step, method='cos')
        # ual_loss2 = cal_ual(seg_logits=preds2[4], seg_gts=preds2_gt)
        # ual_loss2 *= ual_coef2
        # loss_init2 = structure_loss(preds2[0],  preds2_gt) * 0.0625 + structure_loss(preds2[1],  preds2_gt) * 0.125 + structure_loss(
        #     preds2[2],  preds2_gt) * 0.25 + \
        #             structure_loss(preds2[3], preds2_gt) * 0.5
        # loss_final2 = structure_loss(preds2[4],  preds2_gt)
        #
        # unsup_loss = loss_init2 + loss_final2 + 2 * ual_loss2

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
            image, gt,  name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            result = model(image)

            res = F.upsample(result[0], size=gt.shape, mode='bilinear', align_corners=False)
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

    train_loader = get_loader(image_root= args.train_root + 'Imgs/',
                              gt_root= args.train_root + 'GT/',
                              edge_root= args.train_root + 'Edge/',
                              batchsize= args.batchsize,
                              trainsize= args.trainsize,
                              num_workers=0)

    trainu_loader = get_loader(image_root= args.trainu_root + 'Imgs/',
                              gt_root= args.trainu_root + 'GT/',
                              edge_root= args.trainu_root + 'Edge/',
                              batchsize= args.batchsize,
                              trainsize= args.trainsize,
                              num_workers=0)

    # total_step = len(train_loader)

    val_loader = test_dataset(image_root= args.val_root + 'Imgs/',
                              gt_root= args.val_root + 'GT/',
                              testsize= args.trainsize)

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    # model, optimizer, epoch_start, lr_scheduler = prepare_training()
    # model.optimizer = optimizer


    # sam = sam_model_registry["vit_b"](checkpoint="/home/zrh/segment-anything-main/sam_vit_b_01ec64.pth").cuda()
    # predictor = SamPredictor(sam)
    # for p in sam.parameters():
    #     p.requires_grad = False



    model_s = Network().cuda()
    model_t = Network().cuda()
    model_small = Network().cuda()

    # # 加载 checkpoint
    # checkpoint = torch.load('/home/zrh/FEDER/checkpoint.pth', map_location='cpu')
    #
    # # 创建一个新的 state_dict，将所有 key 中的 "module." 去掉
    # new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    #
    # # 加载修改后的 state_dict
    # model3.load_state_dict(new_state_dict)
    model_t.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('/home/zrh/FEDER/checkpoint.pth').items()})
    model_small.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('/home/zrh/FEDER/checkpoint.pth').items()})

    optimizer = torch.optim.Adam(model_s.parameters(), args.lr)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    # lr_scheduler = CosineAnnealingLR(model2.optimizer, args.epoch, eta_min=config.get('lr_min'))
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[args.local_rank],
    #     output_device=args.local_rank,
    #     find_unused_parameters=True,
    #     broadcast_buffers=False
    # )
    # model = model.module

    # sam_checkpoint = torch.load(config['sam_checkpoint'])
    # model.load_state_dict(sam_checkpoint, strict=False)
    
    # for name, para in model.named_parameters():
    #     if "image_encoder" in name and "prompt_generator" not in name:
    #         para.requires_grad_(False)
    # if local_rank == 0:
    #     model_total_params = sum(p.numel() for p in model.parameters())
    #     model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    # epoch_max = config['epoch_max']
    # epoch_val = config.get('epoch_val')
    # max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    # timer = utils.Timer()

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
        writer.add_scalar('learning_rate',  cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format( cosine_schedule.get_lr()[0]))
        train(train_loader,trainu_loader, model_s, args, optimizer, epoch, writer,pseudo_label_pool, model_t, previous_predictions,model_small)
        val(val_loader,model_s, epoch, save_path, writer)



        # if local_rank == 0:
        #     log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        #     writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        #     log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        #     writer.add_scalars('loss', {'train G': train_loss_G}, epoch)
        #
        #     model_spec = config['model']
        #     model_spec['sd'] = model.state_dict()
        #     optimizer_spec = config['optimizer']
        #     optimizer_spec['sd'] = optimizer.state_dict()
        #
        #     save(config, model, save_path, 'last')
        #
        # if (epoch_val is not None) and (epoch % epoch_val == 0):
        #     result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
        #         eval_type=config.get('eval_type'))
        #
        #     if local_rank == 0:
        #         log_info.append('val: {}={:.4f}'.format(metric1, result1))
        #         writer.add_scalars(metric1, {'val': result1}, epoch)
        #         log_info.append('val: {}={:.4f}'.format(metric2, result2))
        #         writer.add_scalars(metric2, {'val': result2}, epoch)
        #         log_info.append('val: {}={:.4f}'.format(metric3, result3))
        #         writer.add_scalars(metric3, {'val': result3}, epoch)
        #         log_info.append('val: {}={:.4f}'.format(metric4, result4))
        #         writer.add_scalars(metric4, {'val': result4}, epoch)
        #
        #         if config['eval_type'] != 'ber':
        #             if result1 > max_val_v:
        #                 max_val_v = result1
        #                 save(config, model, save_path, 'best')
        #         else:
        #             if result3 < max_val_v:
        #                 max_val_v = result3
        #                 save(config, model, save_path, 'best')
        #
        #         t = timer.t()
        #         prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        #         t_epoch = utils.time_text(t - t_epoch_start)
        #         t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        #         log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
        #
        #         log(', '.join(log_info))
        #         writer.flush()


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
    parser.add_argument('--config', default="/home/zrh/SAM-Adapter-PyTorch-main/configs/cod-sam-vit-b.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_root', type=str, default='/home/zrh/FEDER-main/COD-TrainDataset/COD10K/',
                        help='the training rgb images root')
    parser.add_argument('--trainu_root', type=str, default='/home/zrh/FEDER-main/COD-TrainDataset/COD10K/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/home/zrh/FEDER-main/COD-TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
    parser.add_argument('--save_path', type=str, default='./snapshot/model_new/', help='the path to save model and log')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    args = parser.parse_args()

    step = 0
    best_mae = 1
    best_epoch = 0

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    # save_name = args.name
    # if save_name is None:
    #     save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    # if args.tag is not None:
    #     save_name += '_' + args.tag
    # save_path = os.path.join('./save', save_name)
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
