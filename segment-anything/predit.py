from segment_anything import sam_model_registry
from sam_refiner import sam_refiner
import torch
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import numpy as np
from PIL import Image

import torch
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
from segment_anything import SamPredictor, sam_model_registry

# 定义图像增强操作
class StochasticAugmentation:
    def __init__(self):
        self.flip = [T.RandomHorizontalFlip(p=1), T.RandomVerticalFlip(p=1)]
        self.rotations = [T.RandomRotation(degrees=0), T.RandomRotation(degrees=90),
                          T.RandomRotation(degrees=180), T.RandomRotation(degrees=270)]
        self.scalings = [T.Resize((int(256 * 0.5), int(256 * 0.5))),  # 缩放 0.5x
                         T.Resize((256, 256)),  # 不缩放
                         T.Resize((int(256 * 2.0), int(256 * 2.0)))]  # 缩放 2x

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
def multi_augmentation_fusion(image, teacher_model, k=2):
    augmented_images = []
    augmented_masks = []

    # 执行K次随机增强
    for _ in range(k):
        augmented_image = image
        augmented_image = stochastic_augmentation.apply(augmented_image)
        augmented_images.append(augmented_image)

        # 通过教师模型生成分割掩膜
        augmented_image_tensor = T.ToTensor()(augmented_image).permute(2,1,0)
        augmented_image_tensor = np.array(augmented_image_tensor)
        teacher_model.set_image(augmented_image_tensor)  # 设置单张图像
        mask, _, _ = teacher_model.predict()
        mask = torch.tensor(mask) if isinstance( mask, np.ndarray) else  mask
        augmented_masks.append(mask)

    # 对增强后的掩膜进行逆变换使其与原图大小对齐
    original_size = image.size
    height, width = original_size[-2], original_size[-1]
    resized_masks = [
        F.interpolate(mask.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
        for mask in augmented_masks]

    # 对所有生成的掩膜进行融合（通过平均）
    fused_mask = torch.mean(torch.cat(resized_masks, dim=0), dim=0)  # 对K个掩膜进行平均

    # 输出融合后的掩膜
    return fused_mask


# 加载图像
image_path = '/home/zrh/SAMRefiner/examples/2007_000256.jpg'
image = Image.open(image_path)

# 定义增强器和教师模型
stochastic_augmentation = StochasticAugmentation()
sam = sam_model_registry["vit_b"](checkpoint="/home/zrh/segment-anything-main/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

# 进行多次增强并融合
fused_mask = multi_augmentation_fusion(image, predictor, k=2)

# # 通过将融合结果转换为PIL图像查看最终的掩膜
# fused_result_mask_image = Image.fromarray((fused_result_mask * 255).astype(np.uint8), mode='L')
# fused_result_mask_image.show()
mask_path = '/home/zrh/SAMRefiner/examples/2007_000256_init_mask.png'
init_masks = np.asarray(Image.open(mask_path), dtype=np.uint8)
refined_masks = sam_refiner(image_path,
                            [init_masks],
                            sam)[0]

# print(refined_masks.shape)
#
# Image.fromarray(255 * refined_masks[0].astype(np.uint8)).save('/home/zrh/SAMRefiner/examples/2007_000256_refined_mask.png')