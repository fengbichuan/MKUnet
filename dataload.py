import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os.path

# ImageNet 均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None,
                 image_suffixes=('.jpg', '.png'),
                 mask_suffixes=('.png',)):  # <-- 修改点 1: 同样改为元组

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_suffixes = image_suffixes
        self.mask_suffixes = mask_suffixes  # <-- 修改点 2: 存储掩码后缀

        # --- 扫描图像 (逻辑同上) ---
        image_filenames = [f for f in os.listdir(self.image_dir)
                           if f.endswith(self.image_suffixes)]
        self.image_map = {os.path.splitext(f)[0]: f for f in image_filenames}
        image_basenames = set(self.image_map.keys())

        # --- 修改点 3: 使用相同逻辑扫描Mask ---
        mask_filenames = [f for f in os.listdir(self.mask_dir)
                          if f.endswith(self.mask_suffixes)]
        # 创建 mask basename -> filename 的映射
        self.mask_map = {os.path.splitext(f)[0]: f for f in mask_filenames}
        mask_basenames = set(self.mask_map.keys())
        # --- 结束修改 ---

        # 找到交集
        valid_basenames = image_basenames.intersection(mask_basenames)
        self.valid_files = sorted(list(valid_basenames))

        if len(self.valid_files) == 0:
            print(f"警告: 在 {image_dir} 和 {mask_dir} 中没有找到匹配的文件。")
            # 更新警告信息
            print(f"  (查找图像: *{self.image_suffixes} 和 查找掩码: *{self.mask_suffixes})")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, index):
        basename = self.valid_files[index]

        # --- 修改点 4: 使用 map 查找图像和掩码的完整文件名 ---
        image_filename = self.image_map[basename]
        mask_filename = self.mask_map[basename]

        img_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)
        # --- 结束修改 ---

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # 预处理Mask (二值化)
        mask = (mask > 128).astype(np.uint8)

        # 应用数据增强 (Albumentations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            # 转换为 [1, H, W] 的 FloatTensor
        return image, mask.unsqueeze(0).float()


def get_loaders(data_path, batch_size, image_height, image_width, num_workers=4, pin_memory=True):
    """
    创建并返回训练和验证的 DataLoaders
    """
    # (路径定义不变，我只保留一组)
    TRAIN_IMG_DIR = os.path.join(data_path, "train/images")
    TRAIN_MASK_DIR = os.path.join(data_path, "train/masks")
    VAL_IMG_DIR = os.path.join(data_path, "val/images")
    VAL_MASK_DIR = os.path.join(data_path, "val/masks")

    # (数据增强不变)
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # --- 修改点 5: 定义图像和掩码的后缀 ---

    # 定义我们想要读取的图像后缀
    IMG_SUFFIXES = ('.jpg', '.png')
    # 定义我们想要读取的掩码后缀 (很可能你的 val mask 也是 .png)
    MASK_SUFFIXES = ('.png', '.jpg')  # 两个都支持，以防万一

    train_dataset = ISICDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
        image_suffixes=IMG_SUFFIXES,  # 传递元组
        mask_suffixes=MASK_SUFFIXES  # 传递元组
    )

    val_dataset = ISICDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=val_transform,
        image_suffixes=IMG_SUFFIXES,  # 传递元组
        mask_suffixes=MASK_SUFFIXES  # 传递元组
    )
    # --- 结束修改 ---

    # (DataLoader 的调用不变)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        persistent_workers=True
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    return train_loader, val_loader