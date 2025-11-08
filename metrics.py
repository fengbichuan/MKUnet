import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class DiceLoss(nn.Module):
    """
    Dice Loss 损失函数, 适用于二分类 (1个输出通道)
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        :param logits: 模型的原始输出 (未经 sigmoid), shape [B, 1, H, W]
        :param targets: 真实标签, shape [B, 1, H, W]
        """
        # 1. 将 Logits 转换为概率
        probs = torch.sigmoid(logits)

        # 2. 展平
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        # 3. 计算交集
        intersection = (probs_flat * targets_flat).sum()

        # 4. 计算 Dice Score
        p_sum = probs_flat.sum()
        t_sum = targets_flat.sum()

        dice_score = (2. * intersection + self.smooth) / (p_sum + t_sum + self.smooth)

        # 5. 返回 Dice Loss
        return 1 - dice_score


# ==================================================================
# [新增] 组合损失 (BCE + Dice)
# ==================================================================
class CombinedLoss(nn.Module):
    """
    将 BCEWithLogitsLoss 和 DiceLoss 组合
    """

    def __init__(self, weight_bce=0.5, weight_dice=0.5, dice_smooth=1e-6):
        super(CombinedLoss, self).__init__()
        # 确保权重之和为 1
        total_weight = weight_bce + weight_dice
        self.w_bce = weight_bce / total_weight
        self.w_dice = weight_dice / total_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        print(f"CombinedLoss initialized with weights: BCE={self.w_bce:.2f}, Dice={self.w_dice:.2f}")

    def forward(self, logits, targets):
        # 计算两种损失
        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        # 返回加权和
        combined_loss = (self.w_bce * loss_bce) + (self.w_dice * loss_dice)
        return combined_loss

def calculate_metrics_and_loss(loader, model, loss_fn, device):  # <-- 修改点: 签名移除 num_cls
    """
    在验证集上评估模型，计算 Loss, 前景IoU 和 前景Dice
    (BCE 专用版本)
    """
    model.eval()  # 切换到评估模式
    loop = tqdm(loader, desc="Validation", leave=False)

    running_loss = 0.0
    smooth = 1e-6  # 防止除以0

    # <-- 修改点: 我们只累加前景指标，不再需要按类别
    total_intersection = 0.0
    total_union = 0.0
    total_dice_sum = 0.0  # (Dice的分母 P_sum + T_sum)

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)  # [B, 1, H, W], float

            # 1. 前向传播
            predictions = model(data)[0] # [B, 1, H, W], logits

            # 2. 计算Loss
            loss = loss_fn(predictions, targets)
            running_loss += loss.item()

            # 3. 计算指标
            # <-- 修改点: BCE 的指标计算逻辑
            # (1) Sigmoid: 将 logits -> 概率 (0~1)
            preds_probs = torch.sigmoid(predictions)
            # (2) Threshold: 将 概率 -> 标签 (0.0或1.0)
            preds_labels = (preds_probs > 0.5).float()

            # (3) 计算交集、并集 (targets 已经是 [B, 1, H, W], float)
            # .sum() 会在所有维度上求和
            intersection = (preds_labels * targets).sum()
            p_sum = preds_labels.sum()
            t_sum = targets.sum()

            union = (p_sum + t_sum) - intersection
            dice_sum = p_sum + t_sum

            # (4) 累加
            total_intersection += intersection
            total_union += union
            total_dice_sum += dice_sum

    # --- 最终计算 ---

    # IoU (Foreground)
    iou_foreground = (total_intersection + smooth) / (total_union + smooth)

    # Dice (Foreground)
    dice_foreground = (2. * total_intersection + smooth) / (total_dice_sum + smooth)

    # Avg Loss
    avg_loss = running_loss / len(loader)

    # 切换回训练模式
    model.train()

    # <-- 修改点: 返回值
    return avg_loss, iou_foreground, dice_foreground