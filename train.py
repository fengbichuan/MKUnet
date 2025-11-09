
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv  # [新增] 导入 CSV 库

from thop import profile
from tqdm import tqdm

from MKUnet import MK_UNet
# 导入你本地的文件
from dataload import get_loaders
from metrics import calculate_metrics_and_loss, CombinedLoss

# --- 1. 配置参数 ---
# !! 修改为你自己的路径 !!
DATA_PATH = r"D:\controller\dataset\9-isic2018"

# 超参数
LEARNING_RATE = 0.0003
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 300
NUM_WORKERS = 8
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
NUM_CLASSES = 1  # <-- 修改点: 二分类 (BCE) 模式下, 输出通道为 1
SAVE_PATH = "LDCA+CConv"
early_stop_patience = 20
early_stop_counter = 0
stage_channels = [16, 32, 64, 96, 160]
# channels = [4,8,16,24,32] for MK_UNet-T
# channels = [8,16,32,48,80] for MK_UNet-S
# channels = [16,32,64,96,160] for MK_UNet
# channels = [32,64,128,192,320] for MK_UNet-M
# channels = [64,128,256,384,512] for MK_UNet-L

def train_fn(loader, model, optimizer, loss_fn, device):
    """
    训练一个 Epoch
    """
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0

    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # 1. 前向传播
        predictions_list = model(data)  # model(data) 返回 [p4]

        # --- 关键修复 ---
        # 确保这里只有一个 [0]
        predictions = predictions_list[0]
        # --- 修复结束 ---

        # predictions 现在的形状应该是 [8, 1, 256, 256]

        # 2. 计算Loss (这是你报错的第60行)
        loss = loss_fn(predictions, targets)

        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. 更新进度条
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    print(f"Train Epoch Loss: {avg_loss:.4f}")

    return avg_loss


# [新增] 保存结果到 CSV 文件的函数
def save_results_to_csv(filepath, header, data_rows):
    """
    将结果列表保存到 CSV 文件
    :param filepath: CSV 文件的保存路径
    :param header: CSV 文件的表头 (list of strings)
    :param data_rows: 包含指标数据的列表 (list of lists)
    """
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(header)
            # 写入所有数据行
            writer.writerows(data_rows)
        print(f"Metrics successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving CSV to {filepath}: {e}")


def main():
    global early_stop_counter
    print(f"Using device: {DEVICE}")

    # --- 2. Dataloaders ---
    train_loader, val_loader = get_loaders(
        DATA_PATH,
        BATCH_SIZE,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_WORKERS,
        PIN_MEMORY
    )


    model = MK_UNet(num_classes=1, in_channels=3, channels=stage_channels, depths=[1, 1, 1, 1, 1],
                    kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3).to(torch.device('cuda:0'))





    print("\n" + "---" * 15)
    print("Calculating model parameters and FLOPs...")
    trainable_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total Trainable Params (M): {trainable_params_m:.2f}M")
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).to(DEVICE)
    flops, params_thop = profile(model, inputs=(dummy_input,), verbose=False)
    g_flops = flops / 1e9
    print(f"FLOPs (G): {g_flops:.2f}G")
    print("---" * 15 + "\n")
    # ==================================================================

    # Loss 函数
    loss_fn = CombinedLoss() # <-- 修改点: 更换损失函数

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # (可选) 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.1
    )

    # --- 4. 训练循环 ---
    best_iou_fg = -1.0
    best_dice_fg = -1.0

    # [新增] 用于存储 CSV 结果的列表和表头
    results_list = []
    csv_header = ["Epoch", "Train Loss", "Val Loss", "Val IoU (FG)", "Val Dice (FG)"]
    # [新增] 定义 CSV 文件的保存路径 (基于模型保存路径)
    csv_save_path = f"{SAVE_PATH}.csv"

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # 训练 [修改] 接收 train_loss
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)

        # <-- 修改点: 新的评估函数不再需要 num_cls，且返回的直接是前景指标
        val_loss, IoU_foreground, Dice_foreground = calculate_metrics_and_loss(
            val_loader, model, loss_fn, DEVICE
        )

        print(f"\nValidation Results:")
        print(f"  Avg Loss: {val_loss:.4f}")
        print(f"  IoU (Foreground):     {IoU_foreground:.4f}")
        print(f"  Dice (Foreground):    {Dice_foreground:.4f}")

        # [新增] 将本轮次的结果添加到列表中
        epoch_data = [epoch + 1, train_loss, val_loss, IoU_foreground, Dice_foreground]
        results_list.append(epoch_data)

        # 1. (可选) 更新学习率调度器 (基于 Dice)
        scheduler.step(Dice_foreground)

        # 2. (可选) 追踪最佳 IoU (仅用于最后打印)
        if IoU_foreground > best_iou_fg:
            best_iou_fg = IoU_foreground
            print(f"==> New best Foreground IoU: {IoU_foreground:.4f}")

        # 3. 修正后的早停和模型保存逻辑 (基于 Dice)
        if Dice_foreground > best_dice_fg:
            best_dice_fg = Dice_foreground
            print(f"==> New best Dice found! Saving model... Foreground Dice: {Dice_foreground:.4f}")
            torch.save(model.state_dict(), SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter} / {early_stop_patience}")

        # 4. 检查是否触发早停
        if early_stop_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered: Dice score did not improve for {early_stop_patience} epochs.")
            break

    print("\nTraining finished.")
    print(f"Best validation Foreground IoU: {best_iou_fg:.4f}")
    print(f"Best validation Foreground Dice: {best_dice_fg:.4f}")
    print(f"Best model saved to {SAVE_PATH}")

    # [新增] 在训练结束后，调用函数保存 CSV
    save_results_to_csv(csv_save_path, csv_header, results_list)


if __name__ == "__main__":
    main()