import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import load_checkpoint, save_checkpoint, getloaders, check_accuracy, save_predictions_as_imgs

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    # 对于一个可迭代对象，tqdm(iterable)返回一个迭代器，该迭代器包装了原始的可迭代对象，同时在每次迭代时打印一个进度条。
    # tqdm是一个快速，可扩展的Python进度条，可以在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器tqdm(iterator)。

    for batch_idx, (data, targets) in enumerate(loop):
        # enumerate(loop)返回批量索引、批量数据、批量标签
        # data is a tensor of shape (batch_size, 3, input_height, input_width)
        # targets is a tensor of shape (batch_size, 1, input_height, input_width)
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)
        # TODO: Why do we need to unsqueeze the targets? why float()?

        # forward
        with torch.cuda.amp.autocast():  # 作用是开启自动混合精度训练 TODO：Learn more about this
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # 将损失值乘一个缩放因子，然后反向传播
        scaler.step(optimizer)  # 将缩放后的梯度转换为原始的梯度，然后用优化器对参数进行更新
        scaler.update()  # 更新缩放因子
        # TODO: Learn more about scaler

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),  # 旋转角度范围
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.VerticalFlip(p=0.1),  # 垂直翻转
            A.Normalize(  # 归一化
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )  # just like train_transforms

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = getloaders(
        TRAIN_IMG_DIR,  # 训练集图片路径
        TRAIN_MASK_DIR,  # 训练集标签路径
        VAL_IMG_DIR,  # 验证集图片路径
        VAL_MASK_DIR,  # 验证集标签路径
        BATCH_SIZE,  # 批大小
        train_transform,  # 训练集数据增强
        val_transforms,  # 验证集数据增强
        NUM_WORKERS,  # 加载数据的进程数
        PIN_MEMORY,  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
    )

    scaler = torch.cuda.amp.GradScaler()
    # 作用是开启自动混合精度训练 TODO：Learn more about this
    # 表示创建一个梯度缩放器（GradScaler）的对象，用于支持混合精度训练，即在训练过程中使用不同的数值精度，
    # 如32位浮点型（torch.FloatTensor）和16位浮点型（torch.HalfTensor）
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
