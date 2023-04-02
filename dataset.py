import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(
            image_dir)  # os.listdir() returns a list containing the names of the entries in the directory given by path.
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # L表示灰度图像，P表示8位彩色图像，RGB表示24位彩色图像，RGBA表示32位彩色图像，
        # CMYK表示出版用的四色图像，YCbCr表示彩色视频格式，I表示32位整型灰度图像，F表示32位浮点灰度图像
        # dtype=np.float32表示将数据类型转换为32位浮点型, 因为在训练时，需要将数据转换为tensor，而tensor只能接受float32类型的数据
        mask[mask == 255.0] = 1.0  # 因为我们要用Sigmod函数，所以将mask中的255转换为1
        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            # 这一行是对原始图像和掩码同时进行相同的图像增强操作，比如旋转或平移，从而得到一对新的图像和掩码。
            # 这样做的目的是保持图像和掩码之间的对应关系，避免目标区域被改变或丢失
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
