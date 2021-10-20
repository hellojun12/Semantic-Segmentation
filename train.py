import os.path as osp
import os
import random
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataset
from utils import label_accuracy_score, add_hist
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from dataset import CustomDataLoader
from transformer import get_preprocessing, get_training_augmentation, get_validation_augmentation

from trainer import Trainer
from utils import collate_fn

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    num_epochs = 1
    learning_rate = 0.0001
    random_seed = 21
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    dataset_path = './input/data'
    anns_file_path = dataset_path + '/' + 'train_all.json'

    train_path = osp.join(dataset_path, 'train.json')
    val_path = osp.join(dataset_path, 'val.json')
    test_path = osp.join(dataset_path, 'test.json')

    train_transform = A.Compose([ ToTensorV2()])
    val_transform = A.Compose([ ToTensorV2()])
    test_transform = A.Compose([ ToTensorV2()])
    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet34", 'imagenet')

    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=get_validation_augmentation() , preprocessing=get_preprocessing(preprocessing_fn))
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           collate_fn=collate_fn
                                           )

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         collate_fn=collate_fn
                                         )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          collate_fn=collate_fn)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34", 
        encoder_weights="imagenet",
        in_channels=3,
        classes=11
    )

    val_every = 1
    saved_dir = './saved'
    if not osp.isdir(saved_dir):
        os.mkdir(saved_dir)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)
    
    trainer = Trainer(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device)
    trainer.train()


if __name__ == '__main__':
    main()
