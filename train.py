import os.path as osp
import os
import random
import warnings 
from importlib import import_module
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import argparse
from configparser import ConfigParser
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
from augmentation import get_preprocessing, get_training_augmentation, get_validation_augmentation

from trainer import Trainer
from utils import collate_fn, get_save_dir
from optimizer import get_optimizer, get_scheduler

import wandb

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config.getint('hyper_params', 'batch_size')
    num_epochs = config.getint('hyper_params', 'num_epochs')
    learning_rate = config.getfloat('hyper_params', 'learning_rate')
    random_seed = config.getint('hyper_params', 'random_seed')
    val_every = config.getint('hyper_params', 'val_every')
    weight_decay = config.getfloat('hyper_params','weight_decay')
    momentum = config.getfloat('hyper_params', 'momentum')
    num_workers = config.getint('hyper_params', 'num_workers')
    preprocessing = config.getboolean('hyper_params', 'preprocessing')
    #wandb config
    wandb_project = config.get('wandb', 'project')
    wandb_name = config.get('wandb', 'name')

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    dataset_path = config.get('path', 'dataset_path')

    train_path = osp.join(dataset_path, 'train.json')
    train_all_path = osp.join(dataset_path, 'train_all.json')
    val_path = osp.join(dataset_path, 'val.json')

    train_transform = get_training_augmentation()
    val_transform = get_validation_augmentation()

    encoder_name = config.get('model', 'encoder_name')
    encoder_weight = config.get('model', 'encoder_weight')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weight)

    train_dataset = CustomDataLoader(data_dir=train_path if val_every != 0 else train_all_path, mode='train', transform=train_transform, preprocessing=get_preprocessing(preprocessing_fn) if preprocessing else False)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform , preprocessing=get_preprocessing(preprocessing_fn) if preprocessing else False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn
                                           )

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn
                                         )

    architecture = config.get('model','architecture')
   
    model = getattr(import_module("segmentation_models_pytorch"),architecture) 

    model = model(
        encoder_name=encoder_name, 
        encoder_weights=encoder_weight,
        in_channels=3,
        classes=11
    )

    saved_dir = get_save_dir(config.get('path','saved_dir'))
    if not osp.isdir(saved_dir):
        os.mkdir(saved_dir)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config.get('hyper_params','optimizer'), lr=learning_rate,momentum=momentum, weight_decay=weight_decay)
    # optimizer = optimizer(params = model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    
    trainer = Trainer(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, batch_size, device)

    #wandb init
    wandb.init(entity="carry-van", project=wandb_project, name=wandb_name, config={
        "device":device, "batch_size":batch_size,"num_epochs":num_epochs, "learnig_rate":learning_rate, "preprocessing":preprocessing})
    wandb.config.encoder_name = encoder_name
    wandb.config.encoder_weight = encoder_weight
    wandb.config.architecture = architecture

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default="./configs/config.ini"
    )
    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config_dir)
   
    main(config)
