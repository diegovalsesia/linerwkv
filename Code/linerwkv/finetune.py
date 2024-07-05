import os
import time
import argparse
import json

import torch
import pytorch_lightning as pl

from config import Config
from dataset import HySpecNet11k
from model import LineRWKV
from trainer import train_callback


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log_dir', help='Tensorboard log directory')
parser.add_argument('--save_dir', default='Results', help='Trained model directory')
parser.add_argument('--ckpt_file', default='', help='Checkpoint file')
param = parser.parse_args()

model_time = time.strftime("%Y%m%d_%H%M")


# Import config
config = Config()
config.save_dir = os.path.join(param.save_dir, str(model_time))

# Import datasets and validation image

# HySpecNet11k dataset
train_dataset = HySpecNet11k(config, mode='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=4, prefetch_factor=4)
val_dataset = HySpecNet11k(config, mode='val')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=4, prefetch_factor=4)


# Create Tensorboard logger
log_dir = os.path.join(param.log_dir, model_time)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
with open(os.path.join(log_dir,'config.txt'), 'w') as cfg_file:
    cfg_file.write(json.dumps(config, default=lambda obj: obj.__dict__))
logger = pl.loggers.TensorBoardLogger(param.log_dir, name=model_time)


# Create save directory
save_dir = os.path.join(param.save_dir, model_time)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create model and train
accelerator = "gpu" if config.device=="cuda" else "cpu"
trainer = pl.Trainer(accelerator=accelerator, devices=2, logger=logger, max_epochs=8000, callbacks=[train_callback(config)])
model = LineRWKV(config)
trainer.fit(model, train_loader, val_loader, ckpt_path=param.ckpt_file)

