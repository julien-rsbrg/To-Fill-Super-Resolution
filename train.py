#################### USE OTHER SCRIPTS ####################
from SRGAN import SRGAN, get_discriminator, get_generator, get_VGG19
from loadAndProcess import train_ds, val_ds
from utils import compute_psnr, compute_ssim, save_images, path_join, opt

#################### LIBRAIRIES ####################
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import pathlib
import os

#################### CONFIG ####################
import yaml
from yaml.loader import SafeLoader


with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

# print("config:\n", config)

#################### wandb init ####################
import wandb

wandb.init(project="Super-resolution", entity=config["WANDB_ENTITY"])
wandb.config = config
config["WANDB_ITERATOR"] += 1
wandb.run.name = "experience_n"+str(config["WANDB_ITERATOR"])
wandb.run.save()

with open('config.yaml', "w") as f:
    yaml.dump(config, f, sort_keys=False, default_flow_style=False)

#################### TRAINING ####################

epochs = config["EPOCHS"]
dst_folder = config["DST_FOLDER_SAVE_RESULTS"]

generator = get_generator(config["LOW_RESOLUTION_SHAPE"])
# generator.summary()
discriminator = get_discriminator(config["HIGH_RESOLUTION_SHAPE"])
# discriminator.summary()
fe_model = get_VGG19(config["HIGH_RESOLUTION_SHAPE"])
# fe_model.summary()
adv_model = SRGAN(generator=generator,
                  discriminator=discriminator, feature_extractor=fe_model)
adv_model.compile(loss=["binary_crossentropy", "mse"],
                  loss_weight=[1e-3, 1], optimizer=opt)

dst_folder = config["DST_FOLDER_SAVE_RESULTS"]
adv_model.fit(train_ds, epochs=epochs, validation_dataset=val_ds,
              steps_per_epoch=config["STEPS_PER_EPOCH"], save_val_period=config["SAVE_VAL_PERIOD"], dst_folder=dst_folder, use_wandb=True)
